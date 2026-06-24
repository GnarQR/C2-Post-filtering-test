"""
전체 평가 파이프라인
====================
미리 추출된 triple JSON을 사용해서 LLM 호출 없이 평가합니다.

사전 준비:
    python build_kg.py          → domain_kg_flat.json (정답 triple)
    python build_llm_kg.py      → llm_kg_{target}.json (LLM 답변 triple)

이 스크립트는 저장된 triple로만 NLI + 교정을 수행합니다.
LLM 추가 호출 없음.

출력:
    - Accuracy (교정 전 vs 후)
    - ROUGE-1
    - eval_correction_{target}.json

사용법:
    python evaluate_correction.py --target finetune
    python evaluate_correction.py --target zeroshot --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hallucination_corrector import verify_triples, correct_triple, apply_correction
from triple_extractor import Triple
from llm_client import GroqClient, GeminiClient

SLEEP_SEC = 1.0


# ── 데이터 로드 ──────────────────────────────────────────────────────────

def load_llm_kg(path: str) -> dict[int, dict]:
    """llm_kg_{target}.json → {idx: item} 딕셔너리"""
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    return {item["idx"]: item for item in items}


def load_domain_kg(path: str) -> dict[int, list[dict]]:
    """domain_kg.json → {idx: [triple, ...]} 딕셔너리"""
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    return {item["idx"]: item["triples"] for item in items}


# ── 평가 지표 ────────────────────────────────────────────────────────────

def compute_metrics(labels: list[bool], predictions: list[bool]) -> dict:
    tp = sum(1 for l, p in zip(labels, predictions) if l and p)
    fp = sum(1 for l, p in zip(labels, predictions) if not l and p)
    fn = sum(1 for l, p in zip(labels, predictions) if l and not p)
    tn = sum(1 for l, p in zip(labels, predictions) if not l and not p)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / len(labels) if labels else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 3),
        "recall":    round(recall, 3),
        "f1":        round(f1, 3),
        "accuracy":  round(accuracy, 3),
    }


def compute_rouge1(reference: str, hypothesis: str) -> float:
    ref_tokens = set(reference.split())
    hyp_tokens = hypothesis.split()
    if not hyp_tokens or not ref_tokens:
        return 0.0
    overlap   = sum(1 for t in hyp_tokens if t in ref_tokens)
    precision = overlap / len(hyp_tokens)
    recall    = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 3)


# ── 메인 ────────────────────────────────────────────────────────────────

def run_evaluation(
    target:      str,
    client_type: str,
    api_key:     str,
    base_dir:    str,
    dry_run:     bool = False,
):
    # 파일 경로
    llm_kg_path    = os.path.join(base_dir, f"llm_kg_{target}.json")
    domain_kg_path = os.path.join(base_dir, "domain_kg.json")

    # 파일 존재 확인
    for path in [llm_kg_path, domain_kg_path]:
        if not os.path.exists(path):
            print(f"[오류] 파일 없음: {path}")
            print("먼저 build_kg.py와 build_llm_kg.py를 실행하세요.")
            return

    # 데이터 로드 (LLM 호출 없음)
    llm_kg    = load_llm_kg(llm_kg_path)
    domain_kg = load_domain_kg(domain_kg_path)
    print(f"[로드] LLM KG: {len(llm_kg)}개, Domain KG: {len(domain_kg)}개")

    # 교정에만 LLM 필요 (triple 추출은 이미 완료)
    if client_type == "groq":
        client = GroqClient(api_key=api_key)
    else:
        client = GeminiClient(api_key=api_key)

    idxs = sorted(llm_kg.keys())
    if dry_run:
        idxs = idxs[:3]
        print(f"[dry-run] 처음 {len(idxs)}개만 실행")

    labels, pred_before, pred_after = [], [], []
    rouge_scores = []
    results = []

    print(f"\n{'='*60}")
    print(f"평가 시작: {len(idxs)}개 항목 ({target})")
    print(f"{'='*60}\n")

    for idx in idxs:
        item       = llm_kg[idx]
        gt_triples = domain_kg.get(idx, [])

        print(f"\n[Q{idx}] {item['question'][:50]}...")
        print(f"  레이블: {'❌ hallucination' if item['tf_label'] else '✅ 정확'}")

        # LLM 답변 triple (이미 저장된 것 사용)
        llm_triples = [
            Triple(subject=t["subject"], predicate=t["predicate"], obj=t["object"])
            for t in item["triples"]
        ]

        # ground_truth 문장 (정답 triple → 문장으로 합치기)
        gt_sentences = " ".join(
            f"{t['subject']} {t['predicate']} {t['object']}."
            for t in gt_triples
        ) if gt_triples else item["ground_truth"]

        if not llm_triples:
            print("  triple 없음 — 스킵")
            labels.append(item["tf_label"])
            pred_before.append(False)
            pred_after.append(False)
            rouge_scores.append(1.0)
            continue

        try:
            # Step 1: NLI 검증 (LLM 호출 없음)
            verifications  = verify_triples(llm_triples, gt_sentences)
            flagged        = [v for v in verifications if v.is_hallucinated]
            is_hall_before = len(flagged) > 0

            # Step 2: 교정 (틀린 triple만 LLM으로 교정)
            # 교정된 triple을 추적해서 재검증에 재사용
            corrected_answer  = item["llm_answer"]
            corrected_triples = list(llm_triples)  # 복사본 시작

            if flagged:
                for tv in flagged:
                    new_triple = correct_triple(tv.triple, gt_sentences, client)
                    if new_triple and new_triple.to_sentence() != tv.triple.to_sentence():
                        corrected_answer = apply_correction(
                            corrected_answer, tv.triple, new_triple, client
                        )
                        # triple 목록도 교체
                        corrected_triples = [
                            new_triple if t.to_sentence() == tv.triple.to_sentence() else t
                            for t in corrected_triples
                        ]
                time.sleep(SLEEP_SEC)

            # Step 3: 교정 후 재검증
            # 교정된 triple로 NLI 재실행 (LLM 추가 호출 없음)
            after_verifications = verify_triples(corrected_triples, gt_sentences)
            is_hall_after       = any(v.is_hallucinated for v in after_verifications)

            rouge          = compute_rouge1(item["llm_answer"], corrected_answer)
            correct_before = is_hall_before == item["tf_label"]
            correct_after  = is_hall_after  == item["tf_label"]

            # ── 콘솔 출력 ──────────────────────────────────────────────
            print(f"  LLM triple: {len(llm_triples)}개, flagged: {len(flagged)}개")
            print(f"  교정 전: {'❌' if is_hall_before else '✅'} ({'정확' if correct_before else '오류'})")
            print(f"  교정 후: {'❌' if is_hall_after  else '✅'} ({'정확' if correct_after  else '오류'})")
            print(f"  ROUGE-1: {rouge:.3f}")

            # 교정 전/후 텍스트 비교 출력
            if corrected_answer != item["llm_answer"]:
                print(f"\n  [원본 답변]")
                print(f"  {item['llm_answer'][:200]}")
                print(f"\n  [교정된 답변]")
                print(f"  {corrected_answer[:200]}")

                # 어떤 triple이 교정됐는지 출력
                print(f"\n  [교정된 triple]")
                for tv in flagged:
                    print(f"    Before: {tv.triple}")
                for t in corrected_triples:
                    orig = next((tv.triple for tv in flagged
                                 if tv.triple.subject == t.subject), None)
                    if orig and orig.to_sentence() != t.to_sentence():
                        print(f"    After:  {t}")
            else:
                print(f"  교정 없음 (hallucination 미탐지 또는 교정 불필요)")

            labels.append(item["tf_label"])
            pred_before.append(is_hall_before)
            pred_after.append(is_hall_after)
            rouge_scores.append(rouge)

            results.append({
                "idx":             idx,
                "question":        item["question"][:80],
                "tf_label":        item["tf_label"],
                "pred_before":     is_hall_before,
                "pred_after":      is_hall_after,
                "correct_before":  correct_before,
                "correct_after":   correct_after,
                "flagged_count":   len(flagged),
                "rouge1":          rouge,
                "original_answer": item["llm_answer"][:200],
                "corrected_answer": corrected_answer[:200],
            })

        except Exception as e:
            print(f"  오류: {e}")
            labels.append(item["tf_label"])
            pred_before.append(False)
            pred_after.append(False)
            rouge_scores.append(1.0)
            results.append({"idx": idx, "error": str(e)})

    # ── 최종 결과 ────────────────────────────────────────────────────────
    m_before  = compute_metrics(labels, pred_before)
    m_after   = compute_metrics(labels, pred_after)
    avg_rouge = round(sum(rouge_scores) / len(rouge_scores), 3) if rouge_scores else 0.0

    print(f"\n{'='*60}")
    print(f"📊 최종 평가 결과 ({target})")
    print(f"{'='*60}")
    print(f"\n[교정 전]  Accuracy: {m_before['accuracy']:.3f}  "
          f"P: {m_before['precision']:.3f}  R: {m_before['recall']:.3f}  F1: {m_before['f1']:.3f}")
    print(f"[교정 후]  Accuracy: {m_after['accuracy']:.3f}  "
          f"P: {m_after['precision']:.3f}  R: {m_after['recall']:.3f}  F1: {m_after['f1']:.3f}")

    delta = m_after["accuracy"] - m_before["accuracy"]
    print(f"\n  Accuracy 변화: {delta:+.3f} "
          f"({'↑ 개선' if delta > 0 else '↓ 악화' if delta < 0 else '변화 없음'})")
    print(f"  평균 ROUGE-1 (원본 vs 교정본): {avg_rouge:.3f}")

    if not dry_run:
        out_path = os.path.join(base_dir, f"eval_correction_{target}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "target":         target,
                "metrics_before": m_before,
                "metrics_after":  m_after,
                "avg_rouge":      avg_rouge,
                "results":        results,
            }, f, ensure_ascii=False, indent=2)
        print(f"\n결과 저장: {out_path}")


# ── CLI ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",     choices=["finetune", "zeroshot"], default="finetune")
    parser.add_argument("--client",     choices=["groq", "gemini"],       default="groq")
    parser.add_argument("--groq-key",   default=os.getenv("GROQ_API_KEY", ""))
    parser.add_argument("--gemini-key", default=os.getenv("GEMINI_API_KEY", ""))
    parser.add_argument("--dry-run",    action="store_true")
    args = parser.parse_args()

    api_key  = args.groq_key if args.client == "groq" else args.gemini_key
    base_dir = os.path.dirname(os.path.abspath(__file__))

    run_evaluation(
        target=args.target,
        client_type=args.client,
        api_key=api_key,
        base_dir=base_dir,
        dry_run=args.dry_run,
    )