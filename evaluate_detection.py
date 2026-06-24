"""
Hallucination 탐지 전용 평가 스크립트
======================================
LLM 호출 없이 저장된 triple + NLI만으로 평가합니다.

사전 준비:
    python build_kg.py       → domain_kg.json
    python build_llm_kg.py   → llm_kg_{target}.json

사용법:
    python evaluate_detection.py --target finetune
    python evaluate_detection.py --target zeroshot
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hallucination_corrector import verify_triples
from triple_extractor import Triple


# ── 데이터 로드 ──────────────────────────────────────────────────────────

def load_llm_kg(path: str) -> dict[int, dict]:
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    return {item["idx"]: item for item in items}


def load_domain_kg(path: str) -> dict[int, list[dict]]:
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


# ── 메인 ────────────────────────────────────────────────────────────────

def run_detection(
    target:   str,
    base_dir: str,
    dry_run:  bool = False,
):
    llm_kg_path    = os.path.join(base_dir, f"llm_kg_{target}.json")
    domain_kg_path = os.path.join(base_dir, "domain_kg.json")

    for path in [llm_kg_path, domain_kg_path]:
        if not os.path.exists(path):
            print(f"[오류] 파일 없음: {path}")
            return

    llm_kg    = load_llm_kg(llm_kg_path)
    domain_kg = load_domain_kg(domain_kg_path)
    print(f"[로드] LLM KG: {len(llm_kg)}개, Domain KG: {len(domain_kg)}개")

    idxs = sorted(llm_kg.keys())
    if dry_run:
        idxs = idxs[:5]
        print(f"[dry-run] 처음 {len(idxs)}개만 실행")

    labels      = []
    predictions = []
    results     = []

    print(f"\n{'='*60}")
    print(f"탐지 평가 시작: {len(idxs)}개 항목 ({target})")
    print(f"{'='*60}\n")

    for idx in idxs:
        item       = llm_kg[idx]
        gt_triples = domain_kg.get(idx, [])

        tf_label = item["tf_label"]

        # LLM 답변 triple
        llm_triples = [
            Triple(subject=t["subject"], predicate=t["predicate"], obj=t["object"])
            for t in item["triples"]
        ]

        # 정답 triple → context 문장
        # 질문과 관련된 정답 triple만 사용 (전체 사용 시 노이즈 많음)
        gt_sentences = " ".join(
            f"{t['subject']} {t['predicate']} {t['object']}."
            for t in gt_triples
        ) if gt_triples else item.get("ground_truth", "")

        if not llm_triples:
            print(f"[Q{idx}] triple 없음 → 스킵")
            labels.append(tf_label)
            predictions.append(False)
            continue

        try:
            verifications = verify_triples(llm_triples, gt_sentences)
            flagged       = [v for v in verifications if v.is_hallucinated]
            is_hall       = len(flagged) > 0
            correct       = is_hall == tf_label

            labels.append(tf_label)
            predictions.append(is_hall)

            status = "✅ 정확" if correct else "❌ 오류"
            print(f"[Q{idx}] 레이블: {'❌ hall' if tf_label else '✅ ok'} | "
                  f"예측: {'❌ hall' if is_hall else '✅ ok'} | "
                  f"{status} | "
                  f"triple {len(llm_triples)}개, flagged {len(flagged)}개")

            # flagged triple 상세 출력 (이유 + 관련 정답 triple)
            for v in flagged:
                if v.nli_score >= 0.85:
                    reason = "정답과 모순"
                elif v.nli_score >= 0.70:
                    reason = "정답과 불일치"
                else:
                    reason = "정답에 없는 내용"
                print(f"  ⚠ [{reason}] {v.triple} (score: {v.nli_score:.2f})")
                # subject 기준으로 관련 정답 triple 찾기
                subj = v.triple.subject.strip()
                related = [
                    t for t in gt_triples
                    if subj[:4] in t["subject"] or t["subject"][:4] in subj
                ]
                if related:
                    print(f"     → 관련 정답:")
                    for r in related[:2]:
                        print(f"       ({r['subject']}, {r['predicate']}, {r['object']})")
                else:
                    print(f"     → 관련 정답 없음 (정답 KG에 해당 개념 미등재)")

            results.append({
                "idx":           idx,
                "question":      item["question"][:60],
                "tf_label":      tf_label,
                "predicted":     is_hall,
                "correct":       correct,
                "triple_count":  len(llm_triples),
                "flagged_count": len(flagged),
                "flagged_triples": [
                    {"triple": str(v.triple), "score": round(v.nli_score, 3)}
                    for v in flagged
                ],
            })

        except Exception as e:
            print(f"[Q{idx}] 오류: {e}")
            labels.append(tf_label)
            predictions.append(False)
            results.append({"idx": idx, "error": str(e)})

    # ── 최종 결과 ────────────────────────────────────────────────────────
    m = compute_metrics(labels, predictions)

    print(f"\n{'='*60}")
    print(f"📊 탐지 평가 결과 ({target})")
    print(f"{'='*60}")
    print(f"평가 항목: {len(labels)}개")
    print(f"Accuracy : {m['accuracy']:.3f}  ({m['tp']+m['tn']}/{len(labels)})")
    print(f"Precision: {m['precision']:.3f}")
    print(f"Recall   : {m['recall']:.3f}")
    print(f"F1       : {m['f1']:.3f}")
    print(f"\n혼동 행렬:")
    print(f"  TP (hall 정탐): {m['tp']}")
    print(f"  FP (hall 오탐): {m['fp']}")
    print(f"  FN (hall 미탐): {m['fn']}")
    print(f"  TN (정확 정탐): {m['tn']}")

    # 오탐/미탐 분석
    fp_items = [r for r in results if not r.get("error") and
                not r["tf_label"] and r["predicted"]]
    fn_items = [r for r in results if not r.get("error") and
                r["tf_label"] and not r["predicted"]]

    if fn_items:
        print(f"\n[미탐 (FN) {len(fn_items)}개 — hallucination인데 못 잡은 것]")
        for r in fn_items:
            print(f"  Q{r['idx']}: {r['question']}")

    if fp_items:
        print(f"\n[오탐 (FP) {len(fp_items)}개 — 정확한데 hallucination으로 잡은 것]")
        for r in fp_items:
            print(f"  Q{r['idx']}: {r['question']}")

    # 저장
    if not dry_run:
        out_path = os.path.join(base_dir, f"eval_detection_{target}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"target": target, "metrics": m, "results": results},
                      f, ensure_ascii=False, indent=2)
        print(f"\n결과 저장: {out_path}")


# ── CLI ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",  choices=["finetune", "zeroshot"], default="finetune")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    run_detection(target=args.target, base_dir=base_dir, dry_run=args.dry_run)