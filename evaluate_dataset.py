"""
QnA 데이터셋 전체 평가 스크립트
=================================

1차년도_QnA_데이터셋_평가.xlsx의 '연구자 평가' 시트에서
- 질문, 정답(ground_truth), LLM 답변(파인튜닝 후/전), TF 점수
를 읽어서 C2 사후 필터링 결과와 비교합니다.

평가 지표:
    - Precision, Recall, F1-score
    - TF 점수(O/X) vs C2 판정 비교
"""

import json
import os
import sys
import time
from dataclasses import dataclass

import openpyxl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hallucination_filter_groq import HallucinationFilter

# ── 설정 ────────────────────────────────────────────────────────────────

XLSX_PATH    = "1차년도 QnA 데이터셋 평가.xlsx"
AGROVOC_TTL  = "agrovocSubOntology.ttl"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

# 평가할 컬럼 선택: "finetune" 또는 "zeroshot"
EVAL_TARGET = "finetune"   # 파인튜닝 후 추론 결과

# API rate limit 방지용 대기 시간 (초)
SLEEP_BETWEEN = 2.0


# ── 데이터 로드 ──────────────────────────────────────────────────────────

@dataclass
class QnAItem:
    idx:          int
    question:     str
    ground_truth: str
    llm_answer:   str
    tf_label:     bool   # True = O(정답), False = X(오답)


def load_dataset(xlsx_path: str, target: str = "finetune") -> list[QnAItem]:
    """
    엑셀에서 QnA 데이터셋 로드

    컬럼 구조 (연구자 평가 시트):
        0: 번호
        1: 질문
        2: 정답
        3: 파인튜닝 후 추론 결과
        4: TF 점수 (파인튜닝 후)
        5: 의견
        6: 파인튜닝 전 추론 결과
        7: 보고서 제출용 점수
        8: 보고서 제출용 의견
    """
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb["연구자 평가"]

    items = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row[0]:  # 번호 없으면 스킵
            continue

        idx          = int(row[0])
        question     = str(row[1] or "").strip()
        ground_truth = str(row[2] or "").strip()

        if target == "finetune":
            llm_answer = str(row[3] or "").strip()
            tf_raw     = str(row[4] or "").strip().upper()
        else:  # zeroshot
            llm_answer = str(row[6] or "").strip()
            tf_raw     = str(row[7] or "").strip().upper()

        if not llm_answer or not ground_truth:
            continue

        # TF 점수: O → hallucination 없음(False), X → hallucination 있음(True)
        tf_label = (tf_raw == "X")

        items.append(QnAItem(
            idx=idx,
            question=question,
            ground_truth=ground_truth,
            llm_answer=llm_answer,
            tf_label=tf_label,
        ))

    print(f"[load_dataset] {len(items)}개 항목 로드 완료 (target={target})")
    return items


# ── 평가 지표 계산 ───────────────────────────────────────────────────────

def compute_metrics(
    labels: list[bool],
    predictions: list[bool],
) -> dict:
    """
    Precision / Recall / F1 계산
    Positive = hallucination 있음 (True)
    """
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
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "accuracy":  accuracy,
    }


# ── 메인 평가 루프 ───────────────────────────────────────────────────────

def run_evaluation():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 데이터셋 로드
    dataset = load_dataset(
        os.path.join(base_dir, XLSX_PATH),
        target=EVAL_TARGET,
    )

    # 필터 초기화
    f = HallucinationFilter(
        agrovoc_ttl_path=os.path.join(base_dir, AGROVOC_TTL),
        groq_api_key=GROQ_API_KEY,
    )

    labels      = []
    predictions = []
    results     = []

    print(f"\n{'='*60}")
    print(f"평가 시작: {len(dataset)}개 항목 ({EVAL_TARGET})")
    print(f"{'='*60}\n")

    for item in dataset:
        print(f"\n[Q{item.idx}] {item.question[:50]}...")
        print(f"  정답 레이블: {'❌ hallucination' if item.tf_label else '✅ 정확'}")

        try:
            result = f.check(
                final_answer=item.llm_answer,
                ground_truth=item.ground_truth,
            )

            predicted = result.is_hallucinated
            correct   = predicted == item.tf_label
            match_str = "✅ 정확" if correct else "❌ 오류"

            print(f"  C2 판정:    {'❌ hallucination' if predicted else '✅ 정확'} "
                  f"(신뢰도: {result.confidence:.2f})")
            print(f"  판정 결과:  {match_str}")
            print(f"  요약:       {result.summary[:80]}")

            labels.append(item.tf_label)
            predictions.append(predicted)
            results.append({
                "idx":           item.idx,
                "question":      item.question[:80],
                "tf_label":      item.tf_label,
                "predicted":     predicted,
                "correct":       correct,
                "confidence":    result.confidence,
                "summary":       result.summary,
                "flagged_count": len(result.flagged_triples),
            })

        except Exception as e:
            print(f"  오류 발생: {e}")
            # 오류 시 정답 레이블로 fallback
            labels.append(item.tf_label)
            predictions.append(not item.tf_label)  # 틀린 것으로 처리
            results.append({
                "idx":       item.idx,
                "question":  item.question[:80],
                "tf_label":  item.tf_label,
                "predicted": not item.tf_label,
                "correct":   False,
                "confidence": 0.0,
                "summary":   f"오류: {str(e)}",
                "flagged_count": 0,
            })

        # API rate limit 방지
        time.sleep(SLEEP_BETWEEN)

    # ── 최종 결과 출력 ──────────────────────────────────────────────────
    metrics = compute_metrics(labels, predictions)

    print(f"\n{'='*60}")
    print("📊 최종 평가 결과")
    print(f"{'='*60}")
    print(f"평가 항목 수: {len(labels)}")
    print(f"정확도 (Accuracy):  {metrics['accuracy']:.3f} "
          f"({int(metrics['accuracy']*len(labels))}/{len(labels)})")
    print(f"정밀도 (Precision): {metrics['precision']:.3f}")
    print(f"재현율 (Recall):    {metrics['recall']:.3f}")
    print(f"F1-score:           {metrics['f1']:.3f}")
    print(f"\n혼동 행렬:")
    print(f"  TP (hallucination 정탐): {metrics['tp']}")
    print(f"  FP (hallucination 오탐): {metrics['fp']}")
    print(f"  FN (hallucination 미탐): {metrics['fn']}")
    print(f"  TN (정확 정탐):          {metrics['tn']}")

    # ── 결과 저장 ────────────────────────────────────────────────────────
    output_path = os.path.join(base_dir, f"eval_results_{EVAL_TARGET}.json")
    with open(output_path, "w", encoding="utf-8") as out:
        json.dump({
            "eval_target": EVAL_TARGET,
            "metrics":     metrics,
            "results":     results,
        }, out, ensure_ascii=False, indent=2)
    print(f"\n결과 저장 완료: {output_path}")

    return metrics, results


if __name__ == "__main__":
    run_evaluation()
