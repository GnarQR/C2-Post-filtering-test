"""
LLM 답변 Triple 추출 스크립트
==============================
QnA 데이터셋의 LLM 답변(파인튜닝 전/후)에서 triple을 추출해서
JSON으로 저장합니다.

build_kg.py가 전문가 정답 → domain_kg_flat.json 을 만든다면,
이 스크립트는 LLM 답변 → llm_kg_flat_{target}.json 을 만듭니다.

출력:
    llm_kg_{target}.json      - 질문별 triple 목록
    llm_kg_flat_{target}.json - flat triple 목록 (비교 시 사용)

사용법:
    python build_llm_kg.py --target finetune --client groq
    python build_llm_kg.py --target zeroshot --client gemini
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import openpyxl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_client import BaseLLMClient, GroqClient, GeminiClient
from triple_extractor import extract_triples

SLEEP_SEC = 2.0


# ── 데이터 로드 ──────────────────────────────────────────────────────────

def load_llm_answers(xlsx_path: str, target: str = "finetune") -> list[dict]:
    """
    엑셀에서 질문 + LLM 답변 + 전문가 레이블 추출

    컬럼 구조:
        0: 번호
        1: 질문
        2: 정답 (전문가)
        3: 파인튜닝 후 추론 결과
        4: TF 점수 (파인튜닝 후)
        6: 파인튜닝 전 추론 결과
        7: TF 점수 (파인튜닝 전)
    """
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb["연구자 평가"]

    items = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row[0]:
            continue

        idx          = int(row[0])
        question     = str(row[1] or "").strip()
        ground_truth = str(row[2] or "").strip()

        if target == "finetune":
            llm_answer = str(row[3] or "").strip()
            tf_raw     = str(row[4] or "").strip().upper()
        else:
            llm_answer = str(row[6] or "").strip()
            tf_raw     = str(row[7] or "").strip().upper()

        if not llm_answer:
            continue

        tf_label = (tf_raw == "X")  # X = hallucination 있음

        items.append({
            "idx":          idx,
            "question":     question,
            "ground_truth": ground_truth,
            "llm_answer":   llm_answer,
            "tf_label":     tf_label,
        })

    print(f"[load_llm_answers] {len(items)}개 로드 (target={target})")
    return items


# ── Triple 추출 ──────────────────────────────────────────────────────────

def build_llm_kg(
    items:  list[dict],
    client: BaseLLMClient,
) -> tuple[list[dict], list[dict]]:
    """
    LLM 답변 전체에서 triple 추출

    Returns
    -------
    kg : list[dict]
        질문별 triple 목록 (tf_label 포함)

    flat : list[dict]
        전체 triple flat 목록
    """
    kg   = []
    flat = []

    for item in items:
        print(f"\n[Q{item['idx']}] {item['question'][:50]}...")
        print(f"  레이블: {'❌ hallucination' if item['tf_label'] else '✅ 정확'}")
        print(f"  답변: {item['llm_answer'][:80]}...")

        triples = extract_triples(
            text=item["llm_answer"],
            client=client,
            verbose=True,
        )

        kg.append({
            "idx":          item["idx"],
            "question":     item["question"],
            "ground_truth": item["ground_truth"],
            "llm_answer":   item["llm_answer"],
            "tf_label":     item["tf_label"],
            "triples":      [t.to_dict() for t in triples],
        })

        for t in triples:
            flat.append({
                "idx":      item["idx"],
                "question": item["question"],
                "tf_label": item["tf_label"],
                **t.to_dict(),
            })

        print(f"  → {len(triples)}개 triple 추출")
        time.sleep(SLEEP_SEC)

    return kg, flat


# ── 저장 ────────────────────────────────────────────────────────────────

def save_kg(kg: list[dict], flat: list[dict], base_dir: str, target: str):
    kg_path   = os.path.join(base_dir, f"llm_kg_{target}.json")
    flat_path = os.path.join(base_dir, f"llm_kg_flat_{target}.json")

    with open(kg_path, "w", encoding="utf-8") as f:
        json.dump(kg, f, ensure_ascii=False, indent=2)

    with open(flat_path, "w", encoding="utf-8") as f:
        json.dump(flat, f, ensure_ascii=False, indent=2)

    print(f"\n[저장 완료]")
    print(f"  KG      : {kg_path}  ({len(kg)}개 항목)")
    print(f"  Flat KG : {flat_path}  ({len(flat)}개 triple)")


# ── 통계 ────────────────────────────────────────────────────────────────

def print_stats(kg: list[dict], target: str):
    counts     = [len(item["triples"]) for item in kg]
    hall_items = [item for item in kg if item["tf_label"]]
    ok_items   = [item for item in kg if not item["tf_label"]]

    total = sum(counts)
    avg   = total / len(counts) if counts else 0

    print(f"\n{'='*60}")
    print(f"📊 LLM KG 구축 통계 ({target})")
    print(f"{'='*60}")
    print(f"총 항목 수          : {len(kg)}")
    print(f"  - hallucination   : {len(hall_items)}개")
    print(f"  - 정확            : {len(ok_items)}개")
    print(f"총 triple 수        : {total}")
    print(f"평균 triple/항목    : {avg:.1f}")

    if hall_items:
        hall_avg = sum(len(i["triples"]) for i in hall_items) / len(hall_items)
        print(f"hallucination 항목 평균 triple : {hall_avg:.1f}")
    if ok_items:
        ok_avg = sum(len(i["triples"]) for i in ok_items) / len(ok_items)
        print(f"정확 항목 평균 triple          : {ok_avg:.1f}")


# ── 메인 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM 답변 KG 구축")
    parser.add_argument(
        "--target",
        choices=["finetune", "zeroshot"],
        default="finetune",
    )
    parser.add_argument(
        "--client",
        choices=["groq", "gemini"],
        default="groq",
    )
    parser.add_argument("--groq-key",   default=os.getenv("GROQ_API_KEY", ""))
    parser.add_argument("--gemini-key", default=os.getenv("GEMINI_API_KEY", ""))
    parser.add_argument("--xlsx",       default="1차년도 QnA 데이터셋 평가.xlsx")
    parser.add_argument("--dry-run",    action="store_true", help="처음 3개만 테스트")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 클라이언트 초기화
    if args.client == "groq":
        client = GroqClient(api_key=args.groq_key)
        print(f"[LLM] Groq ({GroqClient.DEFAULT_MODEL})")
    else:
        client = GeminiClient(api_key=args.gemini_key)
        print(f"[LLM] Gemini ({GeminiClient.DEFAULT_MODEL})")

    # 데이터 로드
    xlsx = args.xlsx if os.path.isabs(args.xlsx) else os.path.join(base_dir, args.xlsx)
    items = load_llm_answers(xlsx, target=args.target)

    if args.dry_run:
        items = items[:3]
        print(f"[dry-run] 처음 {len(items)}개만 실행")

    # KG 구축
    print(f"\n{'='*60}")
    print(f"LLM KG 구축 시작: {len(items)}개 항목 ({args.target})")
    print(f"{'='*60}")

    kg, flat = build_llm_kg(items, client)

    # 통계
    print_stats(kg, args.target)

    # 저장
    if not args.dry_run:
        save_kg(kg, flat, base_dir, args.target)
    else:
        print("\n[dry-run] 저장 생략")
        print("\n[샘플 triple]")
        for item in kg[:2]:
            label = "❌" if item["tf_label"] else "✅"
            print(f"\n{label} Q{item['idx']}: {item['question'][:50]}")
            for t in item["triples"][:3]:
                print(f"  ({t['subject']}, {t['predicate']}, {t['object']})")


if __name__ == "__main__":
    main()