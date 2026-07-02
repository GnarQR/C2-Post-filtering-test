"""
도메인 KG 구축 스크립트
========================
QnA 데이터셋의 전문가 정답(ground truth) 40개에서 triple을 추출해서
도메인 Knowledge Graph를 JSON으로 저장합니다.

출력:
    domain_kg.json  - 전체 KG (질문별 triple 목록)
    domain_kg_flat.json - flat triple 목록 (검증 시 사용)

사용법:
    python build_kg.py --client groq
    python build_kg.py --client gemini
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

# ── 설정 ────────────────────────────────────────────────────────────────

XLSX_PATH  = "1차년도 QnA 데이터셋 평가.xlsx"
OUT_KG     = "domain_kg.json"
OUT_FLAT   = "domain_kg_flat.json"
SLEEP_SEC  = 2.0  # API rate limit 방지


# ── 데이터 로드 ──────────────────────────────────────────────────────────

def load_ground_truths(xlsx_path: str) -> list[dict]:
    """
    엑셀에서 질문 + 전문가 정답만 추출

    컬럼 구조 (연구자 평가 시트):
        0: 번호
        1: 질문
        2: 정답 (전문가 작성)
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

        if not ground_truth:
            continue

        items.append({
            "idx":          idx,
            "question":     question,
            "ground_truth": ground_truth,
        })

    print(f"[load_ground_truths] {len(items)}개 항목 로드")
    return items


# ── KG 구축 ─────────────────────────────────────────────────────────────

def build_kg(
    items:  list[dict],
    client: BaseLLMClient,
) -> tuple[list[dict], list[dict]]:
    """
    전문가 정답 전체에서 triple 추출

    Returns
    -------
    kg : list[dict]
        질문별 triple 목록
        [{"idx": 1, "question": "...", "ground_truth": "...", "triples": [...]}, ...]

    flat : list[dict]
        전체 triple flat 목록 (검증 시 KG 검색용)
        [{"idx": 1, "subject": "...", "predicate": "...", "object": "..."}, ...]
    """
    kg   = []
    flat = []

    for item in items:
        print(f"\n[Q{item['idx']}] {item['question'][:60]}...")
        print(f"  정답: {item['ground_truth'][:80]}...")

        triples = extract_triples(
            text=item["ground_truth"],
            client=client,
            verbose=True,
        )

        kg.append({
            "idx":          item["idx"],
            "question":     item["question"],
            "ground_truth": item["ground_truth"],
            "triples":      [t.to_dict() for t in triples],
        })

        for t in triples:
            flat.append({
                "idx":       item["idx"],
                "question":  item["question"],
                **t.to_dict(),
            })

        print(f"  → {len(triples)}개 triple 추출")
        time.sleep(SLEEP_SEC)

    return kg, flat


# ── 저장 ────────────────────────────────────────────────────────────────

def save_kg(kg: list[dict], flat: list[dict], base_dir: str):
    kg_path   = os.path.join(base_dir, OUT_KG)
    flat_path = os.path.join(base_dir, OUT_FLAT)

    with open(kg_path, "w", encoding="utf-8") as f:
        json.dump(kg, f, ensure_ascii=False, indent=2)

    with open(flat_path, "w", encoding="utf-8") as f:
        json.dump(flat, f, ensure_ascii=False, indent=2)

    print(f"\n[저장 완료]")
    print(f"  KG       : {kg_path}  ({len(kg)}개 항목)")
    print(f"  Flat KG  : {flat_path}  ({len(flat)}개 triple)")


# ── 통계 출력 ────────────────────────────────────────────────────────────

def print_stats(kg: list[dict]):
    triple_counts = [len(item["triples"]) for item in kg]
    total  = sum(triple_counts)
    avg    = total / len(triple_counts) if triple_counts else 0
    mx     = max(triple_counts) if triple_counts else 0
    mn     = min(triple_counts) if triple_counts else 0

    print(f"\n{'='*60}")
    print(f"📊 KG 구축 통계")
    print(f"{'='*60}")
    print(f"총 항목 수    : {len(kg)}")
    print(f"총 triple 수  : {total}")
    print(f"평균 triple/Q : {avg:.1f}")
    print(f"최대 triple   : {mx}")
    print(f"최소 triple   : {mn}")

    print(f"\n[항목별 triple 수]")
    for item in kg:
        n = len(item["triples"])
        bar = "█" * n
        print(f"  Q{item['idx']:2d}: {bar} ({n})")


# ── 메인 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="도메인 KG 구축")
    parser.add_argument(
        "--client",
        choices=["groq", "gemini"],
        default="gemini",
        help="사용할 LLM 클라이언트 (기본: gemini)",
    )
    parser.add_argument(
        "--groq-key",
        default=os.getenv("GROQ_API_KEY", ""),
        help="Groq API 키",
    )
    parser.add_argument(
        "--gemini-key",
        default=os.getenv("GEMINI_API_KEY", ""),
        help="Gemini API 키",
    )
    parser.add_argument(
        "--xlsx",
        default=XLSX_PATH,
        help="데이터셋 엑셀 파일 경로",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="처음 3개만 테스트 실행",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # LLM 클라이언트 초기화
    if args.client == "groq":
        client = GroqClient(api_key=args.groq_key)
        print(f"[LLM] Groq ({GroqClient.DEFAULT_MODEL})")
    else:
        client = GeminiClient(api_key=args.gemini_key)
        print(f"[LLM] Gemini ({GeminiClient.DEFAULT_MODEL})")

    # 데이터 로드
    xlsx_path = args.xlsx if os.path.isabs(args.xlsx) else os.path.join(base_dir, args.xlsx)
    items = load_ground_truths(xlsx_path)

    if args.dry_run:
        items = items[:3]
        print(f"[dry-run] 처음 {len(items)}개만 실행")

    # KG 구축
    print(f"\n{'='*60}")
    print(f"KG 구축 시작: {len(items)}개 항목")
    print(f"{'='*60}")

    kg, flat = build_kg(items, client)

    # 통계 출력
    print_stats(kg)

    # 저장
    if not args.dry_run:
        save_kg(kg, flat, base_dir)
    else:
        print("\n[dry-run] 저장 생략 — 실제 실행 시 --dry-run 제거")
        print(f"\n[샘플 triple]")
        for item in kg:
            print(f"\nQ{item['idx']}: {item['question'][:50]}")
            for t in item["triples"]:
                print(f"  ({t['subject']}, {t['predicate']}, {t['object']})")


if __name__ == "__main__":
    main()