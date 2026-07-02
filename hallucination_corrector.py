"""
Hallucination 탐지 + 교정 모듈
================================
GraphCorrect 방식으로 구현:
    1. LLM 답변 → triple 추출
    2. 각 triple → 문장으로 재결합 → 전문가 정답과 NLI 비교
    3. contradiction triple만 선택적으로 교정
    4. 교정된 내용을 원본 답변에 반영

참고: GraphEval (Sansford et al., KDD KiL 2024)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from llm_client import BaseLLMClient
from triple_extractor import Triple, extract_triples


# ── 데이터 클래스 ────────────────────────────────────────────────────────

@dataclass
class TripleVerification:
    triple:          Triple
    nli_label:       str    # entailment / contradiction / neutral
    nli_score:       float
    is_hallucinated: bool


@dataclass
class CorrectionResult:
    original_answer:   str
    corrected_answer:  str
    is_hallucinated:   bool      # 교정 전 hallucination 여부
    flagged_triples:   list[TripleVerification] = field(default_factory=list)
    all_triples:       list[TripleVerification] = field(default_factory=list)
    summary:           str = ""

    @property
    def was_corrected(self) -> bool:
        return self.original_answer != self.corrected_answer


# ── NLI ─────────────────────────────────────────────────────────────────

_nli_pipeline = None
NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"


def get_nli_pipeline():
    global _nli_pipeline
    if _nli_pipeline is None:
        from transformers import pipeline
        print(f"[NLI] 모델 로드 중: {NLI_MODEL}")
        _nli_pipeline = pipeline(
            "zero-shot-classification",
            model=NLI_MODEL,
            device=-1,
        )
        print("[NLI] 완료")
    return _nli_pipeline


def run_nli(hypothesis: str, premise: str) -> tuple[str, float]:
    """
    triple 문장(hypothesis) vs 전문가 정답(premise) NLI 판정

    수치가 포함된 경우 숫자 직접 비교로 보완
    """
    # 수치 비교 보조
    nums_h = set(re.findall(r'\d+\.?\d*', hypothesis))
    nums_p = set(re.findall(r'\d+\.?\d*', premise))
    numeric_mismatch = bool(nums_h and nums_p and not nums_h.issubset(nums_p))

    nli = get_nli_pipeline()
    result = nli(
        sequences=premise,
        candidate_labels=[hypothesis, "this is unrelated"],
        hypothesis_template="{}",
        multi_label=False,
    )

    hyp_score = (result["scores"][0]
                 if result["labels"][0] == hypothesis
                 else result["scores"][1])

    if hyp_score >= 0.6:
        label, score = "entailment", hyp_score
    elif hyp_score <= 0.35:
        label, score = "contradiction", 1 - hyp_score
    else:
        label, score = "neutral", hyp_score

    # 수치 불일치 보강
    if numeric_mismatch and label in ("neutral", "entailment"):
        return "contradiction", max(score, 0.65)

    return label, score


# ── Step 1: Triple 검증 ──────────────────────────────────────────────────

def verify_triples(
    triples:      list[Triple],
    ground_truth: str,
) -> list[TripleVerification]:
    """각 triple을 전문가 정답과 NLI로 비교"""
    verifications = []
    for triple in triples:
        sentence  = triple.to_sentence()
        label, score = run_nli(sentence, ground_truth)
        is_hall   = (label == "contradiction")

        verifications.append(TripleVerification(
            triple=triple,
            nli_label=label,
            nli_score=score,
            is_hallucinated=is_hall,
        ))
    return verifications


# ── Step 2: Triple 교정 ──────────────────────────────────────────────────

CORRECT_TRIPLE_PROMPT = """\
The following triple contains factually incorrect information.
Correct it based on the provided ground truth context.

Rules:
1. A triple is defined as [subject, predicate, object]
2. Return ONLY the corrected triple as JSON, no explanation
3. The concatenated triple must make sense as a sentence

Triple (incorrect):
subject: {subject}
predicate: {predicate}
object: {obj}

Ground truth context:
{ground_truth}

Return format:
{{"subject": "...", "predicate": "...", "object": "..."}}
"""

APPLY_CORRECTION_PROMPT = """\
In the following answer, replace the information from the old triple with the new corrected triple.
Do NOT make any other changes to the answer.
Return ONLY the updated answer, nothing else.

Answer:
{answer}

Old triple (incorrect): {old_triple}
New triple (corrected): {new_triple}
"""


def correct_triple(
    triple:       Triple,
    ground_truth: str,
    client:       BaseLLMClient,
) -> Triple | None:
    """틀린 triple을 정답 기반으로 교정"""
    prompt = CORRECT_TRIPLE_PROMPT.format(
        subject=triple.subject,
        predicate=triple.predicate,
        obj=triple.obj,
        ground_truth=ground_truth,
    )
    raw = client.chat(
        prompt=prompt,
        system="You are a knowledge graph correction expert. Output only valid JSON.",
    )

    json_match = re.search(r'\{.*?\}', raw, re.DOTALL)
    if not json_match:
        return None
    try:
        data = json.loads(json_match.group())
        return Triple(
            subject=data.get("subject", triple.subject),
            predicate=data.get("predicate", triple.predicate),
            obj=data.get("object", triple.obj),
        )
    except json.JSONDecodeError:
        return None


def apply_correction(
    answer:     str,
    old_triple: Triple,
    new_triple: Triple,
    client:     BaseLLMClient,
) -> str:
    """교정된 triple을 원본 답변에 반영"""
    prompt = APPLY_CORRECTION_PROMPT.format(
        answer=answer,
        old_triple=old_triple.to_sentence(),
        new_triple=new_triple.to_sentence(),
    )
    return client.chat(
        prompt=prompt,
        system="You are a text editor. Make minimal changes to the answer.",
    ).strip()


# ── 메인 클래스 ──────────────────────────────────────────────────────────

class HallucinationCorrector:
    """
    C2 사후 필터링 - 탐지 + 교정

    사용 예시:
        corrector = HallucinationCorrector(client)
        result = corrector.run(
            llm_answer="건유기는 분만 전 30일 동안 지속됩니다.",
            ground_truth="건유기의 적정 기간은 45일에서 60일입니다.",
        )
        print(result.corrected_answer)
    """

    def __init__(self, client: BaseLLMClient):
        self.client = client

    def run(
        self,
        llm_answer:   str,
        ground_truth: str,
        verbose:      bool = False,
    ) -> CorrectionResult:
        """전체 파이프라인 실행"""

        # Step 1: triple 추출
        if verbose:
            print("[Step 1] Triple 추출...")
        triples = extract_triples(llm_answer, self.client, verbose=verbose)

        if not triples:
            return CorrectionResult(
                original_answer=llm_answer,
                corrected_answer=llm_answer,
                is_hallucinated=False,
                summary="Triple 추출 실패 — 교정 불가",
            )

        # Step 2: NLI 검증
        if verbose:
            print("[Step 2] NLI 검증...")
        verifications = verify_triples(triples, ground_truth)
        flagged = [v for v in verifications if v.is_hallucinated]
        is_hallucinated = len(flagged) > 0

        if verbose:
            print(f"  총 {len(verifications)}개 triple 중 {len(flagged)}개 hallucination")

        # Step 3: 교정
        corrected = llm_answer
        if flagged:
            if verbose:
                print("[Step 3] 교정 중...")
            for tv in flagged:
                new_triple = correct_triple(tv.triple, ground_truth, self.client)
                if new_triple and new_triple.to_sentence() != tv.triple.to_sentence():
                    corrected = apply_correction(corrected, tv.triple, new_triple, self.client)
                    if verbose:
                        print(f"  교정: {tv.triple} → {new_triple}")

        # 요약
        if flagged:
            summary = (f"{len(triples)}개 triple 중 {len(flagged)}개 hallucination 탐지 및 교정")
        else:
            summary = f"{len(triples)}개 triple 모두 정확"

        return CorrectionResult(
            original_answer=llm_answer,
            corrected_answer=corrected,
            is_hallucinated=is_hallucinated,
            flagged_triples=flagged,
            all_triples=verifications,
            summary=summary,
        )


# ── 테스트 ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from llm_client import GroqClient, GeminiClient

    parser = argparse.ArgumentParser()
    parser.add_argument("--client", choices=["groq", "gemini"], default="gemini")
    args = parser.parse_args()

    if args.client == "gemini":
        client = GeminiClient(api_key=os.getenv("GEMINI_API_KEY", ""))
        print(f"[LLM] Gemini ({GeminiClient.DEFAULT_MODEL})")
    else:
        client = GroqClient(api_key=os.getenv("GROQ_API_KEY", ""))
        print(f"[LLM] Groq ({GroqClient.DEFAULT_MODEL})")

    corrector = HallucinationCorrector(client)

    test_cases = [
        {
            "name": "건유기 기간 오류",
            "llm_answer": "건유기는 분만 전 30일 동안 지속됩니다. 이 기간에 유선 조직이 회복됩니다.",
            "ground_truth": "건유기의 목표는 유선 조직 회복과 태아 성장 촉진이며, 적정 기간은 45일에서 60일입니다.",
            "expected_hallucinated": True,
        },
        {
            "name": "암모니아 농도 오류",
            "llm_answer": "축사 내 암모니아 농도는 25ppm 이하로 유지해야 합니다.",
            "ground_truth": "축사 내부의 암모니아 농도는 20ppm 이하로 유지해야 합니다.",
            "expected_hallucinated": True,
        },
        {
            "name": "정확한 답변",
            "llm_answer": "건유기의 적정 기간은 45일에서 60일입니다.",
            "ground_truth": "건유기의 목표는 유선 조직 회복과 태아 성장 촉진이며, 적정 기간은 45일에서 60일입니다.",
            "expected_hallucinated": False,
        },
    ]

    for i, tc in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"[테스트 {i}] {tc['name']}")
        print(f"원본: {tc['llm_answer']}")

        result = corrector.run(
            llm_answer=tc["llm_answer"],
            ground_truth=tc["ground_truth"],
            verbose=True,
        )

        print(f"\n교정 결과: {result.corrected_answer}")
        print(f"Hallucination: {result.is_hallucinated} (예상: {tc['expected_hallucinated']})")
        print(f"교정됨: {result.was_corrected}")
        print(f"요약: {result.summary}")