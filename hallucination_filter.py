"""
C2 사후 필터링 - Hallucination 탐지 모듈 (Groq 버전 v2)
=========================================================

파이프라인:
    1. KG 트리플 추출  : Groq LLM이 final_answer를 트리플로 분해  [LLM 1회]
    2. AGROVOC 검증   : 트리플 용어를 온톨로지에서 확인            [rdflib]
    3. NLI 판정       : 트리플 → 자연어 후 ground_truth 비교       [다국어 NLI]
    4. 규칙 기반 판정 : NLI 결과 + 수치 비교로 최종 판정           [LLM 없음]

변경사항 (v2):
    - Step 4 LLM 종합 판정 제거 → Groq 호출 2회 → 1회 (토큰 절반)
    - NLI 모델 교체: 영어 전용 → 다국어 (한/영 혼용 비교 가능)
      cross-encoder/nli-deberta-v3-small
      → "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    - 규칙 기반 판정: contradiction 비율 + 수치 불일치로 판정

사전 준비:
    1. Groq API 키 발급
       https://console.groq.com
    2. 패키지 설치
       pip install groq transformers torch rdflib
    3. 환경변수 설정
       set GROQ_API_KEY=gsk_your_groq_api_key_here
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any

from rdflib import Graph

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agrovoc_utils import load_graph, validate_term


# ── 설정 ────────────────────────────────────────────────────────────────

GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL    = "llama-3.3-70b-versatile"   # 또는 "mixtral-8x7b-32768"
NLI_MODEL     = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"  # 다국어 지원
AGROVOC_TTL   = "agrovocSubOntology.ttl"


# ── 데이터 클래스 ────────────────────────────────────────────────────────

@dataclass
class Triple:
    subject:   str
    predicate: str
    obj:       str

    def to_sentence(self) -> str:
        return f"{self.subject} {self.predicate} {self.obj}."


@dataclass
class TripleVerification:
    triple:             Triple
    agrovoc_valid:      bool
    agrovoc_uri:        str | None
    nli_label:          str    # entailment / contradiction / neutral
    nli_score:          float
    is_hallucinated:    bool
    hallucination_type: str    # intrinsic / extrinsic / none


@dataclass
class FilterResult:
    original_answer:  str
    is_hallucinated:  bool
    confidence:       float
    flagged_triples:  list[TripleVerification] = field(default_factory=list)
    all_triples:      list[TripleVerification] = field(default_factory=list)
    summary:          str = ""

    def print_report(self):
        print("\n" + "="*60)
        print("📋 C2 사후 필터링 결과")
        print("="*60)
        print(f"원본 답변: {self.original_answer}")
        print(f"Hallucination: {'❌ 탐지됨' if self.is_hallucinated else '✅ 없음'}")
        print(f"신뢰도: {self.confidence:.2f}")
        print(f"요약: {self.summary}")

        if self.all_triples:
            print("\n[트리플별 결과]")
            for tv in self.all_triples:
                status = "❌" if tv.is_hallucinated else "✅"
                print(f"  {status} ({tv.triple.subject}, "
                      f"{tv.triple.predicate}, {tv.triple.obj})")
                print(f"     AGROVOC: {'등재' if tv.agrovoc_valid else '미등재'} "
                      f"| NLI: {tv.nli_label} ({tv.nli_score:.2f})")
                if tv.is_hallucinated:
                    print(f"     유형: {tv.hallucination_type} hallucination")
        print("="*60)


# ── Groq 클라이언트 ──────────────────────────────────────────────────────

class GroqClient:
    """Groq API 클라이언트"""

    def __init__(self, api_key: str = GROQ_API_KEY, model: str = GROQ_MODEL):
        from groq import Groq
        self.client = Groq(api_key=api_key)
        self.model  = model

    def chat(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )
        return response.choices[0].message.content


# ── Step 1: KG 트리플 추출 ───────────────────────────────────────────────

TRIPLE_EXTRACTION_PROMPT = """\
Extract factual claims from the following livestock domain answer as knowledge graph triples.

Rules:
- Include ALL numerical claims (amounts, concentrations, durations, counts)
- Include breed, gender, status classifications
- Use English terms for AGROVOC mapping
- Return ONLY a JSON array, no other text, no markdown

Answer (Korean):
{answer}

Output format:
[
  {{"subject": "...", "predicate": "...", "object": "..."}},
  ...
]
"""


def extract_triples(answer: str, client: GroqClient) -> list[Triple]:
    """Groq LLM으로 트리플 추출 (LLM 호출 1회)"""
    prompt = TRIPLE_EXTRACTION_PROMPT.format(answer=answer)
    raw = client.chat(
        prompt=prompt,
        system="You are a knowledge graph construction expert. Output only valid JSON array.",
    )

    json_match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if not json_match:
        print(f"[extract_triples] JSON 추출 실패: {raw[:200]}")
        return []

    try:
        items = json.loads(json_match.group())
        triples = [
            Triple(
                subject=item["subject"],
                predicate=item["predicate"],
                obj=item["object"],
            )
            for item in items
            if all(k in item for k in ("subject", "predicate", "object"))
        ]
        print(f"[extract_triples] {len(triples)}개 트리플 추출됨")
        for t in triples:
            print(f"  → ({t.subject}, {t.predicate}, {t.obj})")
        return triples
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[extract_triples] 파싱 오류: {e}")
        return []


# ── Step 2: AGROVOC 검증 ─────────────────────────────────────────────────

def verify_with_agrovoc(triple: Triple, g: Graph) -> dict:
    """트리플의 subject/object가 AGROVOC 서브셋에 존재하는지 확인"""
    for term in [triple.subject, triple.obj]:
        result = validate_term(g, term.lower())
        if result["is_valid"]:
            return {
                "valid": True,
                "uri": result["uri"],
                "matched_term": term,
            }
    return {"valid": False, "uri": None, "matched_term": None}


# ── Step 3: NLI 판정 ─────────────────────────────────────────────────────

_nli_pipeline = None


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
        print("[NLI] 모델 로드 완료")
    return _nli_pipeline


def run_nli(hypothesis: str, premise: str) -> tuple[str, float]:
    """
    다국어 NLI 판정 + 수치 비교 보조

    - 다국어 모델 사용으로 한국어 premise vs 영어 hypothesis 비교 가능
    - 수치가 포함된 경우 숫자 직접 비교로 보완
    """
    # 수치 비교 보조 — hypothesis 숫자가 premise에 없으면 mismatch
    nums_h = set(re.findall(r'\d+\.?\d*', hypothesis))
    nums_p = set(re.findall(r'\d+\.?\d*', premise))
    numeric_mismatch = bool(nums_h and nums_p and not nums_h.issubset(nums_p))

    # 다국어 NLI 판정
    # premise(한국어)를 sequence로, hypothesis(영어)를 candidate로
    nli = get_nli_pipeline()
    result = nli(
        sequences=premise,
        candidate_labels=[hypothesis, "this is unrelated"],
        hypothesis_template="{}",
        multi_label=False,
    )

    # 첫 번째 candidate(hypothesis)의 점수로 entailment 판단
    hyp_score = result["scores"][0] if result["labels"][0] == hypothesis else result["scores"][1]

    if hyp_score >= 0.6:
        nli_label = "entailment"
        nli_score = hyp_score
    elif hyp_score <= 0.35:
        nli_label = "contradiction"
        nli_score = 1 - hyp_score
    else:
        nli_label = "neutral"
        nli_score = hyp_score

    # 수치 불일치 + neutral 이면 contradiction으로 강화
    if numeric_mismatch and nli_label in ("neutral", "entailment"):
        return "contradiction", max(nli_score, 0.65)

    return nli_label, nli_score


# ── Step 4: 종합 판정 ────────────────────────────────────────────────────

JUDGE_PROMPT = """\
You are a hallucination detection expert for livestock domain.

[LLM Answer]
{answer}

[Triple Verification Results]
{triple_results}

[Classification]
- intrinsic hallucination: distorts information present in the source (e.g. wrong number)
- extrinsic hallucination: adds information not present in the source
- not_hallucinated: accurate claim

Based on the verification results, make a final judgment.
Return ONLY a JSON object, no markdown:
{{
  "overall_hallucinated": true or false,
  "confidence": 0.0 to 1.0,
  "triples": [
    {{
      "triple": "subject predicate object",
      "verdict": "not_hallucinated" or "intrinsic" or "extrinsic",
      "reason": "brief explanation"
    }}
  ],
  "summary": "overall judgment summary in Korean"
}}
"""


def judge_hallucination(
    answer: str,
    triple_verifications: list[TripleVerification],
    client: GroqClient,
) -> dict:
    """AGROVOC + NLI 결과를 종합해 Groq LLM이 최종 판정 (LLM 호출 1회)"""
    triple_results = []
    for tv in triple_verifications:
        triple_results.append(
            f"- ({tv.triple.subject}, {tv.triple.predicate}, {tv.triple.obj})\n"
            f"  AGROVOC: {'registered' if tv.agrovoc_valid else 'not registered'} "
            f"({tv.agrovoc_uri or 'N/A'})\n"
            f"  NLI: {tv.nli_label} (score={tv.nli_score:.2f})\n"
            f"  Preliminary verdict: {'hallucinated' if tv.is_hallucinated else 'ok'}"
        )

    prompt = JUDGE_PROMPT.format(
        answer=answer,
        triple_results="\n".join(triple_results),
    )

    raw = client.chat(
        prompt=prompt,
        system="You are a hallucination detection expert. Output only valid JSON.",
    )

    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not json_match:
        return {
            "overall_hallucinated": False,
            "confidence": 0.0,
            "triples": [],
            "summary": "판정 실패",
        }

    try:
        return json.loads(json_match.group())
    except json.JSONDecodeError:
        return {
            "overall_hallucinated": False,
            "confidence": 0.0,
            "triples": [],
            "summary": "JSON 파싱 실패",
        }


# ── 메인 필터 클래스 ─────────────────────────────────────────────────────

class HallucinationFilter:
    """
    C2 사후 필터링 메인 클래스

    사용 예시:
        f = HallucinationFilter()
        result = f.check(
            final_answer="건유기는 분만 전 30일입니다.",
            ground_truth="건유기는 분만 전 45~60일입니다.",
        )
        result.print_report()
    """

    def __init__(
        self,
        agrovoc_ttl_path: str = AGROVOC_TTL,
        groq_api_key:     str = GROQ_API_KEY,
        groq_model:       str = GROQ_MODEL,
    ):
        print(f"[HallucinationFilter] 초기화 중...")
        self.agrovoc_graph = load_graph(agrovoc_ttl_path)
        self.client        = GroqClient(api_key=groq_api_key, model=groq_model)
        print(f"[HallucinationFilter] 초기화 완료 (모델: {groq_model})")

    def check(
        self,
        final_answer: str,
        ground_truth: str,
        raw_data: dict[str, Any] | None = None,
    ) -> FilterResult:
        """
        전체 C2 파이프라인 실행

        Parameters
        ----------
        final_answer : str
            LLM이 생성한 최종 답변
        ground_truth : str
            검증 기준 — QnA 데이터셋 정답 or raw_data 요약
        raw_data : dict | None
            DB raw_data (운영 환경에서 ground_truth 대신 사용)
        """
        premise = ground_truth or (
            json.dumps(raw_data, ensure_ascii=False) if raw_data else ""
        )

        # Step 1: 트리플 추출 (Groq 1회)
        print("\n[Step 1] 트리플 추출...")
        triples = extract_triples(final_answer, self.client)
        if not triples:
            return FilterResult(
                original_answer=final_answer,
                is_hallucinated=False,
                confidence=0.0,
                summary="트리플 추출 실패 — 판정 불가",
            )

        # Step 2 + 3: AGROVOC 검증 + NLI 판정
        print("[Step 2+3] AGROVOC 검증 + NLI 판정...")
        triple_verifications = []
        for triple in triples:
            agrovoc_result = verify_with_agrovoc(triple, self.agrovoc_graph)
            nli_label, nli_score = run_nli(triple.to_sentence(), premise)

            is_hallucinated = (nli_label == "contradiction")
            if is_hallucinated:
                h_type = "intrinsic" if agrovoc_result["valid"] else "extrinsic"
            else:
                h_type = "none"

            triple_verifications.append(TripleVerification(
                triple=triple,
                agrovoc_valid=agrovoc_result["valid"],
                agrovoc_uri=agrovoc_result["uri"],
                nli_label=nli_label,
                nli_score=nli_score,
                is_hallucinated=is_hallucinated,
                hallucination_type=h_type,
            ))

        # Step 4: 규칙 기반 판정 (LLM 없음)
        print("[Step 4] 규칙 기반 판정...")
        flagged = [tv for tv in triple_verifications if tv.is_hallucinated]

        total = len(triple_verifications)
        n_contradiction = len(flagged)

        # hallucination 판정: 트리플 하나라도 contradiction이면 전체 hallucination
        # (GraphEval/FactAlign 논문 기준:
        #  "if any of the triples are not grounded → inconsistent")
        is_hallucinated = n_contradiction > 0

        # 신뢰도: hallucination이면 가장 높은 contradiction NLI 점수
        #         정확하면 가장 낮은 NLI 점수 (보수적으로)
        if flagged:
            confidence = max(tv.nli_score for tv in flagged)
        else:
            scores = [tv.nli_score for tv in triple_verifications]
            confidence = min(scores) if scores else 0.5

        # 요약 생성
        if flagged:
            types = [tv.hallucination_type for tv in flagged]
            intrinsic = types.count("intrinsic")
            extrinsic = types.count("extrinsic")
            summary = (f"총 {total}개 트리플 중 {n_contradiction}개 hallucination 탐지 "
                      f"(intrinsic: {intrinsic}, extrinsic: {extrinsic})")
        else:
            summary = f"총 {total}개 트리플 모두 정확한 것으로 판정"

        return FilterResult(
            original_answer=final_answer,
            is_hallucinated=is_hallucinated,
            confidence=round(confidence, 2),
            flagged_triples=flagged,
            all_triples=triple_verifications,
            summary=summary,
        )


# ── 테스트 ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        {
            "name": "건유기 기간 오류",
            "final_answer": "건유기는 분만 전 30일 동안 지속됩니다.",
            "ground_truth": "건유기는 분만 전 45~60일입니다.",
            "expected": True,
        },
        {
            "name": "착유량 정확",
            "final_answer": "젖소의 하루 평균 착유량은 약 25~30kg입니다.",
            "ground_truth": "젖소의 하루 평균 착유량은 25~30kg입니다.",
            "expected": False,
        },
        {
            "name": "암모니아 농도 오류",
            "final_answer": "우사 내 암모니아 농도는 25ppm 이하로 유지해야 합니다.",
            "ground_truth": "우사 내 암모니아 농도는 20ppm 이하로 유지해야 합니다.",
            "expected": True,
        },
    ]

    ttl_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), AGROVOC_TTL
    )

    f = HallucinationFilter(agrovoc_ttl_path=ttl_path)

    print("\n" + "="*60)
    print("C2 사후 필터링 테스트 시작")
    print("="*60)

    correct = 0
    for i, tc in enumerate(test_cases, 1):
        print(f"\n[테스트 {i}] {tc['name']}")
        result = f.check(
            final_answer=tc["final_answer"],
            ground_truth=tc["ground_truth"],
        )
        result.print_report()

        match = result.is_hallucinated == tc["expected"]
        print(f"판정 {'✅ 정확' if match else '❌ 오류'} "
              f"(예상: {tc['expected']}, 실제: {result.is_hallucinated})")
        if match:
            correct += 1

    print(f"\n최종 정확도: {correct}/{len(test_cases)}")