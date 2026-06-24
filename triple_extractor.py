"""
Triple 추출 모듈
================
LLM을 사용해서 텍스트에서 knowledge graph triple을 추출합니다.

사용법:
    from llm_client import GroqClient
    from triple_extractor import extract_triples

    client = GroqClient(api_key="...")
    triples = extract_triples("건유기는 45~60일입니다.", client)
    # → [Triple(subject="건유기", predicate="적정기간", obj="45~60일")]
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from llm_client import BaseLLMClient


# ── 데이터 클래스 ────────────────────────────────────────────────────────

@dataclass
class Triple:
    subject:   str
    predicate: str
    obj:       str

    def to_sentence(self) -> str:
        """triple → 자연어 문장 (NLI 입력용)"""
        return f"{self.subject} {self.predicate} {self.obj}."

    def to_dict(self) -> dict:
        return {
            "subject":   self.subject,
            "predicate": self.predicate,
            "object":    self.obj,
        }

    def __repr__(self) -> str:
        return f"({self.subject}, {self.predicate}, {self.obj})"


# ── 프롬프트 ─────────────────────────────────────────────────────────────

EXTRACTION_SYSTEM = "You are a knowledge graph construction expert for the livestock domain. Output only valid JSON array, no markdown, no explanation."

EXTRACTION_PROMPT = """\
Extract ALL factual claims from the following livestock domain text as knowledge graph triples.

Rules:
1. Include ALL numerical claims (amounts, concentrations, durations, ratios, counts)
2. Include causal relationships (A causes B, A prevents C)
3. Include classifications (A is a type of B)
4. Include recommended actions (A should be done for B)
5. Keep subject/predicate/object concise but complete
6. Use Korean for terms that are domain-specific Korean concepts
7. Use English for internationally standard terms (e.g. BCS, TMR, NLI)
8. Each triple must make sense as a sentence: "subject predicate object"

Text:
{text}

Output format (JSON array only):
[
  {{"subject": "...", "predicate": "...", "object": "..."}},
  ...
]
"""


# ── 핵심 함수 ────────────────────────────────────────────────────────────

def extract_triples(
    text:    str,
    client:  BaseLLMClient,
    verbose: bool = False,
) -> list[Triple]:
    """
    텍스트에서 triple 추출

    Parameters
    ----------
    text : str
        triple을 추출할 텍스트 (한국어 가능)
    client : BaseLLMClient
        LLM 클라이언트 (GroqClient, GeminiClient 등)
    verbose : bool
        True면 추출된 triple 출력

    Returns
    -------
    list[Triple]
        추출된 triple 목록. 실패 시 빈 리스트.
    """
    prompt = EXTRACTION_PROMPT.format(text=text)

    try:
        raw = client.chat(prompt=prompt, system=EXTRACTION_SYSTEM)
    except Exception as e:
        print(f"[extract_triples] LLM 호출 실패: {e}")
        return []

    # JSON 배열 파싱
    json_match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if not json_match:
        print(f"[extract_triples] JSON 추출 실패:\n{raw[:300]}")
        return []

    try:
        items = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        print(f"[extract_triples] JSON 파싱 오류: {e}")
        return []

    triples = []
    for item in items:
        if not all(k in item for k in ("subject", "predicate", "object")):
            continue
        s = str(item["subject"]).strip()
        p = str(item["predicate"]).strip()
        o = str(item["object"]).strip()
        if s and p and o:
            triples.append(Triple(subject=s, predicate=p, obj=o))

    if verbose:
        print(f"[extract_triples] {len(triples)}개 추출")
        for t in triples:
            print(f"  {t}")

    return triples


# ── 테스트 ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from llm_client import GroqClient

    client = GroqClient(api_key=os.getenv("GROQ_API_KEY", ""))

    test_texts = [
        "건유기의 목표는 유선 조직 회복과 태아 성장 촉진이며, 적정 기간은 45일에서 60일입니다. 45일 미만은 유선 조직 회복이 불충분하여 다음 산차의 최대 유량 감소를 유발합니다.",
        "축사 내부의 암모니아 농도는 20ppm 이하로, 이산화탄소 농도는 2,000ppm 이하로 유지해야 합니다.",
        "비유 후기에는 BCS 3.0~3.5를 유지하도록 농후사료 급여량을 줄여야 합니다.",
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"\n{'='*60}")
        print(f"[테스트 {i}] {text[:60]}...")
        triples = extract_triples(text, client, verbose=True)
