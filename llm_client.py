"""
LLM 클라이언트 추상화 레이어
============================
BaseLLMClient를 상속해서 새로운 LLM을 추가할 수 있습니다.

지원 모델:
    - GroqClient  : Groq API (llama-3.3-70b-versatile 등)
    - GeminiClient: Google Gemini API (gemini-2.5-flash-lite 등)

사용법:
    from llm_client import GroqClient, GeminiClient

    client = GroqClient(api_key="...")
    response = client.chat("안녕하세요", system="You are a helpful assistant.")
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

# .env 파일이 있으면 자동 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── Base Class ───────────────────────────────────────────────────────────

class BaseLLMClient(ABC):
    """모든 LLM 클라이언트의 공통 인터페이스"""

    @abstractmethod
    def chat(self, prompt: str, system: str = "") -> str:
        """
        Parameters
        ----------
        prompt : str
            사용자 메시지
        system : str
            시스템 프롬프트 (선택)

        Returns
        -------
        str
            LLM 응답 텍스트
        """
        raise NotImplementedError


# ── Groq ─────────────────────────────────────────────────────────────────

class GroqClient(BaseLLMClient):
    """
    Groq API 클라이언트

    Parameters
    ----------
    api_key : str
        Groq API 키. 없으면 환경변수 GROQ_API_KEY 사용
    model : str
        사용할 모델명 (기본: llama-3.3-70b-versatile)
    temperature : float
        생성 온도 (기본: 0.0 — 재현성 최대화)
    """

    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(
        self,
        api_key:     str   = "",
        model:       str   = DEFAULT_MODEL,
        temperature: float = 0.0,
    ):
        from groq import Groq
        self.client      = Groq(api_key=api_key or os.getenv("GROQ_API_KEY", ""))
        self.model       = model
        self.temperature = temperature

    def chat(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return response.choices[0].message.content


# ── Gemini ───────────────────────────────────────────────────────────────

class GeminiClient(BaseLLMClient):
    """
    Google Gemini API 클라이언트

    Parameters
    ----------
    api_key : str
        Gemini API 키. 없으면 환경변수 GEMINI_API_KEY 사용
    model : str
        사용할 모델명 (기본: gemini-2.5-flash-lite)
    temperature : float
        생성 온도 (기본: 0.0)
    """

    DEFAULT_MODEL = "gemini-2.5-flash-lite"

    def __init__(
        self,
        api_key:     str   = "",
        model:       str   = DEFAULT_MODEL,
        temperature: float = 0.0,
    ):
        import google.generativeai as genai
        genai.configure(api_key=api_key or os.getenv("GEMINI_API_KEY", ""))
        self.model_name  = model
        self.temperature = temperature
        self._genai      = genai

    def chat(self, prompt: str, system: str = "") -> str:
        generation_config = self._genai.types.GenerationConfig(
            temperature=self.temperature,
        )
        model = self._genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system if system else None,
            generation_config=generation_config,
        )
        response = model.generate_content(prompt)
        return response.text


# ── 테스트 ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--client", choices=["groq", "gemini"], default="groq")
    args = parser.parse_args()

    if args.client == "groq":
        client = GroqClient()
    else:
        client = GeminiClient()

    response = client.chat(
        prompt="건유기의 적정 기간은 며칠인가요? 한 문장으로 답하세요.",
        system="You are a livestock domain expert.",
    )
    print(f"[{args.client}] 응답: {response}")