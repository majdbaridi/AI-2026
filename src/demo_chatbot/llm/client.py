"""LLM layer. The app depends on one interface (LLMClient); swapping providers
is a config change, not a code change."""
from __future__ import annotations

from abc import ABC, abstractmethod

from demo_chatbot.config import Settings, get_settings


class LLMClient(ABC):
    @abstractmethod
    async def complete(self, system: str, user: str) -> str: ...


class MockLLM(LLMClient):
    """Offline, no keys. Echoes the prompt back to prove the pipeline works."""

    async def complete(self, system: str, user: str) -> str:
        return f"[mock-llm] {user.strip()}"


class OpenAILLM(LLMClient):
    """Real answers via OpenAI's chat API."""

    def __init__(self, api_key: str, model: str) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def complete(self, system: str, user: str) -> str:
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
        )
        return resp.choices[0].message.content or ""


def build_llm(settings: Settings | None = None) -> LLMClient:
    settings = settings or get_settings()
    if settings.llm_provider == "openai":
        return OpenAILLM(settings.openai_api_key, settings.llm_model)
    return MockLLM()