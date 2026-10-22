from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from .config import Settings, get_settings


class LLMNotConfigured(RuntimeError):
    pass


# OpenAI-compatible endpoint of each toolkit provider kind (Settings → AI connections)
_OPENAI_COMPATIBLE = {
    "ollama": lambda base: base.rstrip("/").removesuffix("/v1") + "/v1",
    "openai": lambda base: base.rstrip("/"),
    "anthropic": lambda base: base.rstrip("/").removesuffix("/v1") + "/v1",
    "gemini": lambda base: base.rstrip("/") + "/v1beta/openai",
}


class OllamaLLM:
    """The Voice Room's AI: the toolkit's active AI connection, used through its
    OpenAI-compatible API (VOICE_ROOM_LLM_BASE_URL / _MODEL / OLLAMA_API_KEY override)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: AsyncOpenAI | None = None
        self._client_for: tuple[str, str] | None = None

    def connection(self) -> dict[str, Any] | None:
        """{base_url, api_key, model, label, reasoning_effort} or None if no AI is set up."""
        override = os.getenv("VOICE_ROOM_LLM_BASE_URL")
        if override:
            return {"base_url": override, "api_key": os.getenv("OLLAMA_API_KEY") or "none",
                    "model": os.getenv("VOICE_ROOM_LLM_MODEL", "minimax-m3"), "label": override,
                    "reasoning_effort": os.getenv("VOICE_ROOM_REASONING_EFFORT", "none") or None}
        from ..settings import providers, store
        conn = store.active_connection(store.load())
        if conn is None or not conn.get("model"):
            return None
        provider = providers.get_provider(conn["provider"])
        base = (conn.get("baseUrl") or provider.base_url).strip()
        return {
            "base_url": _OPENAI_COMPATIBLE.get(provider.api, lambda b: b)(base),
            "api_key": conn.get("apiKey") or "none",
            "model": conn["model"],
            "label": f"{provider.label} · {conn['model']}",
            # Reasoning models on Ollama spend short bid/summary calls thinking; keep it off there
            "reasoning_effort": "none" if provider.api == "ollama" else None,
        }

    @property
    def configured(self) -> bool:
        return self.connection() is not None

    @property
    def model(self) -> str:
        conn = self.connection()
        return conn["model"] if conn else ""

    def _get_client(self) -> AsyncOpenAI:
        conn = self.connection()
        if conn is None:
            raise LLMNotConfigured("No AI connection: add one in Settings → AI connections and make it active")
        key = (conn["base_url"], conn["api_key"])
        if self._client is None or self._client_for != key:
            self._client = AsyncOpenAI(
                base_url=conn["base_url"],
                api_key=conn["api_key"],
                timeout=self.settings.llm_timeout_seconds,
                max_retries=1,
            )
            self._client_for = key
        return self._client

    def _extra(self) -> dict[str, Any]:
        conn = self.connection() or {}
        effort = conn.get("reasoning_effort")
        return {"reasoning_effort": effort} if effort else {}

    async def verify_model(self) -> tuple[bool, str]:
        conn = self.connection()
        if conn is None:
            return False, "No AI connection: add one in Settings → AI connections"
        try:
            models = await self._get_client().models.list()
            names = {m.id for m in models.data}
            if not names or conn["model"] in names:
                return True, f"{conn['label']} is available"
            return False, f"{conn['model']} isn't offered by {conn['base_url']}"
        except Exception as exc:
            return False, f"Could not reach the AI: {exc}"

    async def bid(
        self,
        *,
        name: str,
        personality: str,
        summary: str,
        transcript: str,
    ) -> float:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self.model,
            extra_body=self._extra(),
            temperature=0.2,
            max_tokens=24,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are {name}. {personality}\n"
                        "Decide how strongly you want to speak next. Return only "
                        'JSON in the form {"bid": 0}, using an integer from 0 to 10.'
                    ),
                },
                {
                    "role": "user",
                    "content": f"Conversation summary:\n{summary or '(none)'}\n\nRecent lines:\n{transcript}",
                },
            ],
        )
        content = response.choices[0].message.content or ""
        try:
            match = re.search(r"\{.*?\}", content, flags=re.DOTALL)
            payload: dict[str, Any] = json.loads(match.group(0) if match else content)
            return max(0.0, min(10.0, float(payload["bid"])))
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            number = re.search(r"\b(10|[0-9])\b", content)
            return float(number.group(1)) if number else 5.0

    async def stream_line(
        self,
        *,
        name: str,
        personality: str,
        max_words: int,
        summary: str,
        transcript: str,
        drift_nudge: bool,
    ) -> AsyncIterator[str]:
        client = self._get_client()
        nudge = (
            " Change the angle or move the conversation forward; avoid repeating recent openings."
            if drift_nudge
            else ""
        )
        stream = await client.chat.completions.create(
            model=self.model,
            extra_body=self._extra(),
            temperature=0.85,
            stream=True,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are {name}. Stay fully in character.\n{personality}\n"
                        f"Speak one natural turn of at most {max_words} words. "
                        "Do not prefix your answer with your name and do not narrate actions."
                        f"{nudge}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Conversation summary:\n{summary or '(none)'}\n\n"
                        f"Recent transcript:\n{transcript}\n\nReply as {name}."
                    ),
                },
            ],
        )
        async for chunk in stream:
            content = chunk.choices[0].delta.content if chunk.choices else None
            if content:
                yield content

    async def summarize(self, previous: str, transcript: str) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self.model,
            extra_body=self._extra(),
            temperature=0.1,
            max_tokens=350,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Maintain a compact factual conversation memory. Preserve names, "
                        "decisions, running jokes, disagreements, and unresolved questions."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Previous memory:\n{previous or '(none)'}\n\nLines to fold in:\n{transcript}",
                },
            ],
        )
        return (response.choices[0].message.content or previous).strip()


llm_service = OllamaLLM()
