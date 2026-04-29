from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency for rule-only mode
    OpenAI = None


class HelloAgentsLLM:
    """最小可用的 HelloAgents LLM 接口。"""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str | None = "auto",
        **kwargs: Any,
    ):
        self.provider = provider or os.getenv("LLM_PROVIDER") or "openai"
        self.model = model or os.getenv("LLM_MODEL_ID") or "gpt-4o-mini"
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens")
        self.timeout = kwargs.get("timeout", 60)

        self._client = None
        if OpenAI is not None and self.api_key:
            client_kwargs: dict[str, Any] = {
                "api_key": self.api_key,
                "timeout": self.timeout,
            }
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self._client = OpenAI(**client_kwargs)

    def is_configured(self) -> bool:
        return self._client is not None

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if self._client is None:
            raise RuntimeError(
                "LLM client is not configured. Set OPENAI_API_KEY or use rule-only mode."
            )

        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        resolved_max_tokens = self.max_tokens if max_tokens is None else max_tokens
        if resolved_max_tokens is not None:
            request["max_tokens"] = resolved_max_tokens

        response = self._client.chat.completions.create(**request)
        content = response.choices[0].message.content
        return self._stringify_content(content)

    @staticmethod
    def _stringify_content(content: Any) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                    continue

                text_value = getattr(item, "text", None)
                if text_value:
                    parts.append(str(text_value))

            return "".join(parts)

        return "" if content is None else str(content)


class HelloAgent(ABC):
    """最小可运行的 HelloAgent 基类。"""

    def __init__(
        self,
        name: str,
        description: str = "",
        llm: HelloAgentsLLM | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.name = name
        self.description = description
        self.llm = llm
        self.metadata = metadata or {}

    @abstractmethod
    async def run(self, input_data: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def run_sync(self, input_data: Any, **kwargs: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(input_data, **kwargs))

        raise RuntimeError("run_sync cannot be used inside an active event loop; use await agent.run(...).")

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "llm_configured": bool(self.llm and self.llm.is_configured()),
            "metadata": self.metadata,
        }


__all__ = ["HelloAgent", "HelloAgentsLLM"]