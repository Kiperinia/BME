"""LLM 适配层：优先复用 hello_agents.HelloAgentsLLM，保留 ModelScope 与无 LLM 规则模式支持。"""

from __future__ import annotations

from typing import Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency in rule-only mode
    OpenAI = None

from hello_agents import HelloAgentsLLM

from core.config import Config


class RuleOnlyLLM(HelloAgentsLLM):
    """brief:
        Represent RuleOnlyLLM state and behavior.

    parameter:
        - model: Input value for model.
        - provider: Input value for provider.

    retrival:
        - Provides instances used by the surrounding workflow.
    """

    def __init__(self, model: str = "rule-only", provider: str = "rule-only"):
        """brief:
            Initialize this object.

        parameter:
            - model: Input value for model.
            - provider: Input value for provider.

        retrival:
            - Returns None; performs side effects described in the brief section.
        """
        self.model = model
        self.provider = provider
        self.api_key = None
        self.base_url = None
        self.temperature = 0.0
        self.max_tokens = None
        self.timeout = None
        self.kwargs = {}
        self.last_call_stats = None

    def chat(self, messages, temperature: float | None = None, max_tokens: int | None = None) -> str:
        """brief:
            Handle chat.

        parameter:
            - messages: Input value for messages.
            - temperature: Input value for temperature.
            - max_tokens: Input value for max_tokens.

        retrival:
            - Returns None; performs side effects described in the brief section.
        """
        raise RuntimeError("RuleOnlyLLM does not support chat; enable a real LLM provider first.")


class MyLLM(HelloAgentsLLM):
    """brief:
        Represent MyLLM state and behavior.

    parameter:
        - config: Input value for config.
        - model: Input value for model.
        - api_key: Input value for api_key.
        - base_url: Input value for base_url.
        - provider: Input value for provider.
        - **kwargs: Input value for kwargs.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    def __init__(
        self,
        config: Optional[Config] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = "auto",
        **kwargs,
    ):
        """brief:
            Initialize this object.

        parameter:
            - config: Input value for config.
            - model: Input value for model.
            - api_key: Input value for api_key.
            - base_url: Input value for base_url.
            - provider: Input value for provider.
            - **kwargs: Input value for kwargs.

        retrival:
            - Returns None; performs side effects described in the brief section.
        """
        resolved_config = config or Config.from_env()
        resolved_provider = self._resolve_provider(provider, resolved_config)
        resolved_model = model or resolved_config.default_model
        resolved_base_url = base_url or resolved_config.base_url
        if resolved_provider == "deepseek":
            if not resolved_base_url:
                resolved_base_url = "https://api.deepseek.com/v1"
            if not resolved_model:
                resolved_model = "deepseek-chat"
        elif resolved_provider == "tencent":
            # Hunyuan provides an OpenAI-compatible endpoint.
            if not resolved_base_url:
                resolved_base_url = "https://api.hunyuan.cloud.tencent.com/v1"
            if not resolved_model:
                resolved_model = "hunyuan-turbos-latest"
        resolved_temperature = kwargs.pop("temperature", resolved_config.temperature)
        resolved_max_tokens = kwargs.pop("max_tokens", resolved_config.max_tokens)
        resolved_timeout = kwargs.pop("timeout", resolved_config.timeout)

        if resolved_provider == "modelscope":
            if OpenAI is None:
                raise RuntimeError("openai package is required when provider='modelscope'.")

            self.provider = "modelscope"
            self.api_key = api_key or resolved_config.modelscope_api_key
            self.base_url = base_url or resolved_config.modelscope_base_url
            if not self.api_key:
                raise ValueError(
                    "ModelScope API key not found. Please set MODELSCOPE_API_KEY or fill modelscope_api_key in agent/config/llm_profiles.json."
                )

            self.model = resolved_model or "Qwen/Qwen2.5-VL-72B-Instruct"
            self.temperature = resolved_temperature
            self.max_tokens = resolved_max_tokens
            self.timeout = resolved_timeout
            self.kwargs = kwargs
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        else:
            super().__init__(
                model=resolved_model,
                api_key=api_key or resolved_config.api_key,
                base_url=resolved_base_url,
                temperature=resolved_temperature,
                max_tokens=resolved_max_tokens,
                timeout=resolved_timeout,
                **kwargs,
            )
            self.provider = resolved_provider

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """brief:
            Handle chat.

        parameter:
            - messages: Input value for messages.
            - temperature: Input value for temperature.
            - max_tokens: Input value for max_tokens.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        if hasattr(self, "_client") and self.provider == "modelscope":
            request = {
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

        response = self.invoke(
            messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
        )
        return response.content

    @staticmethod
    def _stringify_content(content) -> str:
        """brief:
            Handle stringify content.

        parameter:
            - content: Input value for content.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
        return "" if content is None else str(content)

    @staticmethod
    def _resolve_provider(provider: Optional[str], config: Config) -> str:
        """brief:
            Resolve provider.

        parameter:
            - provider: Input value for provider.
            - config: Input value for config.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        if provider and provider != "auto":
            return provider
        return config.default_provider


__all__ = ["HelloAgentsLLM", "MyLLM", "RuleOnlyLLM"]
