"""项目配置：优先从配置文件读取 LLM profile，并兼容环境变量覆盖。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from hello_agents.core.config import Config as HelloAgentsConfig


class Config(HelloAgentsConfig):
    """基于 hello_agents.Config 的项目配置。"""

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 60
    modelscope_api_key: Optional[str] = None
    modelscope_base_url: str = "https://api-inference.modelscope.cn/v1/"
    config_profile: Optional[str] = None
    config_path: Optional[str] = None

    # 覆盖 hello_agents 默认的磁盘路径，使所有持久化位于 agent/ 子目录下
    tool_output_dir: str = "agent/tool-output"
    trace_dir: str = "agent/memory/traces"
    skills_dir: str = "agent/skills"
    session_dir: str = "agent/memory/sessions"
    todowrite_persistence_dir: str = "agent/memory/todos"
    devlog_persistence_dir: str = "agent/memory/devlogs"

    @classmethod
    def from_env(cls) -> "Config":
        """从配置文件和环境变量创建配置，并保留 hello_agents 的默认字段。"""
        data = HelloAgentsConfig.from_env().dict()
        file_overrides, profile_name, config_path = cls._load_profile_overrides()
        data.update(file_overrides)
        data.update(cls._load_env_overrides(data))
        data["config_profile"] = profile_name
        data["config_path"] = str(config_path) if config_path is not None else None
        return cls(**data)

    @classmethod
    def _load_profile_overrides(cls) -> tuple[Dict[str, Any], Optional[str], Optional[Path]]:
        config_path = cls._resolve_config_path()
        if not config_path.exists():
            return {}, None, config_path

        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"LLM config file must be a JSON object: {config_path}")

        profiles = raw.get("profiles", {})
        if not isinstance(profiles, dict):
            raise ValueError(f"LLM config file 'profiles' must be an object: {config_path}")

        profile_name = os.getenv("LLM_CONFIG_PROFILE") or raw.get("active_profile")
        if not profile_name and len(profiles) == 1:
            profile_name = next(iter(profiles))

        if not profile_name:
            return {}, None, config_path

        if profile_name not in profiles:
            available = ", ".join(sorted(profiles)) or "<none>"
            raise ValueError(
                f"Selected LLM profile '{profile_name}' not found in {config_path}. Available profiles: {available}"
            )

        profile_data = profiles[profile_name]
        if not isinstance(profile_data, dict):
            raise ValueError(f"LLM profile '{profile_name}' must be an object: {config_path}")

        return profile_data, str(profile_name), config_path

    @staticmethod
    def _resolve_config_path() -> Path:
        custom_path = os.getenv("LLM_CONFIG_FILE")
        if custom_path:
            return Path(custom_path).expanduser().resolve()
        return Path(__file__).resolve().parents[1] / "config" / "llm_profiles.json"

    @staticmethod
    def _load_env_overrides(current: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "default_model": os.getenv("LLM_MODEL_ID", current.get("default_model", "gpt-3.5-turbo")),
            "default_provider": os.getenv("LLM_PROVIDER", current.get("default_provider", "openai")),
            "api_key": os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or current.get("api_key"),
            "base_url": os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or current.get("base_url"),
            "timeout": int(os.getenv("LLM_TIMEOUT", str(current.get("timeout", 60)))),
            "modelscope_api_key": os.getenv("MODELSCOPE_API_KEY") or current.get("modelscope_api_key"),
            "modelscope_base_url": os.getenv(
                "MODELSCOPE_BASE_URL",
                current.get("modelscope_base_url", "https://api-inference.modelscope.cn/v1/"),
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return self.dict()
