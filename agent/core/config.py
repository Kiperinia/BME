"""项目配置：扩展 hello_agents 包内 Config，补充项目自定义字段。"""

import os
from typing import Any, Dict, Optional

from hello_agents.core.config import Config as HelloAgentsConfig


class Config(HelloAgentsConfig):
    """基于 hello_agents.Config 的项目配置。"""

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 60
    modelscope_api_key: Optional[str] = None
    modelscope_base_url: str = "https://api-inference.modelscope.cn/v1/"

    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量创建配置，并保留 hello_agents 的默认字段。"""
        data = HelloAgentsConfig.from_env().dict()
        data.update(
            {
                "default_model": os.getenv("LLM_MODEL_ID", data.get("default_model", "gpt-3.5-turbo")),
                "default_provider": os.getenv("LLM_PROVIDER", data.get("default_provider", "openai")),
                "api_key": os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
                "base_url": os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
                "timeout": int(os.getenv("LLM_TIMEOUT", "60")),
                "modelscope_api_key": os.getenv("MODELSCOPE_API_KEY"),
                "modelscope_base_url": os.getenv(
                    "MODELSCOPE_BASE_URL",
                    "https://api-inference.modelscope.cn/v1/",
                ),
            }
        )
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return self.dict()
