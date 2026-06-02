from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ToolRequest:
    request_id: str
    payload: Dict[str, Any]
    timeout_ms: int


@dataclass
class ToolResponse:
    ok: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ToolClient:
    def call(self, tool_name: str, request: ToolRequest) -> ToolResponse:
        raise NotImplementedError
