"""工具基类：直接复用 hello_agents 包内实现。"""

from hello_agents.tools import Tool, ToolErrorCode, ToolParameter, ToolResponse, ToolStatus, tool_action

__all__ = [
	"Tool",
	"ToolParameter",
	"ToolResponse",
	"ToolStatus",
	"ToolErrorCode",
	"tool_action",
]