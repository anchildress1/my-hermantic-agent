from typing import List, Optional, Dict, Any
import msgspec


class FunctionCall(msgspec.Struct):
    name: str
    arguments: Dict[str, Any]


class ToolCall(msgspec.Struct):
    function: FunctionCall
    type: str = "function"


class ChatMessage(msgspec.Struct, kw_only=True):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    thinking: Optional[str] = None
    tool_name: Optional[str] = None  # For tool results


class Memory(msgspec.Struct, kw_only=True):
    text: str
    type: str = "fact"  # preference, fact, task, insight
    tag: str = "chat"
    importance: float = 1.0  # 0.0 to 3.0, > 2.0 requires user confirmation
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None


class AgentConfig(msgspec.Struct, kw_only=True):
    model: str
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 8192
    streaming: bool = True
