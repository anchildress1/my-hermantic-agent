import inspect
import json
import re
from json import JSONDecodeError
from typing import Any, Callable


def _parse_json_object(raw_value: str) -> dict[str, Any] | None:
    """Parse JSON and return object payloads only."""
    parsed_value = json.loads(raw_value)
    if isinstance(parsed_value, dict):
        return parsed_value
    return None


def _parse_google_arg_descriptions(docstring: str) -> dict[str, str]:
    """Extract Google-style arg descriptions from a docstring."""
    arg_descriptions: dict[str, str] = {}
    if "Args:" not in docstring:
        return arg_descriptions

    args_section = docstring.split("Args:")[1].split("Returns:")[0]
    for line in args_section.split("\n"):
        stripped_line = line.strip()
        if ":" not in stripped_line:
            continue
        name, description = stripped_line.split(":", 1)
        arg_descriptions[name.strip()] = description.strip()

    return arg_descriptions


def _resolve_call_arguments(raw_arguments: Any) -> dict[str, Any] | None:
    """Normalize tool-call arguments from dict or JSON string."""
    if isinstance(raw_arguments, dict):
        return raw_arguments

    if isinstance(raw_arguments, str):
        return _parse_json_object(raw_arguments)

    return None


def _normalize_tool_call(data: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize tool call payloads across Hermes/OpenAI formats."""
    if "name" in data and "arguments" in data and isinstance(data["name"], str):
        arguments = _resolve_call_arguments(data["arguments"])
        if arguments is None:
            return None
        return {"name": data["name"], "arguments": arguments}

    function_block = data.get("function")
    if not isinstance(function_block, dict):
        return None

    function_name = function_block.get("name")
    arguments = _resolve_call_arguments(function_block.get("arguments"))
    if not isinstance(function_name, str) or arguments is None:
        return None

    return {"name": function_name, "arguments": arguments}


def get_function_schema(func: Callable) -> dict[str, Any]:
    """
    Generate OpenAI-compatible tool JSON schema from a Python function.
    Assumes Google-style docstrings and type hints.
    """
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""

    # Parse docstring for description and args
    description = doc.split("\n\n")[0].strip()

    # Simple regex to parse Google-style args
    # Args:
    #     name: description
    arg_descriptions = _parse_google_arg_descriptions(doc)

    parameters = {"type": "object", "properties": {}, "required": []}

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        param_type = "string"  # default
        if param.annotation is float:
            param_type = "number"
        elif param.annotation is int:
            param_type = "integer"
        elif param.annotation is bool:
            param_type = "boolean"

        param_desc = arg_descriptions.get(param_name, "")

        parameters["properties"][param_name] = {
            "type": param_type,
            "description": param_desc,
        }

        if param.default == inspect.Parameter.empty:
            parameters["required"].append(param_name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": parameters,
        },
    }


def format_tools_xml(tools: list[Callable]) -> str:
    """Format tools list into Hermes XML system message addition."""
    schemas = [get_function_schema(t) for t in tools]
    tools_json = [json.dumps(s) for s in schemas]
    return "<tools>\n" + "\n".join(tools_json) + "\n</tools>"


def parse_tool_calls(content: str) -> list[dict[str, Any]]:
    """
    Parse <tool_call>...</tool_call> from response.
    Returns list of dicts: {"name": str, "arguments": dict}
    """
    pattern = r"<tool_call>(.*?)</tool_call>"
    matches = re.findall(pattern, content, re.DOTALL)

    calls: list[dict[str, Any]] = []
    for match in matches:
        try:
            raw_data = _parse_json_object(match)
            if raw_data is None:
                continue
            normalized = _normalize_tool_call(raw_data)
            if normalized:
                calls.append(normalized)
        except JSONDecodeError:
            continue

    return calls
