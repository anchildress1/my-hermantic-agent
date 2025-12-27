import inspect
import json
import re
from typing import List, Dict, Any, Callable


def get_function_schema(func: Callable) -> Dict[str, Any]:
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
    arg_descriptions = {}
    if "Args:" in doc:
        args_section = doc.split("Args:")[1].split("Returns:")[0]
        for line in args_section.split("\n"):
            line = line.strip()
            if ":" in line:
                name, desc = line.split(":", 1)
                arg_descriptions[name.strip()] = desc.strip()

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


def format_tools_xml(tools: List[Callable]) -> str:
    """Format tools list into Hermes XML system message addition."""
    schemas = [get_function_schema(t) for t in tools]
    tools_json = [json.dumps(s) for s in schemas]
    return "<tools>\n" + "\n".join(tools_json) + "\n</tools>"


def parse_tool_calls(content: str) -> List[Dict[str, Any]]:
    """
    Parse <tool_call>...</tool_call> from response.
    Returns list of dicts: {"name": str, "arguments": dict}
    """
    pattern = r"<tool_call>(.*?)</tool_call>"
    matches = re.findall(pattern, content, re.DOTALL)

    calls = []
    for match in matches:
        try:
            # The content inside <tool_call> is assumed to be JSON like:
            # {"name": "func", "arguments": {...}}
            # OR {"type": "function", "function": {"name":..., "arguments":...}}
            # Hermes usually outputs: {"name": "func_name", "arguments": { ... }}
            # Let's try to parse it.
            data = json.loads(match)

            # Normalize to structure: {"name": str, "arguments": dict}
            if "name" in data and "arguments" in data:
                calls.append(data)
            elif "function" in data:  # OpenAI format inside tool_call?
                calls.append(
                    {
                        "name": data["function"]["name"],
                        "arguments": json.loads(data["function"]["arguments"])
                        if isinstance(data["function"]["arguments"], str)
                        else data["function"]["arguments"],
                    }
                )
        except json.JSONDecodeError:
            continue

    return calls
