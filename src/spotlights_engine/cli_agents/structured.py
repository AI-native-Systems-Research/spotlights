"""Structured output helpers for CLI-backed agents.

Structured output is a small contract shared by the adapters:

1. Use the CLI's native schema flag when available.
2. Add a concise JSON instruction to the prompt.
3. Parse JSON from the returned text or CLI JSON wrapper.
4. Optionally validate/coerce it with a Pydantic v2 model.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

JSONSchema = Mapping[str, Any]


@dataclass(frozen=True)
class StructuredOutputSpec:
    """JSON/Pydantic output contract for one agent request."""

    name: str
    json_schema: JSONSchema
    pydantic_model: type[Any] | None = None
    description: str | None = None

    @classmethod
    def from_json_schema(
        cls,
        *,
        name: str,
        json_schema: JSONSchema,
        description: str | None = None,
    ) -> StructuredOutputSpec:
        return cls(name=name, json_schema=json_schema, description=description)

    @classmethod
    def from_pydantic(
        cls,
        model: type[Any],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> StructuredOutputSpec:
        if not hasattr(model, "model_json_schema") or not hasattr(model, "model_validate"):
            raise TypeError("model must be a Pydantic v2 BaseModel subclass")
        schema = model.model_json_schema()
        return cls(
            name=name or getattr(model, "__name__", "structured_output"),
            json_schema=schema,
            pydantic_model=model,
            description=description,
        )

    def prompt_suffix(self) -> str:
        """Instruction appended to prompts for portable structured output."""
        description = f"\nTask-specific schema note: {self.description}" if self.description else ""
        schema = json.dumps(self.json_schema, ensure_ascii=False, sort_keys=True)
        return (
            "\n\nReturn only valid JSON matching this schema."
            "\nDo not wrap the JSON in Markdown fences."
            "\nDo not include commentary outside the JSON."
            f"{description}"
            f"\nJSON schema ({self.name}):\n{schema}"
        )

    def native_prompt_suffix(self) -> str:
        """Prompt instruction used when the schema is passed through a native CLI flag."""
        return "\n\nReturn only valid JSON matching the supplied output schema."

    def parse(self, *, text: str, parsed_json: Any | None = None) -> Any:
        data = extract_structured_json(text=text, parsed_json=parsed_json)
        if self.pydantic_model is None:
            return data
        return self.pydantic_model.model_validate(data)

    def native_json_schema(self) -> dict[str, Any]:
        """Schema form accepted by stricter native CLI/API structured-output paths."""
        return make_strict_json_schema(self.json_schema)


def make_strict_json_schema(schema: JSONSchema) -> dict[str, Any]:
    """Return a JSON Schema copy with closed object shapes.

    OpenAI-compatible structured output endpoints reject object schemas unless
    `additionalProperties: false` is explicit. Applying it recursively is also a
    safe default for Claude's `--json-schema` validation.
    """
    copied = deepcopy(dict(schema))
    _close_object_schemas(copied)
    return copied


def _close_object_schemas(node: Any) -> None:
    if isinstance(node, dict):
        node_type = node.get("type")
        has_properties = isinstance(node.get("properties"), dict)
        if node_type == "object" or has_properties:
            node.setdefault("additionalProperties", False)
            if has_properties:
                node["required"] = list(node["properties"])
        for value in node.values():
            _close_object_schemas(value)
    elif isinstance(node, list):
        for item in node:
            _close_object_schemas(item)


def apply_structured_prompt(
    prompt: str,
    spec: StructuredOutputSpec | None,
    *,
    include_schema: bool = True,
) -> str:
    if spec is None:
        return prompt
    if not include_schema:
        return f"{prompt}{spec.native_prompt_suffix()}"
    return f"{prompt}{spec.prompt_suffix()}"


def extract_structured_json(*, text: str, parsed_json: Any | None = None) -> Any:
    """Extract the most likely structured JSON payload from CLI output."""
    if parsed_json is not None:
        from_payload = _structured_from_payload(parsed_json)
        if from_payload is not _Missing:
            return from_payload
    return parse_json_from_text(text)


def parse_json_from_text(text: str) -> Any:
    """Parse a JSON value from raw model text, including simple fenced output."""
    stripped = _strip_markdown_json_fence(text.strip())
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    candidate = _first_balanced_json_value(stripped)
    if candidate is None:
        raise ValueError("structured output did not contain a JSON object or array")
    return json.loads(candidate)


class _MissingType:
    pass


_Missing = _MissingType()


def _structured_from_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        if "structured_output" in payload:
            return payload["structured_output"]
        wrapper_value = _text_from_wrapper(payload)
        if wrapper_value is not None:
            return parse_json_from_text(wrapper_value)
        if _has_wrapper_markers(payload):
            return _Missing
        return payload
    if isinstance(payload, list):
        for event in reversed(payload):
            if not isinstance(event, dict):
                continue
            try:
                candidate = _structured_from_payload(event)
            except ValueError:
                continue
            if candidate is not _Missing:
                return candidate
        return _Missing
    if isinstance(payload, str):
        return parse_json_from_text(payload)
    return _Missing


def _text_from_wrapper(payload: Mapping[str, Any]) -> str | None:
    """Return text from known CLI wrapper payloads, not arbitrary dicts."""
    if not _has_wrapper_markers(payload):
        if len(payload) != 1:
            return None
    for key in ("result", "response", "text", "message", "content", "answer"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return None


def _has_wrapper_markers(payload: Mapping[str, Any]) -> bool:
    wrapper_markers = {
        "session_id",
        "sessionId",
        "type",
        "subtype",
        "is_error",
        "returncode",
        "stats",
        "usage",
        "modelUsage",
    }
    return bool(wrapper_markers.intersection(payload))


def _strip_markdown_json_fence(text: str) -> str:
    if not text.startswith("```") or not text.endswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) < 3:
        return text
    first = lines[0].strip().lower()
    if first not in {"```", "```json"}:
        return text
    return "\n".join(lines[1:-1]).strip()


def _first_balanced_json_value(text: str) -> str | None:
    starts = [idx for idx in (text.find("{"), text.find("[")) if idx >= 0]
    if not starts:
        return None
    start = min(starts)
    stack: list[str] = []
    in_string = False
    escaped = False

    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or stack[-1] != char:
                return None
            stack.pop()
            if not stack:
                return text[start : idx + 1]
    return None
