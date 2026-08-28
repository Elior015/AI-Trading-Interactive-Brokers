"""The LLM boundary: provider protocol, request/response types, schema helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class StructuredStrategy(str, Enum):
    """How we coax valid JSON out of a given model+host pair.

    Which one works depends on the deployment, not just the model: Ollama Cloud
    does not support schema-constrained decoding at all, so `NATIVE_SCHEMA`
    silently degrades there and one of the others has to carry the load.
    """

    #: `format=<JSON Schema>` grammar-constrained decoding. Local Ollama only.
    NATIVE_SCHEMA = "native_schema"
    #: `format="json"` plus the schema written into the prompt.
    JSON_MODE = "json_mode"
    #: Native tool calling with a single tool whose parameters are the schema.
    SINGLE_TOOL = "single_tool"
    #: Free text, with JSON recovered from fences or Hermes `<tool_call>` tags.
    TEXT_EXTRACT = "text_extract"


#: Tried in this order during capability negotiation.
STRATEGY_ORDER = [
    StructuredStrategy.NATIVE_SCHEMA,
    StructuredStrategy.JSON_MODE,
    StructuredStrategy.SINGLE_TOOL,
    StructuredStrategy.TEXT_EXTRACT,
]


@dataclass
class LLMRequest:
    model: str
    messages: list[dict[str, Any]]
    #: JSON Schema the response must satisfy.
    schema: dict[str, Any] | None = None
    strategy: StructuredStrategy = StructuredStrategy.JSON_MODE
    temperature: float = 0.0
    num_ctx: int = 32768
    num_predict: int = 2048
    seed: int | None = 42
    keep_alive: str = "30m"
    think: bool | str | None = None
    timeout: float = 180.0
    #: Name shown to the model when the SINGLE_TOOL strategy is used.
    tool_name: str = "submit_response"

    def options(self) -> dict[str, Any]:
        """Ollama `options`. `num_ctx` is always explicit.

        The default context is 4096, which a 20-symbol feature table plus a
        session narrative overflows — silently, by truncating the front of the
        prompt. An explicit value is the only way to know what the model saw.
        """
        opts: dict[str, Any] = {
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
        }
        if self.seed is not None:
            opts["seed"] = self.seed
        return opts


@dataclass
class LLMResponse:
    content: str = ""
    thinking: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)
    #: True when this came from the replay cache rather than the model.
    from_cache: bool = False


class LLMError(RuntimeError):
    """Provider-level failure: transport, timeout, auth, or quota."""


class LLMQuotaExhausted(LLMError):
    """The account's Ollama Cloud quota or concurrency budget is spent."""


class LLMSchemaError(RuntimeError):
    """The model could not produce output matching the schema, even after repair."""


@dataclass
class ModelCapabilities:
    model: str
    capabilities: list[str] = field(default_factory=list)

    @property
    def supports_tools(self) -> bool:
        return "tools" in self.capabilities

    @property
    def supports_thinking(self) -> bool:
        return "thinking" in self.capabilities


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    host: str

    async def chat(self, req: LLMRequest) -> LLMResponse: ...
    async def show(self, model: str) -> ModelCapabilities: ...
    async def healthy(self) -> bool: ...


# --------------------------------------------------------------------------- #
# Schema helpers
# --------------------------------------------------------------------------- #


def flatten_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline `$ref`/`$defs` into a single self-contained schema.

    Models handle a flat schema far better than one with internal references,
    and the referenced definitions are invisible to a model reading the schema
    as prose in the prompt.
    """
    defs = schema.get("$defs") or schema.get("definitions") or {}

    def resolve(node: Any, depth: int = 0) -> Any:
        if depth > 12:  # cycle guard
            return {}
        if isinstance(node, dict):
            if "$ref" in node:
                ref = node["$ref"]
                key = ref.split("/")[-1]
                target = defs.get(key)
                if target is None:
                    return {}
                merged = resolve(target, depth + 1)
                extra = {k: v for k, v in node.items() if k != "$ref"}
                if isinstance(merged, dict):
                    return {**merged, **extra}
                return merged
            return {
                k: resolve(v, depth + 1)
                for k, v in node.items()
                if k not in ("$defs", "definitions")
            }
        if isinstance(node, list):
            return [resolve(v, depth + 1) for v in node]
        return node

    out = resolve(schema)
    return out if isinstance(out, dict) else schema


def describe_schema(schema: dict[str, Any]) -> str:
    """Render a schema for embedding in a prompt.

    Ollama's own guidance is to put the schema in the prompt as well as in
    `format`, and on Ollama Cloud — where `format` does nothing — the prompt
    copy is the only thing steering the model at all.
    """
    return json.dumps(flatten_schema(schema), indent=2, ensure_ascii=False)


def schema_to_tool(schema: dict[str, Any], name: str, description: str) -> dict[str, Any]:
    """Wrap a schema as an Ollama tool definition."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": flatten_schema(schema),
        },
    }
