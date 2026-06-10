"""Shared helpers for invoking an agent with structured output and a graceful fallback.

The Portfolio Manager, Trader, and Research Manager all follow the same
canonical pattern:

1. At agent creation, wrap the LLM with ``with_structured_output(Schema)``
   so the model returns a typed Pydantic instance. If the provider does
   not support structured output (rare; mostly older Ollama models), the
   wrap is skipped and the agent uses free-text generation instead.
2. At invocation, run the structured call and render the result back to
   markdown. If the structured call itself fails for any reason
   (malformed JSON from a weak model, transient provider issue), a
   multi-tier recovery chain attempts to reconstruct the typed output
   from the free-text response before falling back to raw prose.

The four-tier degradation chain:

  Tier 1  structured_llm.invoke() → render()
  Tier 2  plain text → extract embedded JSON → schema.model_validate() → render()
  Tier 3  plain text → per-schema heuristic extractor (``recover``) → render()
  Tier 4  plain text returned as-is (pipeline never blocks)

Centralising the pattern here keeps the agent factories small and ensures
all agents log the same warnings when fallback fires.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Optional, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# Internal helpers for Tier-2 JSON recovery
# ---------------------------------------------------------------------------

# Matches a fenced code block (```json ... ``` or ``` ... ```).
_FENCED_JSON_RE = re.compile(
    r"```(?:json)?\s*\n?(.*?)```", re.DOTALL,
)

# Matches the outermost { … } in a string (greedy).
_BRACED_RE = re.compile(r"\{.*\}", re.DOTALL)

# Trailing comma before a closing brace/bracket — common LLM artifact.
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _extract_json_block(text: str) -> Optional[str]:
    """Best-effort extraction of a JSON object from potentially dirty text.

    Handles markdown code fences, leading/trailing prose, and trailing
    commas.  Returns the cleaned JSON string or ``None``.
    """
    # Priority 1: fenced code block
    m = _FENCED_JSON_RE.search(text)
    candidate = m.group(1).strip() if m else None

    # Priority 2: outermost braces in the raw text
    if candidate is None:
        m2 = _BRACED_RE.search(text)
        candidate = m2.group(0) if m2 else None

    if candidate is None:
        return None

    # Clean trailing commas before } or ]
    candidate = _TRAILING_COMMA_RE.sub(r"\1", candidate)
    return candidate


def _try_parse_json(text: str, schema: type[T]) -> Optional[T]:
    """Try to parse *text* as (possibly dirty) JSON and validate against *schema*.

    Returns a validated Pydantic instance or ``None`` on any failure.
    """
    blob = _extract_json_block(text)
    if blob is None:
        return None
    try:
        data = json.loads(blob)
        return schema.model_validate(data)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def bind_structured(llm: Any, schema: type[T], agent_name: str) -> Optional[Any]:
    """Return ``llm.with_structured_output(schema)`` or ``None`` if unsupported.

    Logs a warning when the binding fails so the user understands the agent
    will use free-text generation for every call instead of one-shot fallback.
    """
    try:
        return llm.with_structured_output(schema)
    except (NotImplementedError, AttributeError) as exc:
        logger.warning(
            "%s: provider does not support with_structured_output (%s); "
            "falling back to free-text generation",
            agent_name, exc,
        )
        return None


def invoke_structured_or_freetext(
    structured_llm: Optional[Any],
    plain_llm: Any,
    prompt: Any,
    render: Callable[[T], str],
    agent_name: str,
    *,
    schema: Optional[type[T]] = None,
    recover: Optional[Callable[[str], Optional[T]]] = None,
) -> str:
    """Run the structured call and render to markdown, with multi-tier fallback.

    ``prompt`` is whatever the underlying LLM accepts (a string for chat
    invocations, a list of message dicts for chat models that take that
    shape). The same value is forwarded to the free-text path so the
    fallback sees the same input the structured call did.

    Optional keyword arguments enable deeper recovery when the structured
    call fails:

    *schema*
        Pydantic model class.  When provided, Tier 2 attempts to extract
        embedded JSON from the free-text response and validate it.

    *recover*
        A callable ``(str) -> schema_instance | None``.  When provided,
        Tier 3 uses per-schema heuristics to reconstruct the typed output
        from free-text prose (e.g. regex extraction of section headers).
    """
    # --- Tier 1: native structured output ---
    if structured_llm is not None:
        try:
            result = structured_llm.invoke(prompt)
            return render(result)
        except Exception as exc:
            logger.warning(
                "%s: structured-output invocation failed (%s); "
                "retrying once as free text",
                agent_name, exc,
            )

    response = plain_llm.invoke(prompt)
    content: str = response.content

    # --- Tier 2: JSON recovery from free text ---
    if schema is not None:
        instance = _try_parse_json(content, schema)
        if instance is not None:
            logger.info(
                "%s: recovered structured output from dirty JSON", agent_name,
            )
            return render(instance)

    # --- Tier 3: per-schema heuristic extraction ---
    if recover is not None:
        try:
            instance = recover(content)
        except Exception:
            instance = None
        if instance is not None:
            logger.info(
                "%s: recovered structured output via heuristic extraction",
                agent_name,
            )
            return render(instance)

    # --- Tier 4: raw free text (pipeline never blocks) ---
    return content
