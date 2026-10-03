"""LLM provider boundary.

Deliberately thin, and deliberately fallible. Two properties matter more than
features here:

1. **It can be absent.** No key configured is a normal state, not an error state.
   `get_client()` returns None and callers carry on without adjudication.
2. **It can be substituted.** The adjudicator depends on the `LLMClient` protocol,
   never on `openai`, so a stub client drives the whole adjudication path in tests
   without a network call or a key.

Nothing in this module interprets or trusts a response. Parsing and verification
live in the adjudicator, where the evidence to check a claim against is in scope.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("ASSAY_LLM_MODEL", "gpt-4o-mini")
DEFAULT_TIMEOUT_SECONDS = float(os.getenv("ASSAY_LLM_TIMEOUT", "20"))
# One retry only. A column mapping is not worth a long retry storm, and the
# fused-evidence answer is already a usable fallback.
MAX_ATTEMPTS = 2


class LLMUnavailable(RuntimeError):
    """No provider is configured, or the provider could not be reached."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    # Kept so an audit record can show what was actually asked, not a
    # reconstruction of it.
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    def as_json(self) -> dict[str, Any]:
        """Parse the response body as a JSON object.

        Raises ValueError rather than returning a default: a response that is not
        the requested shape must reach the caller as a failure, because silently
        substituting `{}` would read downstream as "the model declined", which is
        a different and misleading claim.
        """
        text = self.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1] if "```" in text[3:] else text[3:]
            if text.lstrip().startswith("json"):
                text = text.lstrip()[4:]
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
        return parsed


@runtime_checkable
class LLMClient(Protocol):
    """What the adjudicator needs. Any object with this shape will do."""

    def complete_json(self, *, system: str, user: str) -> LLMResponse: ...


class OpenAIClient:
    """Thin OpenAI adapter requesting a JSON object response."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMUnavailable("the 'openai' package is not installed") from exc
        self._client = OpenAI(api_key=api_key, timeout=DEFAULT_TIMEOUT_SECONDS)
        self.model = model

    def complete_json(self, *, system: str, user: str) -> LLMResponse:
        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                completion = self._client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
            except Exception as exc:  # noqa: BLE001 - provider errors are all equivalent here
                last_error = exc
                logger.warning("LLM attempt %d/%d failed: %s", attempt, MAX_ATTEMPTS, exc)
                continue
            usage = getattr(completion, "usage", None)
            return LLMResponse(
                text=completion.choices[0].message.content or "",
                model=self.model,
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
            )
        raise LLMUnavailable(f"LLM unreachable after {MAX_ATTEMPTS} attempts: {last_error}")


def get_client() -> Optional[LLMClient]:
    """The configured client, or None when no key is set.

    None is not a failure. Agent 2 resolves every column from fused evidence
    whether or not adjudication is available; adjudication only re-decides
    low-margin cases.
    """
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        logger.info("no OPENAI_API_KEY configured; adjudication disabled")
        return None
    try:
        return OpenAIClient(api_key=api_key)
    except LLMUnavailable as exc:
        logger.warning("LLM client could not be constructed: %s", exc)
        return None
