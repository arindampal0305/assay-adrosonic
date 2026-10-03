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
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("ASSAY_LLM_MODEL", "gpt-4o-mini")
DEFAULT_TIMEOUT_SECONDS = float(os.getenv("ASSAY_LLM_TIMEOUT", "20"))
# One retry only. A column mapping is not worth a long retry storm, and the
# fused-evidence answer is already a usable fallback.
MAX_ATTEMPTS = 2

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
# Ollama needs its own model setting rather than sharing `ASSAY_LLM_MODEL`. That
# variable's default is an OpenAI model name, and `.env.example` suggests
# `gpt-4o` for it, so reusing it would post `gpt-4o` to a local daemon and fail
# for a reason that reads like a bug rather than a misconfiguration.
DEFAULT_OLLAMA_MODEL = "llama3.1"
# Short on purpose: this probe runs during `get_client()`, on the request path.
OLLAMA_PROBE_TIMEOUT = float(os.getenv("ASSAY_OLLAMA_PROBE_TIMEOUT", "2"))


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


class OllamaClient:
    """Thin Ollama adapter. Local, keyless, and close enough to the same shape.

    Uses stdlib `urllib` rather than `httpx` or `requests`. `httpx` is present in
    this environment only as a transitive dependency of `openai`, and relying on
    transitive presence is exactly what left `.env` unread for a milestone (see
    `backend/__init__.py`). One JSON POST and one GET are not worth either a new
    declared dependency or a repeat of that.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        model: str = DEFAULT_OLLAMA_MODEL,
        *,
        probe: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        if probe:
            self._probe()

    def _probe(self) -> None:
        """Refuse construction unless the daemon answers and has the model.

        A configured-but-unreachable Ollama must collapse to `get_client() -> None`
        rather than to a client that raises once per low-margin column. Raising
        `LLMUnavailable` here reuses the fallback `get_client` already has, instead
        of introducing a second, parallel notion of "degraded".
        """
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/api/tags", timeout=OLLAMA_PROBE_TIMEOUT
            ) as response:
                tags = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise LLMUnavailable(
                f"no Ollama daemon answering at {self.base_url} ({exc})"
            ) from exc

        installed = [str(m.get("name", "")) for m in tags.get("models", [])]
        # Ollama reports tagged names, so `llama3.1` and `llama3.1:latest` are the
        # same model. Compare on the bare name as well as the exact string.
        if installed and not any(
            name == self.model or name.split(":")[0] == self.model.split(":")[0]
            for name in installed
        ):
            raise LLMUnavailable(
                f"Ollama at {self.base_url} has no model '{self.model}' "
                f"(installed: {', '.join(sorted(installed))}). "
                f"Run `ollama pull {self.model}`."
            )

    def complete_json(self, *, system: str, user: str) -> LLMResponse:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                # Ollama's JSON mode, the counterpart of OpenAI's
                # response_format={"type": "json_object"}. Without it a local model
                # tends to wrap the object in prose and `as_json` has to salvage it.
                "format": "json",
                "options": {"temperature": 0},
            }
        ).encode("utf-8")

        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            request = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=DEFAULT_TIMEOUT_SECONDS
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, OSError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "Ollama attempt %d/%d failed: %s", attempt, MAX_ATTEMPTS, exc
                )
                continue
            return LLMResponse(
                text=(payload.get("message") or {}).get("content") or "",
                model=str(payload.get("model") or self.model),
                prompt_tokens=payload.get("prompt_eval_count"),
                completion_tokens=payload.get("eval_count"),
            )
        raise LLMUnavailable(
            f"Ollama unreachable after {MAX_ATTEMPTS} attempts: {last_error}"
        )


def get_client() -> Optional[LLMClient]:
    """The configured client, or None when no provider is usable.

    Precedence, which is a decision rather than an accident:

      1. `OPENAI_API_KEY` non-empty        -> OpenAI
      2. `ASSAY_LLM_PROVIDER=ollama`       -> Ollama at `OLLAMA_BASE_URL`
      3. anything else                     -> None

    None is not a failure. Agent 2 resolves every column from fused evidence
    whether or not adjudication is available; adjudication only re-decides
    low-margin cases, and `schema_mapping` reports the skip as an
    `adjudicator_unavailable` issue plus an audit record naming the columns that
    went unadjudicated.

    Every variable is read here rather than at module import, for two reasons: a
    `.env` loaded by `backend/__init__` must be visible, and a test must be able
    to select a provider with `monkeypatch.setenv` without reimporting the module.
    """
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    provider = (os.getenv("ASSAY_LLM_PROVIDER") or "").strip().lower()

    if api_key:
        if provider == "ollama":
            # Asking for Ollama and silently getting OpenAI — and a bill — is the
            # one case in this function worth a warning rather than an info line.
            logger.warning(
                "ASSAY_LLM_PROVIDER=ollama but OPENAI_API_KEY is set, so OpenAI "
                "takes precedence. Unset OPENAI_API_KEY to use Ollama."
            )
        try:
            return OpenAIClient(api_key=api_key)
        except LLMUnavailable as exc:
            logger.warning("OpenAI client could not be constructed: %s", exc)
            return None

    if provider == "ollama":
        base_url = (os.getenv("OLLAMA_BASE_URL") or "").strip() or DEFAULT_OLLAMA_BASE_URL
        model = (os.getenv("ASSAY_OLLAMA_MODEL") or "").strip() or DEFAULT_OLLAMA_MODEL
        try:
            client = OllamaClient(base_url=base_url, model=model)
        except LLMUnavailable as exc:
            logger.warning("Ollama client could not be constructed: %s", exc)
            return None
        logger.info("adjudication via Ollama model '%s' at %s", model, base_url)
        return client

    if provider and provider != "openai":
        logger.warning(
            "unknown ASSAY_LLM_PROVIDER '%s'; expected 'openai' or 'ollama'. "
            "Adjudication disabled.",
            provider,
        )
    else:
        logger.info("no LLM provider configured; adjudication disabled")
    return None
