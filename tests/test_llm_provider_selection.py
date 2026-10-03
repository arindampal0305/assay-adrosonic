"""Proof that `get_client()` picks the provider the environment actually asks for.

The selection logic is three lines of `if`, which is exactly the kind of code that
looks too simple to test and then silently costs a milestone: `.env` went unread
for one because nothing asserted that a configured key produced a client.

Nothing here touches the network. `OllamaClient` reaches the daemon through
`urllib.request.urlopen`, so that one function is stubbed and the real client, the
real probe and the real request-body construction all run against it. Stubbing the
transport rather than the client is deliberate — a test that patched
`OllamaClient._probe` would pass even if the probe asked the wrong URL.
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from backend.llm import client as client_mod
from backend.llm.client import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    LLMClient,
    LLMUnavailable,
    OllamaClient,
    OpenAIClient,
    get_client,
)

LLM_VARS = (
    "OPENAI_API_KEY",
    "ASSAY_LLM_PROVIDER",
    "OLLAMA_BASE_URL",
    "ASSAY_OLLAMA_MODEL",
)


@pytest.fixture
def clean_env(monkeypatch):
    """No LLM configuration at all, so each test states its own.

    Without this the suite's result depends on the developer's real shell, which
    is how a provider test ends up passing only on the machine that wrote it.
    """
    for name in LLM_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


class FakeHTTP:
    """Stands in for `urllib.request.urlopen`, recording what it was asked."""

    def __init__(self, *, tags=("llama3.1:latest",), chat_content='{"target": null}'):
        self.tags = tags
        self.chat_content = chat_content
        self.requests: list[dict] = []

    def __call__(self, request, timeout=None):
        url = request if isinstance(request, str) else request.full_url
        body = None
        if not isinstance(request, str) and request.data:
            body = json.loads(request.data.decode("utf-8"))
        self.requests.append({"url": url, "timeout": timeout, "body": body})

        if url.endswith("/api/tags"):
            payload = {"models": [{"name": name} for name in self.tags]}
        elif url.endswith("/api/chat"):
            payload = {
                "model": "llama3.1",
                "message": {"role": "assistant", "content": self.chat_content},
                "prompt_eval_count": 611,
                "eval_count": 42,
            }
        else:  # pragma: no cover - a URL this client should never request
            raise AssertionError(f"unexpected request to {url}")

        return _Response(json.dumps(payload).encode("utf-8"))


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# --------------------------------------------------------------------------
# Provider selection
# --------------------------------------------------------------------------


def test_nothing_configured_returns_none(clean_env):
    """The degraded path the adjudicator_unavailable issue depends on."""
    assert get_client() is None


def test_openai_key_alone_selects_openai(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "sk-placeholder")
    assert isinstance(get_client(), OpenAIClient)


def test_provider_ollama_selects_ollama(clean_env, monkeypatch):
    clean_env.setenv("ASSAY_LLM_PROVIDER", "ollama")
    http = FakeHTTP()
    monkeypatch.setattr(client_mod.urllib.request, "urlopen", http)

    client = get_client()
    assert isinstance(client, OllamaClient)
    assert client.base_url == DEFAULT_OLLAMA_BASE_URL
    assert client.model == DEFAULT_OLLAMA_MODEL
    # Selection must have actually probed the daemon, not just constructed.
    assert http.requests[0]["url"] == f"{DEFAULT_OLLAMA_BASE_URL}/api/tags"


def test_ollama_honours_base_url_and_model_from_env(clean_env, monkeypatch):
    clean_env.setenv("ASSAY_LLM_PROVIDER", "ollama")
    # Trailing slash included on purpose: it must not produce `//api/chat`.
    clean_env.setenv("OLLAMA_BASE_URL", "http://ollama.internal:9999/")
    clean_env.setenv("ASSAY_OLLAMA_MODEL", "mistral")
    http = FakeHTTP(tags=("mistral:latest",))
    monkeypatch.setattr(client_mod.urllib.request, "urlopen", http)

    client = get_client()
    assert isinstance(client, OllamaClient)
    assert client.base_url == "http://ollama.internal:9999"
    assert client.model == "mistral"
    assert http.requests[0]["url"] == "http://ollama.internal:9999/api/tags"


def test_openai_wins_when_both_are_configured(clean_env, monkeypatch, caplog):
    """Documented precedence. Silently reaching for a paid provider when the
    environment explicitly asked for a local one must at least be audible."""
    clean_env.setenv("OPENAI_API_KEY", "sk-placeholder")
    clean_env.setenv("ASSAY_LLM_PROVIDER", "ollama")
    probed = FakeHTTP()
    monkeypatch.setattr(client_mod.urllib.request, "urlopen", probed)

    client = get_client()

    assert isinstance(client, OpenAIClient)
    # Ollama must not even be contacted when OpenAI is taking the request.
    assert probed.requests == []
    assert "takes precedence" in caplog.text


def test_unknown_provider_returns_none(clean_env, caplog):
    clean_env.setenv("ASSAY_LLM_PROVIDER", "anthropic-but-typo")
    assert get_client() is None
    assert "unknown ASSAY_LLM_PROVIDER" in caplog.text


def test_blank_openai_key_is_treated_as_absent(clean_env):
    """A `.env` with `OPENAI_API_KEY=` present-but-empty is the common case, and
    must not construct a client that 401s on every column."""
    clean_env.setenv("OPENAI_API_KEY", "   ")
    assert get_client() is None


# --------------------------------------------------------------------------
# Ollama availability falls back rather than failing per column
# --------------------------------------------------------------------------


def test_unreachable_ollama_returns_none(clean_env, monkeypatch):
    """Configured but down must be `None`, not a client that raises 2x per run."""
    clean_env.setenv("ASSAY_LLM_PROVIDER", "ollama")

    def refuse(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(client_mod.urllib.request, "urlopen", refuse)
    assert get_client() is None


def test_missing_model_returns_none_and_names_the_pull(clean_env, monkeypatch):
    clean_env.setenv("ASSAY_LLM_PROVIDER", "ollama")
    clean_env.setenv("ASSAY_OLLAMA_MODEL", "llama3.1")
    monkeypatch.setattr(
        client_mod.urllib.request, "urlopen", FakeHTTP(tags=("qwen2:7b",))
    )
    assert get_client() is None

    # The message a developer has to act on, asserted directly.
    with pytest.raises(LLMUnavailable, match=r"ollama pull llama3\.1"):
        OllamaClient(model="llama3.1")


def test_tag_suffix_counts_as_the_same_model(clean_env, monkeypatch):
    """`llama3.1` and `llama3.1:latest` are one model; requiring the exact string
    would reject a correctly pulled daemon."""
    monkeypatch.setattr(
        client_mod.urllib.request, "urlopen", FakeHTTP(tags=("llama3.1:latest",))
    )
    assert OllamaClient(model="llama3.1").model == "llama3.1"


# --------------------------------------------------------------------------
# Protocol conformance and request shape
# --------------------------------------------------------------------------


def test_ollama_client_satisfies_the_protocol(monkeypatch):
    monkeypatch.setattr(client_mod.urllib.request, "urlopen", FakeHTTP())
    assert isinstance(OllamaClient(), LLMClient)


def test_complete_json_posts_json_mode_and_parses_usage(monkeypatch):
    http = FakeHTTP(chat_content='{"target": "Construction"}')
    monkeypatch.setattr(client_mod.urllib.request, "urlopen", http)

    response = OllamaClient(model="llama3.1").complete_json(
        system="SYSTEM", user='{"source_column": "Const"}'
    )

    assert response.as_json() == {"target": "Construction"}
    assert response.prompt_tokens == 611
    assert response.completion_tokens == 42

    chat = next(r for r in http.requests if r["url"].endswith("/api/chat"))
    body = chat["body"]
    # JSON mode and determinism are the two settings the verifier depends on.
    assert body["format"] == "json"
    assert body["options"]["temperature"] == 0
    assert body["stream"] is False
    assert body["messages"] == [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": '{"source_column": "Const"}'},
    ]


def test_complete_json_retries_then_raises(monkeypatch):
    monkeypatch.setattr(client_mod.urllib.request, "urlopen", FakeHTTP())
    client = OllamaClient(model="llama3.1")

    attempts = {"n": 0}

    def fail(request, timeout=None):
        attempts["n"] += 1
        raise urllib.error.URLError("daemon died mid-run")

    monkeypatch.setattr(client_mod.urllib.request, "urlopen", fail)
    with pytest.raises(LLMUnavailable, match="after 2 attempts"):
        client.complete_json(system="s", user="u")
    assert attempts["n"] == client_mod.MAX_ATTEMPTS
