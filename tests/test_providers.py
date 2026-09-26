"""Request/response shapes of the AI provider adapters, and settings key masking."""

import asyncio
import json

import httpx
import pytest

from app.settings import providers, store


def run(coro):
    return asyncio.run(coro)


def mock_client(handler):
    """AsyncClient whose requests go to handler(request) -> (status, json)."""
    seen = []

    def transport(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        status, body = handler(request)
        return httpx.Response(status, json=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(transport)), seen


def body(request):
    return json.loads(request.content)


# ---- Models -------------------------------------------------------------------------


def test_openai_models():
    client, seen = mock_client(lambda r: (200, {"data": [{"id": "gpt-b"}, {"id": "gpt-a"}]}))
    conn = {"provider": "openai", "apiKey": "sk-test"}
    assert run(providers.list_models(conn, client)) == ["gpt-a", "gpt-b"]
    assert str(seen[0].url) == "https://api.openai.com/v1/models"
    assert seen[0].headers["authorization"] == "Bearer sk-test"


def test_anthropic_models():
    client, seen = mock_client(lambda r: (200, {"data": [{"id": "claude-x"}]}))
    conn = {"provider": "anthropic", "apiKey": "key"}
    assert run(providers.list_models(conn, client)) == ["claude-x"]
    assert seen[0].url.path == "/v1/models"
    assert seen[0].headers["x-api-key"] == "key"
    assert seen[0].headers["anthropic-version"] == providers.ANTHROPIC_VERSION


def test_gemini_models_keeps_chat_models_only():
    models = {"models": [
        {"name": "models/gemini-pro", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/embedding", "supportedGenerationMethods": ["embedContent"]},
    ]}
    client, seen = mock_client(lambda r: (200, models))
    conn = {"provider": "gemini", "apiKey": "g"}
    assert run(providers.list_models(conn, client)) == ["gemini-pro"]
    assert seen[0].headers["x-goog-api-key"] == "g"


def test_ollama_models_without_key():
    client, seen = mock_client(lambda r: (200, {"models": [{"name": "llama3.2:latest"}]}))
    conn = {"provider": "ollama", "baseUrl": "http://localhost:11434/"}
    assert run(providers.list_models(conn, client)) == ["llama3.2:latest"]
    assert str(seen[0].url) == "http://localhost:11434/api/tags"
    assert "authorization" not in seen[0].headers


def test_ollama_cloud_sends_key():
    client, seen = mock_client(lambda r: (200, {"models": []}))
    run(providers.list_models({"provider": "ollama-cloud", "apiKey": "k"}, client))
    assert str(seen[0].url) == "https://ollama.com/api/tags"
    assert seen[0].headers["authorization"] == "Bearer k"


def test_perplexity_has_no_model_list():
    assert run(providers.list_models({"provider": "perplexity", "apiKey": "k"})) == []


def test_missing_key_is_friendly():
    with pytest.raises(providers.ProviderError, match="API key"):
        run(providers.list_models({"provider": "openai"}))


def test_rejected_key_is_friendly():
    client, _ = mock_client(lambda r: (401, {"error": {"message": "Incorrect API key"}}))
    with pytest.raises(providers.ProviderError, match="API key rejected.*Incorrect API key"):
        run(providers.list_models({"provider": "openrouter", "apiKey": "bad"}, client))


# ---- Chat ----------------------------------------------------------------------------

HISTORY = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
           {"role": "user", "content": "time?"}]


def test_openai_chat():
    client, seen = mock_client(lambda r: (200, {"choices": [{"message": {"content": " Noon. "}}]}))
    conn = {"provider": "openai", "apiKey": "k", "model": "gpt-x"}
    assert run(providers.chat(conn, HISTORY, system="be brief", max_tokens=100, client=client)) == "Noon."
    sent = body(seen[0])
    assert seen[0].url.path == "/v1/chat/completions"
    assert sent["messages"][0] == {"role": "system", "content": "be brief"}
    assert sent["max_completion_tokens"] >= 100 and "max_tokens" not in sent


def test_compatible_chat_uses_max_tokens():
    client, seen = mock_client(lambda r: (200, {"choices": [{"message": {"content": "ok"}}]}))
    conn = {"provider": "groq", "apiKey": "k", "model": "m"}
    run(providers.chat(conn, HISTORY, max_tokens=50, client=client))
    assert body(seen[0])["max_tokens"] == 50


def test_retry_without_temperature():
    def handler(request):
        if "temperature" in body(request):
            return 400, {"error": {"message": "Unsupported value: 'temperature'"}}
        return 200, {"choices": [{"message": {"content": "ok"}}]}

    client, seen = mock_client(handler)
    conn = {"provider": "openai", "apiKey": "k", "model": "o-model"}
    assert run(providers.chat(conn, HISTORY, temperature=0.7, client=client)) == "ok"
    assert len(seen) == 2


def test_anthropic_chat():
    reply = {"content": [{"type": "text", "text": "It is noon."}]}
    client, seen = mock_client(lambda r: (200, reply))
    conn = {"provider": "anthropic", "apiKey": "k", "model": "claude-x"}
    assert run(providers.chat(conn, HISTORY, system="sys", max_tokens=99, client=client)) == "It is noon."
    sent = body(seen[0])
    assert seen[0].url.path == "/v1/messages"
    assert sent["system"] == "sys" and sent["max_tokens"] == 99
    assert sent["messages"] == HISTORY


def test_gemini_chat():
    reply = {"candidates": [{"content": {"parts": [{"text": "thinking", "thought": True}, {"text": "Noon"}]}}]}
    client, seen = mock_client(lambda r: (200, reply))
    conn = {"provider": "gemini", "apiKey": "k", "model": "gemini-pro"}
    assert run(providers.chat(conn, HISTORY, system="sys", client=client)) == "Noon"
    sent = body(seen[0])
    assert seen[0].url.path == "/v1beta/models/gemini-pro:generateContent"
    assert [c["role"] for c in sent["contents"]] == ["user", "model", "user"]
    assert sent["systemInstruction"]["parts"][0]["text"] == "sys"


def test_ollama_chat():
    client, seen = mock_client(lambda r: (200, {"message": {"content": "Noon"}}))
    conn = {"provider": "ollama", "model": "llama3"}
    assert run(providers.chat(conn, HISTORY, system="sys", max_tokens=64, client=client)) == "Noon"
    sent = body(seen[0])
    assert sent["stream"] is False and sent["options"]["num_predict"] == 64
    assert sent["messages"][0]["role"] == "system"


def test_empty_answer_is_an_error():
    client, _ = mock_client(lambda r: (200, {"choices": [{"message": {"content": ""}}]}))
    with pytest.raises(providers.ProviderError, match="empty"):
        run(providers.chat({"provider": "groq", "apiKey": "k", "model": "m"}, HISTORY, client=client))


def test_every_provider_has_a_known_api():
    assert {p.api for p in providers.PROVIDERS} <= {"openai", "anthropic", "gemini", "ollama"}
    assert len({p.id for p in providers.PROVIDERS}) == len(providers.PROVIDERS)


# ---- Settings store --------------------------------------------------------------


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    monkeypatch.setattr(store, "SETTINGS_FILE", path)
    return path


def test_keys_are_masked_and_kept(settings_file):
    conn = {"name": "OR", "provider": "openrouter", "apiKey": "sk-or-1234567890abcd", "model": "m"}
    public = store.update({"connections": [conn]})
    saved = public["connections"][0]
    assert "apiKey" not in saved and saved["hasKey"] and saved["keyHint"].endswith("abcd")
    assert "sk-or-1234567890abcd" not in json.dumps(store.public())

    # Saving again from the browser (no key) keeps the stored key
    store.update({"connections": [{**saved, "model": "m2"}], "activeConnection": saved["id"]})
    stored = store.load()["connections"][0]
    assert stored["apiKey"] == "sk-or-1234567890abcd" and stored["model"] == "m2"
    assert store.active_connection()["id"] == saved["id"]


def test_fill_key_uses_stored_key(settings_file):
    saved = store.update({"connections": [{"provider": "openai", "apiKey": "sk-secret-key-123"}]})
    draft = {"id": saved["connections"][0]["id"], "provider": "openai", "apiKey": ""}
    assert store.fill_key(draft)["apiKey"] == "sk-secret-key-123"


def test_deleted_active_connection_is_cleared(settings_file):
    saved = store.update({"connections": [{"provider": "ollama", "model": "llama3"}]})
    store.update({"activeConnection": saved["connections"][0]["id"]})
    assert store.update({"connections": []})["activeConnection"] is None


def test_sections_ignore_unknown_keys(settings_file):
    store.update({"audio": {"threshold": 0.7, "bogus": 1}})
    audio = store.load()["audio"]
    assert audio["threshold"] == 0.7 and "bogus" not in audio


def test_out_of_tokens_is_explained():
    reply = {"choices": [{"message": {"content": "", "reasoning": "hmm..."}, "finish_reason": "length"}]}
    client, _ = mock_client(lambda r: (200, reply))
    conn = {"provider": "openrouter", "apiKey": "k", "model": "some/thinker"}
    with pytest.raises(providers.EmptyAnswer) as err:
        run(providers.chat(conn, [{"role": "user", "content": "hi"}], max_tokens=500, client=client))
    assert err.value.out_of_tokens and "500 tokens" in str(err.value)


def test_dropped_connections_are_retried(monkeypatch):
    monkeypatch.setattr(providers, "RETRY_PAUSES", [0, 0])
    calls = []

    def transport(request):
        calls.append(request)
        if len(calls) < 3:
            raise httpx.ConnectError("network is unreachable", request=request)
        return httpx.Response(200, json={"message": {"content": "OK"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    conn = {"provider": "ollama-cloud", "apiKey": "k", "model": "m"}
    assert run(providers.chat(conn, [{"role": "user", "content": "hi"}], client=client)) == "OK"
    assert len(calls) == 3


def test_unreachable_after_retries(monkeypatch):
    monkeypatch.setattr(providers, "RETRY_PAUSES", [0])

    def transport(request):
        raise httpx.ConnectError("no route", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    with pytest.raises(providers.ProviderError, match="tried 2 times"):
        run(providers.chat({"provider": "ollama-cloud", "apiKey": "k", "model": "m"},
                           [{"role": "user", "content": "hi"}], client=client))


def test_ollama_think_switch_and_fallback():
    bodies = []

    def handler(request):
        bodies.append(body(request))
        if "think" in bodies[-1]:
            return 400, {"error": "model does not support thinking"}
        return 200, {"message": {"content": "OK"}}

    client, _ = mock_client(handler)
    conn = {"provider": "ollama-cloud", "apiKey": "k", "model": "m"}
    assert run(providers.chat(conn, [{"role": "user", "content": "hi"}], client=client, think=False)) == "OK"
    assert bodies[0]["think"] is False and "think" not in bodies[1]
