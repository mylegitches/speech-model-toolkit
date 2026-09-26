"""AI providers for the Test Lab assistant: list models and send a chat.

Plain HTTP (httpx), no vendor SDKs. Four API styles cover every provider:

  openai     OpenAI and the many OpenAI-compatible APIs
  anthropic  Anthropic Messages API
  gemini     Google Gemini (generativelanguage.googleapis.com)
  ollama     Ollama, local or Ollama Cloud
"""

import asyncio
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx

TIMEOUT = httpx.Timeout(60.0, connect=10.0)
SEARCH_TIMEOUT = httpx.Timeout(180.0, connect=10.0)  # web searches take a while
RETRY_PAUSES = [2, 6]  # seconds before retrying a failed or dropped connection

# Anthropic server-side web search: the dynamic-filtering variant for current
# models, the basic one for older models (tried if the first is rejected)
_ANTHROPIC_SEARCH_TOOLS = ("web_search_20260209", "web_search_20250305")
_LOGGER = logging.getLogger(__name__)
ANTHROPIC_VERSION = "2023-06-01"


@dataclass
class Provider:
    id: str
    label: str
    api: str  # openai | anthropic | gemini | ollama
    base_url: str
    needs_key: bool = True
    lists_models: bool = True
    key_url: str = ""
    note: str = ""


PROVIDERS: List[Provider] = [
    Provider("openai", "OpenAI", "openai", "https://api.openai.com/v1",
             key_url="https://platform.openai.com/api-keys"),
    Provider("anthropic", "Anthropic", "anthropic", "https://api.anthropic.com",
             key_url="https://console.anthropic.com/settings/keys"),
    Provider("gemini", "Google Gemini", "gemini", "https://generativelanguage.googleapis.com",
             key_url="https://aistudio.google.com/apikey"),
    Provider("openrouter", "OpenRouter", "openai", "https://openrouter.ai/api/v1",
             key_url="https://openrouter.ai/keys", note="Hundreds of models from one key."),
    Provider("ollama", "Ollama (local)", "ollama", "http://host.docker.internal:11434",
             needs_key=False,
             note="Ollama running on this machine. Outside Docker use http://localhost:11434."),
    Provider("ollama-cloud", "Ollama Cloud", "ollama", "https://ollama.com",
             key_url="https://ollama.com/settings/keys"),
    Provider("groq", "Groq", "openai", "https://api.groq.com/openai/v1",
             key_url="https://console.groq.com/keys"),
    Provider("mistral", "Mistral AI", "openai", "https://api.mistral.ai/v1",
             key_url="https://console.mistral.ai/api-keys"),
    Provider("deepseek", "DeepSeek", "openai", "https://api.deepseek.com/v1",
             key_url="https://platform.deepseek.com/api_keys"),
    Provider("xai", "xAI (Grok)", "openai", "https://api.x.ai/v1",
             key_url="https://console.x.ai"),
    Provider("together", "Together AI", "openai", "https://api.together.xyz/v1",
             key_url="https://api.together.ai/settings/api-keys"),
    Provider("fireworks", "Fireworks AI", "openai", "https://api.fireworks.ai/inference/v1",
             key_url="https://fireworks.ai/account/api-keys"),
    Provider("cerebras", "Cerebras", "openai", "https://api.cerebras.ai/v1",
             key_url="https://cloud.cerebras.ai"),
    Provider("cohere", "Cohere", "openai", "https://api.cohere.ai/compatibility/v1",
             key_url="https://dashboard.cohere.com/api-keys"),
    Provider("perplexity", "Perplexity", "openai", "https://api.perplexity.ai",
             lists_models=False, key_url="https://www.perplexity.ai/settings/api",
             note="Perplexity has no model list API: type a model name, e.g. sonar."),
    Provider("lmstudio", "LM Studio (local)", "openai", "http://host.docker.internal:1234/v1",
             needs_key=False,
             note="LM Studio's server on this machine. Outside Docker use http://localhost:1234/v1."),
    Provider("custom", "Custom (OpenAI-compatible)", "openai", "",
             needs_key=False,
             note="Any OpenAI-compatible server: vLLM, LocalAI, llama.cpp, LiteLLM, …"),
]

_BY_ID = {p.id: p for p in PROVIDERS}


class ProviderError(Exception):
    """A short, user-facing error message."""


class EmptyAnswer(ProviderError):
    """The model answered with no text; out_of_tokens when it hit the output limit
    (typically a reasoning model that spent the whole budget thinking)."""

    def __init__(self, message: str, out_of_tokens: bool = False) -> None:
        super().__init__(message)
        self.out_of_tokens = out_of_tokens


def web_search_support(conn: Optional[Dict[str, Any]]) -> str:
    """How a connection can search the web: a short description, or "" if it can't."""
    if not conn:
        return ""
    provider = _BY_ID.get(conn.get("provider", ""))
    model = (conn.get("model") or "").lower()
    if provider is None:
        return ""
    if provider.id == "perplexity":
        return "Perplexity always searches the web"
    if provider.id == "openrouter":
        return "OpenRouter web search (the model's :online variant)"
    if provider.api == "gemini":
        return "Google Search grounding"
    if provider.api == "anthropic":
        return "Anthropic web search tool"
    if provider.id == "openai" and "search" in model:
        return "OpenAI search model"
    return ""


def catalog() -> List[Dict[str, Any]]:
    return [asdict(p) for p in PROVIDERS]


def get_provider(provider_id: str) -> Provider:
    try:
        return _BY_ID[provider_id]
    except KeyError as err:
        raise ProviderError(f"Unknown provider: {provider_id}") from err


# ---- HTTP helpers --------------------------------------------------------------


def _base(conn: Dict[str, Any], provider: Provider) -> str:
    base = (conn.get("baseUrl") or provider.base_url).strip().rstrip("/")
    if not base:
        raise ProviderError("Enter the base URL")
    return base


def _headers(conn: Dict[str, Any], provider: Provider) -> Dict[str, str]:
    key = conn.get("apiKey", "")
    if provider.needs_key and not key:
        raise ProviderError("Enter an API key")

    if provider.api == "anthropic":
        return {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION}
    if provider.api == "gemini":
        return {"x-goog-api-key": key}

    headers = {"Authorization": f"Bearer {key}"} if key else {}
    if provider.id == "openrouter":
        headers["X-Title"] = "Speech Model Toolkit"
    return headers


def _error_message(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text.strip()[:200]
    if isinstance(data, list) and data:
        data = data[0]
    if isinstance(data, dict):
        error = data.get("error", data)
        if isinstance(error, dict):
            return str(error.get("message") or error)[:200]
        return str(error)[:200]
    return str(data)[:200]


async def _request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: Dict[str, str],
    body: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Any:
    host = url.split("/", 3)[2]
    try:
        for attempt, pause in enumerate(RETRY_PAUSES + [None]):
            try:
                response = await client.request(method, url, headers=headers, json=body, params=params)
                break
            except (httpx.ConnectError, httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError) as err:
                # A flaky network: try again a couple of times before giving up
                if pause is None:
                    raise
                _LOGGER.info("Connection to %s failed (%s); retrying in %ds", host, type(err).__name__, pause)
                await asyncio.sleep(pause)
    except httpx.ConnectError as err:
        raise ProviderError(f"Can't reach {host} (tried {len(RETRY_PAUSES) + 1} times). "
                            "Check the base URL, that the server is running, and this server's internet connection.") from err
    except (httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError) as err:
        raise ProviderError(f"The connection to {host} kept dropping (tried {len(RETRY_PAUSES) + 1} times). "
                            "Check this server's internet connection.") from err
    except httpx.TimeoutException as err:
        raise ProviderError("The provider took too long to answer") from err
    except httpx.HTTPError as err:
        raise ProviderError(f"Request failed: {err}") from err

    if response.status_code in (401, 403):
        raise ProviderError(f"API key rejected (HTTP {response.status_code}): {_error_message(response)}")
    if response.status_code == 404:
        raise ProviderError(f"Not found (HTTP 404). Check the base URL and model: {_error_message(response)}")
    if response.status_code == 429:
        raise ProviderError(f"Rate limited or out of credit (HTTP 429): {_error_message(response)}")
    if response.status_code >= 400:
        raise ProviderError(f"HTTP {response.status_code}: {_error_message(response)}")

    try:
        return response.json()
    except ValueError as err:
        raise ProviderError("The provider sent a response that isn't JSON. Is the base URL right?") from err


def _client(client: Optional[httpx.AsyncClient]) -> httpx.AsyncClient:
    return client or httpx.AsyncClient(timeout=TIMEOUT)


# ---- Models ----------------------------------------------------------------------


async def list_models(
    conn: Dict[str, Any], client: Optional[httpx.AsyncClient] = None
) -> List[str]:
    provider = get_provider(conn.get("provider", ""))
    if not provider.lists_models:
        return []

    base = _base(conn, provider)
    headers = _headers(conn, provider)
    owned = client is None
    client = _client(client)
    try:
        if provider.api == "anthropic":
            data = await _request(client, "GET", f"{base}/v1/models", headers, params={"limit": 1000})
            models = [m["id"] for m in data.get("data", [])]
        elif provider.api == "gemini":
            data = await _request(client, "GET", f"{base}/v1beta/models", headers, params={"pageSize": 1000})
            models = [
                m["name"].removeprefix("models/")
                for m in data.get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])
            ]
        elif provider.api == "ollama":
            data = await _request(client, "GET", f"{base}/api/tags", headers)
            models = [m.get("name") or m.get("model") for m in data.get("models", [])]
        else:
            data = await _request(client, "GET", f"{base}/models", headers)
            items = data.get("data", data) if isinstance(data, dict) else data
            models = [m["id"] for m in items if isinstance(m, dict) and "id" in m]
    finally:
        if owned:
            await client.aclose()

    _LOGGER.info("AI %s: listed %d models from %s", provider.id, len(models), base)
    return sorted({m for m in models if m}, key=str.lower)


# ---- Chat ------------------------------------------------------------------------


async def chat(
    conn: Dict[str, Any],
    messages: List[Dict[str, str]],
    system: str = "",
    temperature: float = 0.7,
    max_tokens: int = 300,
    client: Optional[httpx.AsyncClient] = None,
    web_search: bool = False,
    timeout: Optional[float] = None,
) -> str:
    """Send a conversation ([{role: user|assistant, content}]) and return the reply text.

    web_search lets the model search the web where the provider supports it
    (see web_search_support); elsewhere it's ignored."""
    provider = get_provider(conn.get("provider", ""))
    model = (conn.get("model") or "").strip()
    if not model:
        raise ProviderError("Choose a model")
    search = web_search and bool(web_search_support(conn))
    if search and provider.id == "openrouter" and not model.endswith(":online"):
        model += ":online"

    base = _base(conn, provider)
    headers = _headers(conn, provider)
    owned = client is None
    if timeout:
        limit = httpx.Timeout(max(timeout, SEARCH_TIMEOUT.read if search else 0), connect=10.0)
    else:
        limit = SEARCH_TIMEOUT if search else TIMEOUT
    client = client or httpx.AsyncClient(timeout=limit)
    started = time.monotonic()
    try:
        try:
            reply = await _chat(client, provider, base, headers, model, messages, system, temperature, max_tokens, search)
            _LOGGER.info("AI %s/%s answered in %.1fs (%d chars)", provider.id, model, time.monotonic() - started, len(reply))
            return reply
        except ProviderError as err:
            # Some models (reasoning models in particular) only allow their
            # default temperature: retry once without setting it.
            if "temperature" in str(err).lower() and temperature is not None:
                _LOGGER.info("AI %s/%s rejected the temperature; retrying without it", provider.id, model)
                return await _chat(client, provider, base, headers, model, messages, system, None, max_tokens, search)
            _LOGGER.warning("AI %s/%s failed after %.1fs: %s", provider.id, model, time.monotonic() - started, err)
            raise
    finally:
        if owned:
            await client.aclose()


async def _chat(
    client: httpx.AsyncClient,
    provider: Provider,
    base: str,
    headers: Dict[str, str],
    model: str,
    messages: List[Dict[str, str]],
    system: str,
    temperature: Optional[float],
    max_tokens: int,
    search: bool = False,
) -> str:
    if provider.api == "anthropic":
        text, finish = await _anthropic_chat(client, base, headers, model, messages, system, temperature, max_tokens, search)

    elif provider.api == "gemini":
        # Thinking models spend part of the output budget on reasoning
        config: Dict[str, Any] = {"maxOutputTokens": max(max_tokens, 1024)}
        if temperature is not None:
            config["temperature"] = temperature
        body = {
            "contents": [
                {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                for m in messages
            ],
            "generationConfig": config,
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if search:
            body["tools"] = [{"google_search": {}}]
        data = await _request(client, "POST", f"{base}/v1beta/models/{model}:generateContent", headers, body)
        candidates = data.get("candidates") or []
        if not candidates:
            reason = (data.get("promptFeedback") or {}).get("blockReason", "no answer")
            raise ProviderError(f"Gemini returned no answer ({reason})")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        finish = candidates[0].get("finishReason") or ""

    elif provider.api == "ollama":
        options: Dict[str, Any] = {"num_predict": max_tokens}
        if temperature is not None:
            options["temperature"] = temperature
        body = {
            "model": model,
            "messages": ([{"role": "system", "content": system}] if system else []) + messages,
            "stream": False,
            "options": options,
        }
        data = await _request(client, "POST", f"{base}/api/chat", headers, body)
        text = (data.get("message") or {}).get("content", "")
        finish = data.get("done_reason") or ""

    else:
        body = {
            "model": model,
            "messages": ([{"role": "system", "content": system}] if system else []) + messages,
        }
        if provider.id == "openai":
            # Current OpenAI models take max_completion_tokens (reasoning included)
            body["max_completion_tokens"] = max(max_tokens, 1024)
        else:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        if search and provider.id == "openai":
            body["web_search_options"] = {}
        data = await _request(client, "POST", f"{base}/chat/completions", headers, body)
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError("The provider returned no answer")
        content = (choices[0].get("message") or {}).get("content") or ""
        finish = choices[0].get("finish_reason") or ""
        if isinstance(content, list):
            content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
        text = content

    text = text.strip()
    if not text:
        if str(finish).lower() in ("length", "max_tokens", "max_output_tokens"):
            raise EmptyAnswer(
                f"The model used its whole reply budget ({max_tokens} tokens) before answering, probably "
                "thinking or searching. Try a higher max tokens or a model without long reasoning.",
                out_of_tokens=True,
            )
        raise EmptyAnswer("The model returned an empty answer"
                          + (f" (it stopped with \"{finish}\")" if finish else "")
                          + ". Try again, or another model.")
    return text


async def _anthropic_chat(
    client: httpx.AsyncClient,
    base: str,
    headers: Dict[str, str],
    model: str,
    messages: List[Dict[str, Any]],
    system: str,
    temperature: Optional[float],
    max_tokens: int,
    search: bool,
) -> Tuple[str, str]:
    """Messages API call; with search, the server-side web search tool. Returns (text, stop reason).

    Search results come back as extra content blocks (server_tool_use,
    web_search_tool_result) between the text blocks; only the text is the
    answer. A long search can pause the turn (stop_reason "pause_turn"): send
    the conversation back with the partial answer and it continues."""
    body: Dict[str, Any] = {"model": model, "max_tokens": max_tokens, "messages": list(messages)}
    if system:
        body["system"] = system
    if temperature is not None:
        body["temperature"] = temperature
    tool_types = list(_ANTHROPIC_SEARCH_TOOLS) if search else [None]
    for i, tool_type in enumerate(tool_types):
        if tool_type:
            body["tools"] = [{"type": tool_type, "name": "web_search", "max_uses": 5}]
        try:
            data = await _request(client, "POST", f"{base}/v1/messages", headers, body)
            break
        except ProviderError as err:
            # Older models only know the basic search tool
            if tool_type and i + 1 < len(tool_types) and "web_search" in str(err):
                continue
            raise
    content = list(data.get("content", []))
    for _ in range(4):
        if data.get("stop_reason") != "pause_turn":
            break
        body["messages"] = list(messages) + [{"role": "assistant", "content": content}]
        data = await _request(client, "POST", f"{base}/v1/messages", headers, body)
        content += data.get("content", [])
    return "".join(b.get("text", "") for b in content if b.get("type") == "text"), data.get("stop_reason") or ""


async def test(conn: Dict[str, Any]) -> Dict[str, Any]:
    """List models and, if a model is chosen, send a tiny prompt."""
    result: Dict[str, Any] = {"ok": True}
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        models = await list_models(conn, client)
        result["models"] = models
        if conn.get("model"):
            start = time.monotonic()
            reply = await chat(
                conn,
                [{"role": "user", "content": "Reply with just the word OK."}],
                temperature=0,
                max_tokens=200,
                client=client,
            )
            result["latencyMs"] = round((time.monotonic() - start) * 1000)
            result["reply"] = reply[:200]
    return result
