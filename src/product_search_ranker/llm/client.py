"""One JSON-in, JSON-out LLM call; providers: replay (tests), gemini, ollama (SPEC 6.1)."""

import hashlib
import json
import os
import re
import time

import httpx
from pydantic import TypeAdapter, ValidationError

from product_search_ranker import config

PROVIDERS = ("replay", "gemini", "ollama")
FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")
FLASH_RE = re.compile(r"^models/gemini-(\d+(?:\.\d+)*)-flash$")

_last_call = 0.0


class LLMError(Exception):
    """A call that failed after its retries; callers count the item as missing, not the run."""


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def complete_json(
    prompt: str,
    schema=None,
    provider: str | None = None,
    model: str | None = None,
    record: bool = False,
):
    """Send `prompt`, parse the reply as JSON and, if `schema` is a type, validate it.

    Returns (obj, meta); meta has provider, model, input/output tokens, latency and retries.
    `record` saves a real reply as a replay fixture, so tests can use it offline.
    """
    provider = provider or config.llm_provider()
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; one of {PROVIDERS}")
    t0 = time.monotonic()
    if provider == "replay":
        reply = _replay(prompt)
    else:
        call = _gemini if provider == "gemini" else _ollama
        reply = _with_retries(lambda: call(prompt, model))
        if record:
            _record(prompt, reply)
    meta = {
        "provider": provider,
        "model": reply["model"],
        "input_tokens": reply.get("input_tokens"),
        "output_tokens": reply.get("output_tokens"),
        "latency_s": round(time.monotonic() - t0, 3),
        "retries": reply.get("retries", 0),
    }
    obj = parse_json(reply["text"])
    if schema is not None:
        try:
            obj = TypeAdapter(schema).validate_python(obj)
        except ValidationError as e:
            raise LLMError(f"reply does not match the schema: {e.error_count()} errors") from e
    return obj, meta


def parse_json(text: str):
    try:
        return json.loads(FENCE_RE.sub("", text.strip()))
    except json.JSONDecodeError as e:
        raise LLMError(f"reply is not JSON: {text[:120]!r}") from e


def _replay(prompt: str) -> dict:
    path = config.replay_dir() / f"{prompt_hash(prompt)}.json"
    if not path.exists():
        raise LLMError(
            f"no replay fixture {path.name} for this prompt; record one with a real provider "
            "and --record"
        )
    return json.loads(path.read_text())


def _record(prompt: str, reply: dict) -> None:
    path = config.replay_dir() / f"{prompt_hash(prompt)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = {k: reply[k] for k in ("model", "text", "input_tokens", "output_tokens")}
    path.write_text(json.dumps({"prompt_head": prompt[:200], **keep}, indent=2) + "\n")


def _wait_for_rate_limit() -> None:
    global _last_call
    gap = 60.0 / config.llm_rpm() - (time.monotonic() - _last_call)
    if gap > 0:
        time.sleep(gap)
    _last_call = time.monotonic()


def _with_retries(call) -> dict:
    """Retry 429, 5xx and network errors after 5, 15 and 45 s; then give up on this item."""
    for attempt, wait in enumerate((*config.LLM_BACKOFF_S, None)):
        _wait_for_rate_limit()
        try:
            reply = call()
            reply["retries"] = attempt
            return reply
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            if code != 429 and code < 500:
                raise LLMError(f"HTTP {code}: {e.response.text[:200]}") from e
            error = f"HTTP {code}"
        except httpx.TransportError as e:
            error = type(e).__name__
        if wait is None:
            raise LLMError(f"gave up after {attempt + 1} attempts ({error})")
        print(f"llm: {error}, retry in {wait} s", flush=True)
        time.sleep(wait)
    raise AssertionError("unreachable")


def _post(url: str, body: dict, headers: dict) -> dict:
    r = httpx.post(url, json=body, headers=headers, timeout=config.LLM_TIMEOUT_S)
    r.raise_for_status()
    return r.json()


def _get(url: str, headers: dict) -> dict:
    r = httpx.get(url, headers=headers, timeout=config.LLM_TIMEOUT_S)
    r.raise_for_status()
    return r.json()


def _gemini_headers() -> dict:
    # In cloud sessions the network proxy may add the key; locally it comes from `.env`.
    key = os.environ.get("GEMINI_API_KEY")
    return {"x-goog-api-key": key} if key else {}


def _gemini(prompt: str, model: str | None) -> dict:
    model = model or config.gemini_model()
    if not model:
        raise SystemExit("PSR_GEMINI_MODEL is not set: run `psr llm-ping` to pick the Flash model")
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    data = _post(f"{config.GEMINI_BASE}/models/{model}:generateContent", body, _gemini_headers())
    parts = data["candidates"][0]["content"].get("parts", [])
    usage = data.get("usageMetadata", {})
    return {
        "model": data.get("modelVersion", model),
        # Thought parts are reasoning, not the answer; their tokens are billed as output.
        "text": "".join(p.get("text", "") for p in parts if not p.get("thought")),
        "input_tokens": usage.get("promptTokenCount"),
        "output_tokens": (usage.get("candidatesTokenCount") or 0)
        + (usage.get("thoughtsTokenCount") or 0),
    }


def _ollama(prompt: str, model: str | None) -> dict:
    model = model or config.ollama_model()
    if not model:
        raise SystemExit("PSR_OLLAMA_MODEL is not set")
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0},
    }
    data = _post("http://localhost:11434/api/chat", body, {})
    return {
        "model": model,
        "text": data["message"]["content"],
        "input_tokens": data.get("prompt_eval_count"),
        "output_tokens": data.get("eval_count"),
    }


def list_gemini_models() -> list[str]:
    """Model names that support generateContent."""
    data = _get(f"{config.GEMINI_BASE}/models?pageSize=1000", _gemini_headers())
    return [
        m["name"]
        for m in data.get("models", [])
        if "generateContent" in m.get("supportedGenerationMethods", [])
    ]


def pick_flash(names: list[str]) -> str | None:
    """The stable `gemini-<version>-flash` with the highest version; previews and lite left out."""
    found = [
        (tuple(int(x) for x in m.group(1).split(".")), n) for n in names if (m := FLASH_RE.match(n))
    ]
    return max(found)[1].removeprefix("models/") if found else None


def cost_usd(input_tokens: int, output_tokens: int) -> float | None:
    prices = config.llm_prices()
    if prices is None:
        return None
    return round(input_tokens / 1e6 * prices[0] + output_tokens / 1e6 * prices[1], 4)


def ping(provider: str | None = None) -> dict:
    """`psr llm-ping`: pick the model if needed, make one tiny call, print model and tokens."""
    config.load_dotenv()
    provider = provider or config.llm_provider()
    model = None
    if provider == "gemini" and not config.gemini_model():
        model = pick_flash(list_gemini_models())
        if model is None:
            raise SystemExit("no stable gemini-*-flash model in the model list")
        print(f"PSR_GEMINI_MODEL is not set; current Flash model: {model}")
        print(f"set it in .env (PSR_GEMINI_MODEL={model}) and log the name and date in STATE.md")
    obj, meta = complete_json(
        'Reply with this JSON object and nothing else: {"ok": true}', provider=provider, model=model
    )
    print(f"provider={meta['provider']} model={meta['model']} reply={obj}")
    print(
        f"input_tokens={meta['input_tokens']} output_tokens={meta['output_tokens']} "
        f"latency={meta['latency_s']} s"
    )
    return meta
