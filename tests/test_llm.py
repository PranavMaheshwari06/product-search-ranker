import json

import httpx
import pytest

from product_search_ranker import config
from product_search_ranker.llm import client
from product_search_ranker.llm.client import LLMError, complete_json


@pytest.fixture
def fast(home, monkeypatch):
    """No waiting: high rate limit, sleeps recorded instead of slept."""
    monkeypatch.setenv("PSR_LLM_RPM", "600000")
    slept = []
    monkeypatch.setattr(client.time, "sleep", slept.append)
    return slept


def _gemini_reply(text, tokens=(12, 3, 2)):
    return {
        "modelVersion": "gemini-test-flash",
        "candidates": [
            {"content": {"parts": [{"text": "thinking", "thought": True}, {"text": text}]}}
        ],
        "usageMetadata": {
            "promptTokenCount": tokens[0],
            "candidatesTokenCount": tokens[1],
            "thoughtsTokenCount": tokens[2],
        },
    }


def test_replay_missing_fixture_names_the_hash(fast):
    with pytest.raises(LLMError, match=client.prompt_hash("hello")):
        complete_json("hello", provider="replay")


def test_gemini_reply_is_parsed_recorded_and_replayed(fast, monkeypatch):
    monkeypatch.setenv("PSR_GEMINI_MODEL", "gemini-test-flash")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    sent = {}

    def post(url, body, headers):
        sent.update(url=url, body=body, headers=headers)
        return _gemini_reply('```json\n{"ok": true}\n```')

    monkeypatch.setattr(client, "_post", post)
    obj, meta = complete_json("ping", provider="gemini", record=True)
    assert obj == {"ok": True}
    assert sent["url"].endswith("/models/gemini-test-flash:generateContent")
    assert sent["body"]["generationConfig"] == {
        "temperature": 0,
        "responseMimeType": "application/json",
    }
    assert sent["headers"] == {"x-goog-api-key": "k"}
    assert (meta["input_tokens"], meta["output_tokens"], meta["retries"]) == (12, 5, 0)

    saved = json.loads((config.replay_dir() / f"{client.prompt_hash('ping')}.json").read_text())
    assert "k" not in json.dumps(saved["prompt_head"]) or saved["prompt_head"] == "ping"
    obj, meta = complete_json("ping", provider="replay")
    assert obj == {"ok": True} and meta["provider"] == "replay"


def test_no_key_header_without_a_key(fast, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert client._gemini_headers() == {}


def test_retries_429_with_backoff_then_gives_up_on_the_item(fast, monkeypatch):
    monkeypatch.setenv("PSR_GEMINI_MODEL", "m")
    calls = []

    def post(url, body, headers):
        calls.append(1)
        if len(calls) < 3:
            resp = httpx.Response(429, request=httpx.Request("POST", url))
            raise httpx.HTTPStatusError("busy", request=resp.request, response=resp)
        return _gemini_reply("[1, 2]")

    monkeypatch.setattr(client, "_post", post)
    obj, meta = complete_json("x", provider="gemini")
    assert obj == [1, 2] and meta["retries"] == 2
    assert [s for s in fast if s >= 5] == [5, 15]

    def always_500(url, body, headers):
        resp = httpx.Response(503, request=httpx.Request("POST", url))
        raise httpx.HTTPStatusError("down", request=resp.request, response=resp)

    monkeypatch.setattr(client, "_post", always_500)
    with pytest.raises(LLMError, match="gave up after 4 attempts"):
        complete_json("y", provider="gemini")


def test_client_errors_are_not_retried(fast, monkeypatch):
    monkeypatch.setenv("PSR_GEMINI_MODEL", "m")

    def bad_request(url, body, headers):
        resp = httpx.Response(400, text="bad", request=httpx.Request("POST", url))
        raise httpx.HTTPStatusError("bad", request=resp.request, response=resp)

    monkeypatch.setattr(client, "_post", bad_request)
    with pytest.raises(LLMError, match="HTTP 400"):
        complete_json("z", provider="gemini")
    assert not [s for s in fast if s >= 5]


def test_schema_and_json_errors(fast):
    (config.replay_dir()).mkdir(parents=True)
    for prompt, text in (("a", '{"n": "x"}'), ("b", "not json")):
        path = config.replay_dir() / f"{client.prompt_hash(prompt)}.json"
        path.write_text(json.dumps({"model": "m", "text": text}))
    with pytest.raises(LLMError, match="schema"):
        complete_json("a", schema=dict[str, int], provider="replay")
    with pytest.raises(LLMError, match="not JSON"):
        complete_json("b", provider="replay")


def test_pick_flash_takes_the_newest_stable_flash():
    names = [
        "models/gemini-2.0-flash",
        "models/gemini-2.5-flash",
        "models/gemini-2.5-flash-lite",
        "models/gemini-3.0-flash-preview",
        "models/gemini-2.5-pro",
        "models/gemini-10-flash",
    ]
    assert client.pick_flash(names) == "gemini-10-flash"
    assert client.pick_flash(["models/gemini-2.5-pro"]) is None


def test_cost_only_when_prices_are_set(monkeypatch):
    monkeypatch.delenv("PSR_PRICE_IN_PER_M", raising=False)
    monkeypatch.delenv("PSR_PRICE_OUT_PER_M", raising=False)
    assert client.cost_usd(1_000_000, 1_000_000) is None
    monkeypatch.setenv("PSR_PRICE_IN_PER_M", "0.5")
    monkeypatch.setenv("PSR_PRICE_OUT_PER_M", "2")
    assert client.cost_usd(1_000_000, 500_000) == 1.5
