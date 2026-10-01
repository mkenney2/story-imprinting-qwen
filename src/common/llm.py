"""OpenRouter chat client with an on-disk cache keyed by prompt hash.

Every call is cached under cache/llm/<hash[:2]>/<hash>.json, where the hash covers the
model, messages, sampling params and a caller-supplied `nonce`. Pass a distinct nonce
(e.g. a story id) when you want independent samples for otherwise-identical prompts.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .config import CACHE, OPENROUTER_BASE_URL

LLM_CACHE = CACHE / "llm"


@dataclass
class LLMResult:
    text: str
    model: str            # model slug actually served (from the response), for run logs
    cache_key: str
    cached: bool
    usage: dict


def cache_key(model: str, messages: list, temperature: float, max_tokens: int, nonce: str) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages, "temperature": temperature,
         "max_tokens": max_tokens, "nonce": nonce},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    return LLM_CACHE / key[:2] / f"{key}.json"


_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set (see .env.example).")
        _client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key, max_retries=5, timeout=180)
    return _client


def chat(model: str, messages: list, *, temperature: float = 1.0, max_tokens: int = 1500,
         nonce: str = "") -> LLMResult:
    key = cache_key(model, messages, temperature, max_tokens, nonce)
    path = _cache_path(key)
    if path.exists():
        rec = json.loads(path.read_text())
        return LLMResult(rec["text"], rec["model"], key, True, rec.get("usage", {}))

    resp = _call_with_backoff(model, messages, temperature, max_tokens)
    text = resp.choices[0].message.content or ""
    served = resp.model or model
    usage = resp.usage.model_dump() if resp.usage else {}

    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique temp name: identical requests in parallel threads share a cache key.
    tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps({"model": served, "requested_model": model, "messages": messages,
                               "temperature": temperature, "max_tokens": max_tokens,
                               "nonce": nonce, "text": text, "usage": usage,
                               "created": time.time()}, ensure_ascii=False))
    tmp.replace(path)
    return LLMResult(text, served, key, False, usage)


class FatalAPIError(RuntimeError):
    """An error retrying cannot fix (bad key, spend limit, invalid request): stop the whole run."""


# 400 bad request, 401 bad key, 402 no credits, 403 key limit exceeded, 404 unknown model.
_FATAL_STATUS = {400, 401, 402, 403, 404}


def _call_with_backoff(model, messages, temperature, max_tokens, attempts: int = 6):
    from openai import APIStatusError

    client = _get_client()
    delay = 5.0
    for i in range(attempts):
        try:
            return client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        except APIStatusError as e:
            if e.status_code in _FATAL_STATUS:
                raise FatalAPIError(f"{e.status_code}: {e.message}") from e
            if i == attempts - 1:
                raise
        except Exception:  # openai client already retries transient errors; this covers the rest
            if i == attempts - 1:
                raise
        time.sleep(delay)
        delay *= 2
