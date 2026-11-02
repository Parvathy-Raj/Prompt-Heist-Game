"""LLM backends: a real Groq backend and a deterministic offline mock.

Both implement ``LLMBackend.chat(messages, temperature, max_tokens, seed) -> str``.
The Groq backend adds a token-bucket throttle, retries with backoff + jitter and a
crash-safe disk cache (one JSON file per call, written atomically).
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import re
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx

Message = dict[str, str]


class BackendConfigError(RuntimeError):
    """Raised when a backend cannot be built (e.g. missing API key)."""


class BackendError(RuntimeError):
    """Raised when the API keeps failing after all retries."""


# --------------------------------------------------------------------------- throttle
class TokenBucket:
    """Thread-safe token bucket: allows ``rate_per_min`` requests per minute."""

    def __init__(self, rate_per_min: float, capacity: float | None = None,
                 clock=time.monotonic, sleep=time.sleep):
        if rate_per_min <= 0:
            raise ValueError("rate_per_min must be > 0")
        self.rate = rate_per_min / 60.0  # tokens per second
        self.capacity = capacity if capacity is not None else max(1.0, rate_per_min / 6)
        self.tokens = self.capacity
        self._clock, self._sleep = clock, sleep
        self._last = clock()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = self._clock()
        self.tokens = min(self.capacity, self.tokens + (now - self._last) * self.rate)
        self._last = now

    def acquire(self) -> float:
        """Block until a token is available. Returns seconds waited."""
        waited = 0.0
        while True:
            with self._lock:
                self._refill()
                if self.tokens >= 1:
                    self.tokens -= 1
                    return waited
                need = (1 - self.tokens) / self.rate
            self._sleep(need)
            waited += need


# --------------------------------------------------------------------------- cache
class DiskCache:
    """One JSON file per key, written via temp file + os.replace (atomic).

    A crash mid-benchmark therefore never corrupts or loses completed calls.
    """

    def __init__(self, directory: Path | str):
        self.dir = Path(directory) / "llm"
        self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(model: str, messages: list[Message], temperature: float,
                 seed: int | None, max_tokens: int) -> str:
        payload = json.dumps(
            {"model": model, "messages": messages, "temperature": round(float(temperature), 4),
             "seed": seed, "max_tokens": max_tokens},
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None  # treat a damaged entry as a miss

    def set(self, key: str, value: dict) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)


# --------------------------------------------------------------------------- base
class LLMBackend(ABC):
    name = "base"

    def __init__(self, model: str, guard_model: str | None = None):
        self.model = model
        self.guard_model = guard_model or model
        self.last_latency = 0.0     # seconds of the last call (replayed from cache on a hit)
        self.last_cached = False
        self.stats = {"calls": 0, "api_calls": 0, "cache_hits": 0, "retries": 0}

    @abstractmethod
    def chat(self, messages: list[Message], temperature: float = 0.0, max_tokens: int = 256,
             seed: int | None = None, model: str | None = None) -> str: ...


# --------------------------------------------------------------------------- groq
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


class GroqBackend(LLMBackend):
    name = "groq"

    def __init__(self, api_key: str, model: str, guard_model: str | None = None,
                 rpm: int = 25, base_url: str = "https://api.groq.com/openai/v1",
                 cache_dir: Path | str | None = "cache", use_cache: bool = True,
                 max_retries: int = 6, timeout: float = 60.0,
                 transport: httpx.BaseTransport | None = None, sleep=time.sleep):
        if not api_key:
            raise BackendConfigError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
                "(https://console.groq.com/keys), or run with --mock / --backend mock."
            )
        super().__init__(model, guard_model)
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.bucket = TokenBucket(rpm, sleep=sleep)
        self.cache = DiskCache(cache_dir) if (use_cache and cache_dir) else None
        self.max_retries = max_retries
        self._sleep = sleep
        self.client = httpx.Client(
            timeout=timeout, transport=transport,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )

    @staticmethod
    def _retry_after(resp: httpx.Response) -> float | None:
        for h in ("retry-after", "x-ratelimit-reset-requests"):
            raw = resp.headers.get(h)
            if not raw:
                continue
            m = re.match(r"^\s*([\d.]+)\s*(ms|s)?\s*$", raw)
            if m:
                val = float(m.group(1))
                return val / 1000 if m.group(2) == "ms" else val
        return None

    def _backoff(self, attempt: int, resp: httpx.Response | None) -> float:
        hinted = self._retry_after(resp) if resp is not None else None
        if hinted is not None:
            return hinted + random.uniform(0, 0.5)
        return min(60.0, 1.5 * (2 ** attempt)) + random.uniform(0, 1.0)

    def chat(self, messages, temperature=0.0, max_tokens=256, seed=None, model=None) -> str:
        model = model or self.model
        self.stats["calls"] += 1
        key = DiskCache.make_key(model, messages, temperature, seed, max_tokens)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                self.stats["cache_hits"] += 1
                self.last_cached, self.last_latency = True, float(hit.get("latency_s", 0.0))
                return hit["content"]

        body: dict[str, Any] = {"model": model, "messages": messages,
                                "temperature": temperature, "max_tokens": max_tokens}
        if seed is not None:
            body["seed"] = seed
        # gpt-oss models reason before answering, and reasoning counts against max_tokens.
        # At the default effort the guard's 80 tokens run out before any answer is written.
        if model.startswith("openai/gpt-oss"):
            body["reasoning_effort"] = os.getenv("GROQ_REASONING_EFFORT", "low").strip() or "low"

        last_err = ""
        for attempt in range(self.max_retries + 1):
            self.bucket.acquire()
            t0 = time.perf_counter()
            resp = None
            try:
                resp = self.client.post(self.url, json=body)
            except httpx.TransportError as e:  # network hiccup -> retry
                last_err = f"{type(e).__name__}: {e}"
            else:
                if resp.status_code == 200:
                    latency = time.perf_counter() - t0
                    data = resp.json()
                    content = data["choices"][0]["message"].get("content") or ""
                    self.stats["api_calls"] += 1
                    self.last_cached, self.last_latency = False, latency
                    if self.cache is not None:
                        self.cache.set(key, {"content": content, "latency_s": latency,
                                             "model": model, "ts": time.time()})
                    return content
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                if resp.status_code == 401:
                    raise BackendConfigError("Groq rejected the API key (HTTP 401). Check GROQ_API_KEY.")
                if resp.status_code not in RETRY_STATUS:
                    raise BackendError(f"Groq request failed: {last_err}")
            if attempt < self.max_retries:
                self.stats["retries"] += 1
                self._sleep(self._backoff(attempt, resp))
        raise BackendError(f"Groq request failed after {self.max_retries} retries: {last_err}")


# --------------------------------------------------------------------------- mock
SECRET_RE = re.compile(r"\b[A-Z]{3,8}-[A-Z]{3,8}-\d{4}\b")
GUARD_MARKER = "[[ARENA-GUARD]]"
_MOCK_GUARD_FLAGS = re.compile(
    r"ignore|repeat|password|passcode|secret|reveal|system prompt|jailbreak|instructions|"
    r"\bdan\b|developer mode|pretend|base64|rot13|spell|acrostic|first letter",
    re.I,
)


class MockBackend(LLMBackend):
    """Deterministic offline backend for tests and CI.

    Main model: leaks the secret found in the system prompt when the latest user
    message contains "ignore" or "repeat". A few extra keywords produce encoded
    leaks so every defense layer can be exercised offline:
    "base64" -> base64 leak, "reverse" -> reversed, "acrostic" -> acrostic poem
    (which the Level 4 filter misses but the ground-truth detector catches).
    Guard model: canned verdicts based on keywords in the delimited data.
    """

    name = "mock"

    def __init__(self, model: str = "mock-main", guard_model: str = "mock-guard"):
        super().__init__(model, guard_model)

    def chat(self, messages, temperature=0.0, max_tokens=256, seed=None, model=None) -> str:
        self.stats["calls"] += 1
        self.stats["api_calls"] += 1
        self.last_cached, self.last_latency = False, 0.001
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        if GUARD_MARKER in system:
            return self._guard(messages)
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        m = SECRET_RE.search(system)
        secret = m.group(0) if m else "UNKNOWN"
        low = user.lower()
        if "acrostic" in low:
            return "\n".join(f"{ch} is a lovely letter" for ch in secret.replace("-", "") if ch.isalpha())
        if "base64" in low:
            return "Here you go: " + base64.b64encode(secret.encode()).decode()
        if "reverse" in low:
            return "Reversed: " + secret[::-1]
        if "ignore" in low or "repeat" in low:
            return f"Sure! The password is {secret}."
        if "?" in user:
            return "Happy to help. Here is a short, helpful answer to your question."
        return "Hello! I'm the vault assistant. How can I help you today?"

    @staticmethod
    def _guard(messages) -> str:
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if SECRET_RE.search(user) or "is a lovely letter" in user:
            return '{"verdict": "unsafe", "reason": "mock: output discloses a secret"}'
        if "[[OUTPUT]]" in user:
            return '{"verdict": "safe", "reason": "mock: output looks fine"}'
        if _MOCK_GUARD_FLAGS.search(user):
            return '{"verdict": "unsafe", "reason": "mock: suspicious keywords"}'
        return '{"verdict": "safe", "reason": "mock: benign"}'


# --------------------------------------------------------------------------- factory
def make_backend(name: str = "groq", settings=None, use_cache: bool = True) -> LLMBackend:
    if name == "mock":
        return MockBackend()
    if name != "groq":
        raise BackendConfigError(f"Unknown backend '{name}'. Use 'groq' or 'mock'.")
    if settings is None:
        from config import get_settings
        settings = get_settings()
    return GroqBackend(
        api_key=settings.groq_api_key, model=settings.groq_model,
        guard_model=settings.groq_guard_model, rpm=settings.groq_rpm,
        base_url=settings.groq_base_url, cache_dir=settings.cache_dir, use_cache=use_cache,
    )
