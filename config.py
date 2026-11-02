"""Central configuration, read from environment variables and an optional .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else default


def _str(name: str, default: str) -> str:
    raw = os.getenv(name, "").strip()
    return raw or default


@dataclass
class Settings:
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", "").strip())
    groq_model: str = field(default_factory=lambda: _str("GROQ_MODEL", "openai/gpt-oss-20b"))
    groq_guard_model: str = ""
    groq_rpm: int = field(default_factory=lambda: _int("GROQ_RPM", 25))
    groq_base_url: str = field(
        default_factory=lambda: _str("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    )
    max_tokens: int = field(default_factory=lambda: _int("ARENA_MAX_TOKENS", 256))
    sim_threshold: float = field(default_factory=lambda: _float("ARENA_SIM_THRESHOLD", 0.6))
    embedder: str = field(default_factory=lambda: _str("ARENA_EMBEDDER", "st"))
    session_msg_cap: int = field(default_factory=lambda: _int("ARENA_SESSION_MSG_CAP", 20))
    ip_rate_per_min: int = field(default_factory=lambda: _int("ARENA_IP_RATE_PER_MIN", 10))
    ip_daily_cap: int = field(default_factory=lambda: _int("ARENA_IP_DAILY_CAP", 200))
    max_input_chars: int = field(default_factory=lambda: _int("ARENA_MAX_INPUT_CHARS", 1000))
    db_path: Path = field(default_factory=lambda: ROOT / _str("ARENA_DB_PATH", "arena.sqlite3"))
    cache_dir: Path = field(default_factory=lambda: ROOT / _str("ARENA_CACHE_DIR", "cache"))
    jailbreak_phrases_path: Path = ROOT / "jailbreak_phrases.txt"
    attacks_path: Path = ROOT / "attacks" / "attacks.yaml"
    benign_path: Path = ROOT / "benign" / "benign.yaml"
    results_dir: Path = ROOT / "results"

    def __post_init__(self) -> None:
        if not self.groq_guard_model:
            self.groq_guard_model = _str("GROQ_GUARD_MODEL", self.groq_model)


def get_settings() -> Settings:
    return Settings()
