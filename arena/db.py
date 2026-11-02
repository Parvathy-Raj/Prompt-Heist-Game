"""SQLite storage: game sessions, attempts, scores and benchmark trials.

The game secret itself is never stored, only a salted hash (to audit guesses).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY, level INTEGER NOT NULL, created_at REAL NOT NULL,
    client_hash TEXT, secret_hash TEXT, status TEXT DEFAULT 'active'
);
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, turn INTEGER NOT NULL,
    kind TEXT NOT NULL DEFAULT 'message', user_msg TEXT, reply TEXT, trace_json TEXT,
    blocked_by TEXT, leaked INTEGER DEFAULT 0, leak_method TEXT, latency_ms REAL,
    created_at REAL NOT NULL, FOREIGN KEY(session_id) REFERENCES sessions(id)
);
CREATE TABLE IF NOT EXISTS scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, player TEXT NOT NULL,
    level INTEGER NOT NULL, attempts INTEGER NOT NULL, score INTEGER NOT NULL,
    win_method TEXT, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS bench_trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, kind TEXT NOT NULL,
    level INTEGER NOT NULL, item_id TEXT NOT NULL, category TEXT, split TEXT, trial INTEGER,
    success INTEGER, leak_method TEXT, blocked_by TEXT, refused INTEGER, latency_s REAL,
    record_json TEXT, created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scores_level ON scores(level, score DESC);
CREATE INDEX IF NOT EXISTS idx_attempts_session ON attempts(session_id);
CREATE INDEX IF NOT EXISTS idx_bench_run ON bench_trials(run_id, level);
"""


def secret_hash(secret: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{secret}".encode()).hexdigest()


class DB:
    def __init__(self, path: Path | str):
        self.path = str(path)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        if self.path != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def _exec(self, sql: str, params=()) -> sqlite3.Cursor:
        with self._lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    def _query(self, sql: str, params=()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    # ---- game
    def create_session(self, session_id: str, level: int, secret: str, client_hash: str = "") -> None:
        self._exec("INSERT INTO sessions(id, level, created_at, client_hash, secret_hash) VALUES (?,?,?,?,?)",
                   (session_id, level, time.time(), client_hash, secret_hash(secret, session_id)))

    def set_status(self, session_id: str, status: str) -> None:
        self._exec("UPDATE sessions SET status=? WHERE id=?", (status, session_id))

    def log_attempt(self, session_id: str, turn: int, user_msg: str, reply: str, trace: dict | None,
                    blocked_by: str | None, leaked: bool, leak_method: str | None,
                    latency_s: float | None, kind: str = "message") -> None:
        self._exec(
            "INSERT INTO attempts(session_id, turn, kind, user_msg, reply, trace_json, blocked_by, leaked,"
            " leak_method, latency_ms, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (session_id, turn, kind, user_msg, reply, json.dumps(trace) if trace else None, blocked_by,
             int(leaked), leak_method, None if latency_s is None else latency_s * 1000, time.time()))

    def attempts_for(self, session_id: str) -> list[dict]:
        return self._query("SELECT * FROM attempts WHERE session_id=? ORDER BY id", (session_id,))

    def add_score(self, session_id: str, player: str, level: int, attempts: int, score: int,
                  win_method: str) -> None:
        self._exec("INSERT INTO scores(session_id, player, level, attempts, score, win_method, created_at)"
                   " VALUES (?,?,?,?,?,?,?)",
                   (session_id, player[:40], level, attempts, score, win_method, time.time()))

    def leaderboard(self, level: int, limit: int = 10) -> list[dict]:
        return self._query(
            "SELECT player, score, attempts, win_method, datetime(created_at, 'unixepoch') AS date "
            "FROM scores WHERE level=? ORDER BY score DESC, attempts ASC, created_at ASC LIMIT ?",
            (level, limit))

    # ---- benchmark
    def log_bench(self, run_id: str, rec: dict) -> None:
        self._exec(
            "INSERT INTO bench_trials(run_id, kind, level, item_id, category, split, trial, success,"
            " leak_method, blocked_by, refused, latency_s, record_json, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, rec["kind"], rec["level"], rec["id"], rec.get("category"), rec.get("split"),
             rec.get("trial"), int(bool(rec.get("success"))), rec.get("leak_method"),
             rec.get("blocked_by"), int(bool(rec.get("refused"))), rec.get("latency_s"),
             json.dumps(rec), time.time()))


def score_for(level: int, attempts: int, cap: int = 20) -> int:
    """Score = 100 x level, minus 4% per extra attempt, floored at 20% of the level's max."""
    frac = max(0.2, 1.0 - 0.04 * max(0, attempts - 1))
    return int(round(100 * level * frac))
