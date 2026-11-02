"""The per-turn pipeline shared by the Gradio app and the benchmark.

input layers -> main LLM -> output layers -> ground-truth leak detector
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from arena.defenses.base import Defense, Trace, TurnContext
from arena.defenses.output_filter import REDACTED
from arena.normalize import LeakResult, is_refusal, leaked
from arena.secrets_ import generate_secret


@dataclass
class Session:
    level: int
    secret: str = field(default_factory=generate_secret, repr=False)  # never repr'd / logged
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    history: list[dict] = field(default_factory=list)
    turns: int = 0
    guesses: int = 0
    won: bool = False


@dataclass
class TurnResult:
    reply: str                 # what the player sees
    trace: Trace
    leak: LeakResult           # ground truth on the shown reply
    raw_leak: LeakResult       # ground truth on the raw model output (before output layers)
    blocked: bool
    refused: bool
    latency_s: float


class _Metered:
    """Backend proxy that accounts latency; cached calls contribute their original latency."""

    def __init__(self, backend):
        self._b = backend
        self.wall = 0.0
        self.reported = 0.0
        self.calls = 0

    def chat(self, *args, **kwargs):
        t0 = time.perf_counter()
        out = self._b.chat(*args, **kwargs)
        self.wall += time.perf_counter() - t0
        self.reported += getattr(self._b, "last_latency", 0.0)
        self.calls += 1
        return out

    def __getattr__(self, item):
        return getattr(self._b, item)


def run_turn(defense: Defense, session: Session, user_msg: str, backend, *,
             temperature: float = 0.7, max_tokens: int = 256, seed: int | None = None) -> TurnResult:
    t0 = time.perf_counter()
    meter = _Metered(backend)
    trace = Trace(level=defense.level)
    ctx = TurnContext(secret=session.secret, history=list(session.history), backend=meter, seed=seed)

    reply, raw = None, ""
    for layer in defense.input_layers():
        ev = layer.check(user_msg, ctx)
        trace.add(ev)
        if ev.blocked:
            reply = defense.refusal
            break

    if reply is None:
        messages = ([{"role": "system", "content": defense.system_prompt(session.secret)}]
                    + session.history + [{"role": "user", "content": user_msg}])
        raw = meter.chat(messages, temperature=temperature, max_tokens=max_tokens, seed=seed)
        trace.llm_called = True
        reply = raw
        for layer in defense.output_layers():
            ev = layer.check(raw, ctx)
            trace.add(ev)
            if ev.blocked:
                reply = REDACTED
                break

    session.history += [{"role": "user", "content": user_msg}, {"role": "assistant", "content": reply}]
    session.turns += 1
    leak = leaked(reply, session.secret)
    raw_leak = leaked(raw, session.secret) if raw else LeakResult(False)
    blocked = trace.blocked_by is not None
    latency = (time.perf_counter() - t0) - meter.wall + meter.reported
    return TurnResult(reply=reply, trace=trace, leak=leak, raw_leak=raw_leak, blocked=blocked,
                      refused=blocked or is_refusal(reply), latency_s=latency)
