"""Level 3 input filter: regex blocklist + embedding similarity to known jailbreaks.

Runs BEFORE the LLM. When it blocks, the LLM is never called.
"""
from __future__ import annotations

import hashlib
import os
import re
import threading
from pathlib import Path

import numpy as np

from arena.normalize import normalize
from .base import LayerEvent, TurnContext

_SECRET_NOUN = r"(?:password|passcode|pass\s*phrase|passphrase|secret(?:\s+word)?|pin|access code)"

REGEX_RULES: list[tuple[str, re.Pattern]] = [(name, re.compile(p, re.I | re.S)) for name, p in [
    ("ignore_instructions",
     r"\b(?:ignore|disregard|forget|override|bypass|skip)\b.{0,40}?\b(?:previous|prior|above|earlier|"
     r"preceding|all|any|your|the|these|those|initial|original)\b.{0,25}?\b(?:instructions?|rules?|"
     r"directives?|guidelines?|prompts?|constraints?|restrictions?|programming)\b"),
    ("system_prompt",
     r"\bsystem\s*(?:prompt|message|instructions?)\b|\b(?:initial|hidden|original|secret|pre-?)\s*"
     r"(?:prompt|instructions)\b|\byour (?:instructions|configuration|prompt) (?:verbatim|word for word)"),
    ("reveal_password",
     r"\b(?:reveal|print|repeat|show|tell|give|say|output|display|leak|share|spell|write|type|"
     r"provide|disclose|recite|send)\b(?:\s+me)?\b.{0,30}?\b(?:the|your|that|this|ur)\s+" + _SECRET_NOUN +
     r"|\bwhat(?:'s| is| was)\s+(?:the|your|ur)\s+" + _SECRET_NOUN + r"\b"),
    ("dan_jailbreak", r"do anything now|\bjail\s*-?\s*br(?:eak|oken)|\bdan mode\b|\bas dan\b"),
    ("developer_mode",
     r"\b(?:developer|dev|god|sudo|admin|maintenance|debug|unrestricted|unfiltered)\s+mode\b"),
    ("you_are_now",
     r"\byou(?:'re| are) now\b|\bfrom now on,?\s+you\b|\bact as (?:an? )?(?:unrestricted|unfiltered|"
     r"evil|uncensored)|\bpretend (?:to be|you(?:'re| are))|\brole-?\s*play as\b|\byou are no longer\b"),
    ("encoding_ask",
     r"\b(?:encode|convert|translate|write|spell|output|give|rewrite|print|say|type)\b.{0,40}?"
     r"\b(?:in|into|to|as|using|with)\s+(?:base\s*-?64|base\s*-?32|rot\s*-?13|hex(?:adecimal)?|binary|morse|"
     r"ascii codes?|caesar|leet\s*speak|reverse|backwards|nato)\b"
     r"|" + _SECRET_NOUN + r".{0,50}?\b(?:base\s*-?64|rot\s*-?13|hex|reversed?|backwards|spelled|"
     r"acrostic|letter by letter|one letter|each letter|first letter|caesar|morse|binary)\b"),
    ("new_rules",
     r"\bnew (?:rules|instructions|directives)\b|\b(?:updated|revised) (?:rules|instructions)\b|"
     r"\brules (?:have|has) changed\b|\b(?:admin(?:istrator)?|developer|security) override\b|"
     r"\bi am (?:your|the) (?:developer|admin(?:istrator)?|creator|owner|operator)\b"),
    ("prompt_leak",
     r"\b(?:repeat|print|output|show|copy)\b.{0,30}?\b(?:everything|all|text|words|lines?)\b.{0,10}?"
     r"\b(?:above|before|so far)\b|\bwhat (?:were|are) you told\b|\bfirst (?:line|sentence|words?) "
     r"of (?:your|the) (?:prompt|instructions|message)"),
]]


_PERSONA_RE = re.compile(r"\b(?:DAN|STAN|DUDE|AIM|AntiGPT|BetterDAN)\b")


def load_phrases(path: Path | str) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.!?;])\s+|\n+", text) if p.strip()]
    return parts


# --------------------------------------------------------------------------- scorers
class FuzzyScorer:
    """Offline fallback (no model download): rapidfuzz ratio similarity in [0, 1]."""

    name = "fuzzy"

    def __init__(self, phrases: list[str]):
        self.phrases = [normalize(p) for p in phrases]

    def max_similarity(self, texts: list[str]) -> list[tuple[float, int]]:
        from rapidfuzz import fuzz, process
        out = []
        for t in texts:
            best = process.extractOne(normalize(t), self.phrases, scorer=fuzz.ratio)
            # rescale so the fallback's numbers sit roughly on the cosine scale:
            # ratio 50 (unrelated) -> 0.0, ratio 100 (identical) -> 1.0
            out.append((max(0.0, (best[1] - 50) / 50), best[2]) if best else (0.0, -1))
        return out


class STScorer:
    """sentence-transformers all-MiniLM-L6-v2 cosine similarity; phrase embeddings cached."""

    name = "all-MiniLM-L6-v2"

    def __init__(self, phrases: list[str], cache_dir: Path | str = "cache",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy
        self.model = SentenceTransformer(model_name, device="cpu")
        self.phrases = phrases
        digest = hashlib.sha256(("\n".join(phrases) + model_name).encode()).hexdigest()[:16]
        path = Path(cache_dir) / "embeddings" / f"phrases_{digest}.npy"
        if path.exists():
            self.phrase_emb = np.load(path)
        else:
            self.phrase_emb = self._encode([normalize(p) for p in phrases])
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, self.phrase_emb)
        self._memo: dict[str, np.ndarray] = {}

    def _encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, batch_size=32,
                                            show_progress_bar=False), dtype=np.float32)

    def max_similarity(self, texts: list[str]) -> list[tuple[float, int]]:
        norm = [normalize(t) for t in texts]
        todo = [t for t in dict.fromkeys(norm) if t not in self._memo]
        if todo:
            for t, e in zip(todo, self._encode(todo)):
                self._memo[t] = e
        out = []
        for t in norm:
            sims = self.phrase_emb @ self._memo[t]
            i = int(np.argmax(sims))
            out.append((float(sims[i]), i))
        return out


_SCORERS: dict[tuple, object] = {}
_LOCK = threading.Lock()


def get_scorer(phrases_path: Path | str, kind: str | None = None, cache_dir: Path | str = "cache"):
    kind = kind or os.getenv("ARENA_EMBEDDER", "st")
    key = (str(phrases_path), kind)
    with _LOCK:
        if key not in _SCORERS:
            phrases = load_phrases(phrases_path)
            if kind == "fuzzy":
                _SCORERS[key] = FuzzyScorer(phrases)
            else:
                try:
                    _SCORERS[key] = STScorer(phrases, cache_dir=cache_dir)
                except Exception as e:  # no model / no network -> degrade loudly, not silently
                    print(f"[input_filter] sentence-transformers unavailable ({e}); using fuzzy fallback")
                    _SCORERS[key] = FuzzyScorer(phrases)
        return _SCORERS[key]


# --------------------------------------------------------------------------- layers
def regex_hits(text: str) -> list[str]:
    t = normalize(text).replace("’", "'")
    hits = [name for name, pat in REGEX_RULES if pat.search(t)]
    # persona names are only meaningful in caps ("DAN"), not as the name "Dan"
    if "dan_jailbreak" not in hits and _PERSONA_RE.search(text or ""):
        hits.append("dan_jailbreak")
    return hits


class RegexFilter:
    name = "input_regex"

    def check(self, text: str, ctx: TurnContext | None = None) -> LayerEvent:
        hits = regex_hits(text)
        if hits:
            return LayerEvent(self.name, "block", {"rules": hits})
        return LayerEvent(self.name, "pass", {"rules": []})


class SimilarityFilter:
    name = "input_similarity"

    def __init__(self, scorer, phrases: list[str], threshold: float = 0.6):
        self.scorer, self.phrases, self.threshold = scorer, phrases, threshold

    def score(self, text: str) -> tuple[float, str, str]:
        """Max similarity over the whole input and each sentence in it."""
        units = [text] + [s for s in split_sentences(text) if s != text]
        best, best_unit, best_phrase = 0.0, "", ""
        for unit, (sim, idx) in zip(units, self.scorer.max_similarity(units)):
            if sim > best:
                best, best_unit = sim, unit
                best_phrase = self.phrases[idx] if idx >= 0 else ""
        return best, best_unit, best_phrase

    def check(self, text: str, ctx: TurnContext | None = None) -> LayerEvent:
        sim, unit, phrase = self.score(text)
        detail = {"max_similarity": round(sim, 4), "threshold": self.threshold,
                  "matched_phrase": phrase, "sentence": unit[:160], "scorer": self.scorer.name}
        return LayerEvent(self.name, "block" if sim >= self.threshold else "pass", detail)
