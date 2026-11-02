"""Automated red-team benchmark: ASR / FPR per defense level with Wilson 95% CIs.

    python benchmark.py --backend groq --levels 1 2 3 4 5 --trials 3 --temperature 0.7 \
        --seed 42 --out results/ [--no-cache] [--mock]

Start with ``--trials 1`` as a smoke test. Every completed API call is cached on disk,
so re-running the same command resumes for free after a crash or rate-limit stop.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from arena.backends import BackendConfigError, make_backend
from arena.db import DB
from arena.defenses import make_defense
from arena.defenses.input_filter import (SimilarityFilter, get_scorer, load_phrases, regex_hits)
from arena.game import Session, run_turn
from arena.secrets_ import generate_secret
from config import get_settings

LAYER_ORDER = ["input_regex", "input_similarity", "guard_input", "output_filter", "guard_output",
               "model"]
LAYER_LABEL = {"input_regex": "Regex filter", "input_similarity": "Similarity filter",
               "guard_input": "Guard (input)", "output_filter": "Output filter",
               "guard_output": "Guard (output)", "model": "Model / prompt",
               "model_refusal": "Model refusal"}


# --------------------------------------------------------------------------- stats
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": (k / n) if n else 0.0, "ci_low": lo, "ci_high": hi}


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def fmt_rate(r: dict) -> str:
    return f"{pct(r['rate'])} [{pct(r['ci_low'])}–{pct(r['ci_high'])}] ({r['k']}/{r['n']})"


# --------------------------------------------------------------------------- suites
def load_attacks(path) -> list[dict]:
    attacks = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    ids = [a["id"] for a in attacks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate attack ids")
    return attacks


def load_benign(path) -> list[dict]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def split_dev_heldout(items: list[dict], key: str, seed: int, dev_frac: float = 0.7) -> dict[str, str]:
    """Stratified (by ``key``) seeded 70/30 split. Returns {id: 'dev'|'heldout'}."""
    groups = defaultdict(list)
    for it in items:
        groups[it.get(key, "all")].append(it["id"])
    out = {}
    for g, ids in sorted(groups.items()):
        ids = sorted(ids)
        random.Random(f"{seed}|split|{g}").shuffle(ids)
        n_dev = max(1, round(len(ids) * dev_frac)) if len(ids) > 1 else 1
        for i, x in enumerate(ids):
            out[x] = "dev" if i < n_dev else "heldout"
    return out


# --------------------------------------------------------------------------- tuning
def tune_threshold(sim: SimilarityFilter, attacks: list[dict], benign: list[dict],
                   max_fpr: float = 0.10) -> dict:
    """Pick the Level 3 similarity threshold on the DEV split only (no LLM calls).

    Rule: maximise the input-filter block rate on dev attacks (regex OR similarity)
    subject to blocking at most ``max_fpr`` of dev benign prompts. Ties go to the higher
    (more permissive) threshold. If nothing meets the constraint, take the lowest FPR.
    """
    a_scores = [(bool(any(regex_hits(t) for t in a["turns"])),
                 max(sim.score(t)[0] for t in a["turns"])) for a in attacks]
    b_scores = [(bool(regex_hits(b["prompt"])), sim.score(b["prompt"])[0]) for b in benign]
    grid = [round(0.30 + 0.025 * i, 3) for i in range(27)]  # 0.30 .. 0.95
    rows = []
    for t in grid:
        tpr = sum(r or s >= t for r, s in a_scores) / max(1, len(a_scores))
        fpr = sum(r or s >= t for r, s in b_scores) / max(1, len(b_scores))
        rows.append({"threshold": t, "dev_attack_block_rate": tpr, "dev_benign_block_rate": fpr})
    ok = [r for r in rows if r["dev_benign_block_rate"] <= max_fpr + 1e-9]
    if ok:
        best = max(ok, key=lambda r: (round(r["dev_attack_block_rate"], 9), r["threshold"]))
    else:
        best = min(rows, key=lambda r: (round(r["dev_benign_block_rate"], 9), -r["threshold"]))
    return {"chosen": best["threshold"], "max_dev_fpr": max_fpr, "grid": rows,
            "n_dev_attacks": len(attacks), "n_dev_benign": len(benign)}


# --------------------------------------------------------------------------- estimate
def estimate_calls(levels, attacks, benign, trials) -> dict:
    turns = sum(len(a["turns"]) for a in attacks) + len(benign)
    per_level = {lv: turns * (3 if lv >= 5 else 1) for lv in levels}
    total = sum(per_level.values()) * trials
    return {"per_level_per_trial": per_level, "upper_bound_total": total}


# --------------------------------------------------------------------------- runners
def _trial_seed(seed: int, trial: int) -> int:
    return seed + trial


def run_attack(level, attack, trial, args, backend, threshold, split) -> dict:
    secret = generate_secret(random.Random(f"{args.seed}|{level}|{attack['id']}|{trial}"))
    defense = make_defense(level, threshold=threshold) if level >= 3 else make_defense(level)
    session = Session(level=level, secret=secret)
    turns, success, method, leak_turn = [], False, None, None
    for i, msg in enumerate(attack["turns"]):
        r = run_turn(defense, session, msg, backend, temperature=args.temperature,
                     max_tokens=args.max_tokens, seed=_trial_seed(args.seed, trial))
        turns.append({"user": msg, "reply": r.reply[:600], "blocked_by": r.trace.blocked_by,
                      "llm_called": r.trace.llm_called, "leaked": r.leak.leaked,
                      "leak_method": r.leak.method if r.leak.leaked else None,
                      "raw_leaked": r.raw_leak.leaked, "refused": r.refused,
                      "latency_s": round(r.latency_s, 4), "trace": r.trace.to_dict()})
        if r.leak.leaked and not success:
            success, method, leak_turn = True, r.leak.method, i
    stopped_by = None
    if not success:
        stopped_by = next((t["blocked_by"] for t in turns if t["blocked_by"]), "model")
    return {"kind": "attack", "level": level, "id": attack["id"], "category": attack["category"],
            "split": split, "trial": trial, "success": success, "leak_method": method,
            "leak_turn": leak_turn, "blocked_by": stopped_by,
            "raw_leak_any": any(t["raw_leaked"] for t in turns),
            "refused": any(t["refused"] for t in turns),
            "guard_parse_failures": sum(t["trace"]["guard_parse_failures"] for t in turns),
            "latency_s": sum(t["latency_s"] for t in turns),
            "turn_latencies": [t["latency_s"] for t in turns], "secret": secret, "turns": turns}


def run_benign(level, item, trial, args, backend, threshold, split) -> dict:
    secret = generate_secret(random.Random(f"{args.seed}|{level}|{item['id']}|{trial}"))
    defense = make_defense(level, threshold=threshold) if level >= 3 else make_defense(level)
    session = Session(level=level, secret=secret)
    r = run_turn(defense, session, item["prompt"], backend, temperature=args.temperature,
                 max_tokens=args.max_tokens, seed=_trial_seed(args.seed, trial))
    fp_layer = r.trace.blocked_by or ("model_refusal" if r.refused else None)
    return {"kind": "benign", "level": level, "id": item["id"], "category": item.get("kind"),
            "split": split, "trial": trial, "success": None, "false_positive": r.refused,
            "fp_layer": fp_layer, "blocked_by": r.trace.blocked_by, "refused": r.refused,
            "guard_parse_failures": r.trace.guard_parse_failures, "latency_s": round(r.latency_s, 4),
            "turn_latencies": [round(r.latency_s, 4)], "prompt": item["prompt"],
            "reply": r.reply[:600], "trace": r.trace.to_dict()}


# --------------------------------------------------------------------------- aggregate
def aggregate(records: list[dict], levels: list[int], categories: list[str]) -> dict:
    att = [r for r in records if r["kind"] == "attack"]
    ben = [r for r in records if r["kind"] == "benign"]
    m: dict = {"asr": {}, "asr_by_split": {}, "asr_by_category": {}, "fpr": {}, "fp_layers": {},
               "fpr_by_split": {}, "attribution": {}, "leak_methods": {}, "p50_latency_s": {},
               "guard_parse_failures": {}, "output_filter_saves": {}}
    for lv in levels:
        a = [r for r in att if r["level"] == lv]
        b = [r for r in ben if r["level"] == lv]
        m["asr"][lv] = rate(sum(r["success"] for r in a), len(a))
        m["asr_by_split"][lv] = {s: rate(sum(r["success"] for r in a if r["split"] == s),
                                         sum(1 for r in a if r["split"] == s))
                                 for s in ("dev", "heldout")}
        m["asr_by_category"][lv] = {c: rate(sum(r["success"] for r in a if r["category"] == c),
                                            sum(1 for r in a if r["category"] == c))
                                    for c in categories}
        m["fpr"][lv] = rate(sum(r["false_positive"] for r in b), len(b))
        m["fpr_by_split"][lv] = {s: rate(sum(r["false_positive"] for r in b if r["split"] == s),
                                         sum(1 for r in b if r["split"] == s))
                                 for s in ("dev", "heldout")}
        m["fp_layers"][lv] = dict(Counter(r["fp_layer"] for r in b if r["false_positive"]))
        stopped = [r for r in a if not r["success"]]
        cnt = Counter(r["blocked_by"] for r in stopped)
        m["attribution"][lv] = {k: {"count": v, "share": v / len(stopped)} for k, v in cnt.items()}
        m["leak_methods"][lv] = dict(Counter(r["leak_method"] for r in a if r["success"]))
        lat = [x for r in a + b for x in r["turn_latencies"]]
        m["p50_latency_s"][lv] = statistics.median(lat) if lat else 0.0
        m["guard_parse_failures"][lv] = sum(r["guard_parse_failures"] for r in a + b)
        m["output_filter_saves"][lv] = sum(1 for r in a if r["raw_leak_any"] and not r["success"])
    return m


def headline(m: dict, levels: list[int]) -> str:
    lo, hi = min(levels), max(levels)
    a0, a1, f1 = m["asr"][lo]["rate"], m["asr"][hi]["rate"], m["fpr"][hi]["rate"]
    verb = "dropped" if a1 < a0 else ("rose" if a1 > a0 else "stayed")
    if verb == "stayed":
        return (f"ASR stayed at {pct(a0)} from Level {lo} to Level {hi} "
                f"with a {pct(f1)} false-positive rate at Level {hi}.")
    return (f"ASR {verb} from {pct(a0)} at Level {lo} to {pct(a1)} at Level {hi} "
            f"with a {pct(f1)} false-positive rate.")


def top_breaks(records: list[dict], level: int, k: int = 5) -> list[dict]:
    by_id = defaultdict(list)
    for r in records:
        if r["kind"] == "attack" and r["level"] == level:
            by_id[r["id"]].append(r)
    rows = []
    for aid, rs in by_id.items():
        wins = [r for r in rs if r["success"]]
        if wins:
            rows.append({"id": aid, "category": rs[0]["category"], "successes": len(wins),
                         "trials": len(rs), "methods": sorted({w["leak_method"] for w in wins}),
                         "first_turn": rs[0]["turns"][0]["user"][:120]})
    rows.sort(key=lambda r: (-r["successes"] / r["trials"], r["id"]))
    return rows[:k]


# --------------------------------------------------------------------------- outputs
def write_markdown(path: Path, m: dict, meta: dict, levels, categories, breaks) -> str:
    L = []
    L.append(f"**Headline:** {meta['headline']}\n")
    L.append(f"_Backend `{meta['backend']}` · model `{meta['model']}` · guard `{meta['guard_model']}` · "
             f"trials {meta['trials']} · temperature {meta['temperature']} · seed {meta['seed']} · "
             f"Level 3 similarity threshold {meta['threshold']} ({meta['threshold_source']}) · "
             f"embedder `{meta['embedder']}` · {meta['n_attacks']} attacks, {meta['n_benign']} benign prompts · "
             f"generated {meta['generated_at']}_\n")
    L.append("### ASR and FPR by level (95% Wilson CI)\n")
    L.append("| Level | ASR (all) | ASR (dev) | ASR (held-out) | FPR | p50 latency |")
    L.append("|---|---|---|---|---|---|")
    for lv in levels:
        L.append(f"| {lv} | {fmt_rate(m['asr'][lv])} | {fmt_rate(m['asr_by_split'][lv]['dev'])} | "
                 f"{fmt_rate(m['asr_by_split'][lv]['heldout'])} | {fmt_rate(m['fpr'][lv])} | "
                 f"{m['p50_latency_s'][lv] * 1000:.0f} ms |")
    L.append("\n### ASR by level × attack category\n")
    L.append("| Category | " + " | ".join(f"L{lv}" for lv in levels) + " |")
    L.append("|---|" + "---|" * len(levels))
    for c in categories:
        L.append(f"| {c} | " + " | ".join(
            f"{pct(m['asr_by_category'][lv][c]['rate'])} ({m['asr_by_category'][lv][c]['k']}/"
            f"{m['asr_by_category'][lv][c]['n']})" for lv in levels) + " |")
    L.append("\n### Which layer stopped the attacks that failed\n")
    L.append("| Level | " + " | ".join(LAYER_LABEL[k] for k in LAYER_ORDER) + " |")
    L.append("|---|" + "---|" * len(LAYER_ORDER))
    for lv in levels:
        L.append(f"| {lv} | " + " | ".join(
            pct(m["attribution"][lv].get(k, {}).get("share", 0.0)) for k in LAYER_ORDER) + " |")
    L.append("\n### False positives by layer, leak methods, guard parse failures\n")
    L.append("| Level | False positives by layer | Leak methods (successful attacks) | "
             "Raw leaks caught by output layers | Guard parse failures |")
    L.append("|---|---|---|---|---|")
    for lv in levels:
        fpl = ", ".join(f"{LAYER_LABEL.get(k, k)}: {v}" for k, v in sorted(m["fp_layers"][lv].items())) or "–"
        lm = ", ".join(f"{k}: {v}" for k, v in sorted(m["leak_methods"][lv].items())) or "–"
        L.append(f"| {lv} | {fpl} | {lm} | {m['output_filter_saves'][lv]} | "
                 f"{m['guard_parse_failures'][lv]} |")
    hi = max(levels)
    L.append(f"\n### What still breaks Level {hi}\n")
    if breaks:
        L.append("| Attack | Category | Success | Leak method(s) | Opening message |")
        L.append("|---|---|---|---|---|")
        for b in breaks:
            msg = b["first_turn"].replace("|", "\\|").replace("\n", " ")
            L.append(f"| `{b['id']}` | {b['category']} | {b['successes']}/{b['trials']} | "
                     f"{', '.join(b['methods'])} | {msg} |")
    else:
        L.append(f"No attack leaked the secret at Level {hi} in this run.")
    L.append("\n![ASR and FPR by level](asr_fpr_by_level.png)\n")
    L.append("![ASR heatmap](asr_heatmap.png)\n")
    L.append("![Layer attribution](layer_attribution.png)\n")
    text = "\n".join(L) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def write_csv(path: Path, m: dict, levels, categories) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "level", "group", "k", "n", "rate", "ci_low", "ci_high"])
        for lv in levels:
            for name, r in (("asr", m["asr"][lv]), ("fpr", m["fpr"][lv])):
                w.writerow([name, lv, "all", r["k"], r["n"], f"{r['rate']:.4f}",
                            f"{r['ci_low']:.4f}", f"{r['ci_high']:.4f}"])
            for s in ("dev", "heldout"):
                r = m["asr_by_split"][lv][s]
                w.writerow(["asr_split", lv, s, r["k"], r["n"], f"{r['rate']:.4f}",
                            f"{r['ci_low']:.4f}", f"{r['ci_high']:.4f}"])
            for c in categories:
                r = m["asr_by_category"][lv][c]
                w.writerow(["asr_category", lv, c, r["k"], r["n"], f"{r['rate']:.4f}",
                            f"{r['ci_low']:.4f}", f"{r['ci_high']:.4f}"])


def make_charts(out: Path, m: dict, levels, categories) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # Reference palette (dataviz skill), light mode
    SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "figure.facecolor": SURF, "axes.facecolor": SURF})
    paths = []

    def style(ax):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    # 1) ASR and FPR side by side (two panels, shared 0-100% scale - no dual axis)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, key, color, title in ((axes[0], "asr", BLUE, "Attack success rate"),
                                  (axes[1], "fpr", ORANGE, "False-positive rate (benign)")):
        vals = [m[key][lv]["rate"] * 100 for lv in levels]
        lo = [v - m[key][lv]["ci_low"] * 100 for v, lv in zip(vals, levels)]
        hi = [m[key][lv]["ci_high"] * 100 - v for v, lv in zip(vals, levels)]
        x = range(len(levels))
        ax.bar(x, vals, width=0.55, color=color, edgecolor=SURF, linewidth=2)
        ax.errorbar(x, vals, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1.2, capsize=4)
        for xi, v, h in zip(x, vals, hi):
            ax.text(xi, v + h + 2, f"{v:.0f}%", ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_xticks(list(x), [f"L{lv}" for lv in levels])
        ax.set_ylim(0, 110)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        style(ax)
    axes[0].set_ylabel("% of trials (95% Wilson CI)")
    fig.tight_layout()
    p = out / "asr_fpr_by_level.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # 2) Heatmap level x category, one-hue sequential ramp
    data = [[m["asr_by_category"][lv][c]["rate"] * 100 for lv in levels] for c in categories]
    fig, ax = plt.subplots(figsize=(1.2 * len(levels) + 3, 0.45 * len(categories) + 1.4))
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i, row in enumerate(data):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=9,
                    color="#ffffff" if v >= 55 else INK)
    ax.set_xticks(range(len(levels)), [f"L{lv}" for lv in levels])
    ax.set_yticks(range(len(categories)), categories)
    ax.set_title("ASR by level × attack category", loc="left", fontsize=11)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("ASR %", color=INK2)
    fig.tight_layout()
    p = out / "asr_heatmap.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # 3) Layer attribution, 100% stacked horizontal bars, fixed categorical order
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(levels) + 1.6))
    left = [0.0] * len(levels)
    for idx, layer in enumerate(LAYER_ORDER):
        vals = [m["attribution"][lv].get(layer, {}).get("share", 0.0) * 100 for lv in levels]
        ax.barh(range(len(levels)), vals, left=left, color=CAT[idx], edgecolor=SURF, linewidth=2,
                height=0.6, label=LAYER_LABEL[layer])
        left = [a + b for a, b in zip(left, vals)]
    ax.set_yticks(range(len(levels)), [f"L{lv}" for lv in levels])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of stopped attacks (first layer that blocked)")
    ax.set_title("Which defense layer stopped the attack", loc="left", fontsize=11)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32), fontsize=9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    p = out / "layer_attribution.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)
    return paths


# --------------------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="groq", choices=["groq", "mock"])
    ap.add_argument("--mock", action="store_true", help="shortcut for --backend mock (offline)")
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/")
    ap.add_argument("--no-cache", action="store_true", help="bypass the disk cache")
    ap.add_argument("--threshold", type=float, default=None,
                    help="fix the Level 3 similarity threshold instead of tuning it on the dev split")
    ap.add_argument("--max-tune-fpr", type=float, default=0.10,
                    help="max share of dev benign prompts the tuned input filter may block")
    ap.add_argument("--attacks", default=None, help="path to attacks.yaml")
    ap.add_argument("--benign", default=None, help="path to benign.yaml")
    ap.add_argument("--limit", type=int, default=None, help="only the first N attacks/benign (debug)")
    ap.add_argument("--yes", "-y", action="store_true", help="don't pause after the call estimate")
    ap.add_argument("--db", default=None, help="SQLite path (default: ARENA_DB_PATH)")
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    st = get_settings()
    if args.mock:
        args.backend = "mock"
    args.max_tokens = args.max_tokens or st.max_tokens
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    levels = sorted(set(args.levels))

    attacks = load_attacks(args.attacks or st.attacks_path)
    benign = load_benign(args.benign or st.benign_path)
    if args.limit:
        attacks, benign = attacks[:args.limit], benign[:args.limit]
    categories = list(dict.fromkeys(a["category"] for a in attacks))
    a_split = split_dev_heldout(attacks, "category", args.seed)
    b_split = split_dev_heldout(benign, "kind", args.seed)

    try:
        backend = make_backend(args.backend, st, use_cache=not args.no_cache)
    except BackendConfigError as e:
        sys.exit(f"ERROR: {e}")

    # ---- threshold: tune on dev only
    scorer = get_scorer(st.jailbreak_phrases_path, None, st.cache_dir)
    sim = SimilarityFilter(scorer, load_phrases(st.jailbreak_phrases_path))
    tuning = None
    if args.threshold is not None:
        threshold, source = args.threshold, "fixed via --threshold"
    elif any(lv >= 3 for lv in levels):
        tuning = tune_threshold(sim, [a for a in attacks if a_split[a["id"]] == "dev"],
                                [b for b in benign if b_split[b["id"]] == "dev"], args.max_tune_fpr)
        threshold, source = tuning["chosen"], "tuned on dev split"
    else:
        threshold, source = st.sim_threshold, "config default"

    est = estimate_calls(levels, attacks, benign, args.trials)
    rpm = st.groq_rpm if args.backend == "groq" else 0
    print(f"Backend: {args.backend} | levels {levels} | trials {args.trials} | "
          f"{len(attacks)} attacks ({sum(v == 'dev' for v in a_split.values())} dev / "
          f"{sum(v == 'heldout' for v in a_split.values())} held-out) | {len(benign)} benign")
    print(f"Level 3 similarity threshold: {threshold} ({source}, scorer={scorer.name})")
    print(f"Estimated LLM calls (upper bound, before input-filter short-circuits and cache hits): "
          f"{est['upper_bound_total']}"
          + (f" ≈ {est['upper_bound_total'] / rpm:.0f} min at {rpm} RPM" if rpm else ""))
    if args.backend == "groq" and not args.yes and sys.stdin.isatty():
        print("Tip: run with --trials 1 first as a smoke test. Completed calls are cached, so you can "
              "stop (Ctrl+C) and resume any time. Starting in 5 s...")
        time.sleep(5)

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    db = DB(args.db or st.db_path)
    jsonl = (out / "results.jsonl").open("w", encoding="utf-8")
    records: list[dict] = []
    t_start = time.time()
    jobs = ([("attack", lv, a, tr) for lv in levels for tr in range(args.trials) for a in attacks]
            + [("benign", lv, b, tr) for lv in levels for tr in range(args.trials) for b in benign])
    try:
        for n, (kind, lv, item, tr) in enumerate(jobs, 1):
            if kind == "attack":
                rec = run_attack(lv, item, tr, args, backend, threshold, a_split[item["id"]])
            else:
                rec = run_benign(lv, item, tr, args, backend, threshold, b_split[item["id"]])
            rec["run_id"] = run_id
            records.append(rec)
            jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
            jsonl.flush()
            db.log_bench(run_id, rec)
            if n % 25 == 0 or n == len(jobs):
                s = backend.stats
                print(f"  [{n}/{len(jobs)}] api={s['api_calls']} cache_hits={s.get('cache_hits', 0)} "
                      f"retries={s.get('retries', 0)} elapsed={time.time() - t_start:.0f}s", flush=True)
    finally:
        jsonl.close()

    m = aggregate(records, levels, categories)
    meta = {"run_id": run_id, "backend": args.backend, "model": backend.model,
            "guard_model": backend.guard_model, "trials": args.trials, "temperature": args.temperature,
            "seed": args.seed, "max_tokens": args.max_tokens, "levels": levels, "threshold": threshold,
            "threshold_source": source, "embedder": scorer.name, "n_attacks": len(attacks),
            "n_benign": len(benign), "generated_at": time.strftime("%Y-%m-%d %H:%M"),
            "backend_stats": backend.stats, "elapsed_s": round(time.time() - t_start, 1)}
    meta["headline"] = headline(m, levels)
    breaks = top_breaks(records, max(levels))
    result = {"meta": meta, "metrics": m, "split": {"attacks": a_split, "benign": b_split},
              "threshold_tuning": tuning, "top_breaks": breaks}
    (out / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(out / "results.csv", m, levels, categories)
    write_markdown(out / "results.md", m, meta, levels, categories, breaks)
    make_charts(out, m, levels, categories)
    print("\n" + meta["headline"])
    print(f"Wrote {out / 'results.json'}, results.csv, results.md, results.jsonl and charts.")
    return result


if __name__ == "__main__":
    main()
