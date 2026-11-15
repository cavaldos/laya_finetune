"""Metrics for p(win) predictions: probability quality, ranking, and what it means in R.

No sklearn: everything here is a few lines of numpy so the folder needs only laya + torch.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

EPS = 1e-6


def auc(y: Sequence[int], p: Sequence[float]) -> Optional[float]:
    """ROC AUC (Mann-Whitney, ties = half). None when only one class is present."""
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    sorted_p = p[order]
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def log_loss(y, p) -> float:
    y, p = np.asarray(y, dtype=float), np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def prob_metrics(y, p) -> Dict[str, Optional[float]]:
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    a = auc(y, p)
    return {
        "n": int(len(y)),
        "log_loss": round(log_loss(y, p), 4),
        "brier": round(brier(y, p), 4),
        "auc": round(a, 4) if a is not None else None,
        "accuracy@0.5": round(float(np.mean((p >= 0.5) == y)), 4),
        "mean_p_win": round(float(p.mean()), 4),
        "p_win_spread": round(float(p.max() - p.min()), 4) if len(p) else 0.0,
    }


def per_symbol_auc(y, p, symbols) -> Dict[str, dict]:
    groups = defaultdict(list)
    for yi, pi, s in zip(y, p, symbols):
        groups[s].append((yi, pi))
    out = {}
    for s, pairs in sorted(groups.items()):
        ys, ps = zip(*pairs)
        a = auc(ys, ps)
        out[s] = {"n": len(ys), "win_rate": round(float(np.mean(ys)), 4),
                  "auc": round(a, 4) if a is not None else None}
    return out


def within_symbol_auc(per_symbol: Dict[str, dict]) -> Optional[float]:
    """Trade-weighted mean of per-symbol AUCs: ranking skill that per-coin base rates cannot fake."""
    pairs = [(v["n"], v["auc"]) for v in per_symbol.values() if v["auc"] is not None]
    if not pairs:
        return None
    return round(sum(n * a for n, a in pairs) / sum(n for n, _ in pairs), 4)


def trading(y, p, r, rr) -> Dict[str, dict]:
    """Gross R (fees not included) if only the trades the model likes were taken."""
    y, p, r, rr = (np.asarray(v, dtype=float) for v in (y, p, r, rr))

    def block(mask):
        n = int(mask.sum())
        if n == 0:
            return {"n": 0, "win_rate": None, "mean_r": None, "sum_r": 0.0}
        return {"n": n, "win_rate": round(float(y[mask].mean()), 4),
                "mean_r": round(float(r[mask].mean()), 4), "sum_r": round(float(r[mask].sum()), 2)}

    breakeven = 1.0 / (1.0 + np.where(rr > 0, rr, np.inf))   # WR a trade needs at its RR
    order = np.argsort(-p, kind="mergesort")
    top_half = np.zeros(len(p), dtype=bool)
    top_half[order[: max(1, len(p) // 2)]] = True
    top_third = np.zeros(len(p), dtype=bool)
    top_third[order[: max(1, len(p) // 3)]] = True
    return {
        "all_trades": block(np.ones(len(p), dtype=bool)),
        "p_win_above_breakeven": block(p > breakeven),
        "top_half_by_p": block(top_half),
        "top_third_by_p": block(top_third),
        "bottom_half_by_p": block(~top_half),
    }


def full_report(y, p, r, rr, symbols) -> dict:
    per_sym = per_symbol_auc(y, p, symbols)
    return {**prob_metrics(y, p), "within_symbol_auc": within_symbol_auc(per_sym),
            "per_symbol": per_sym, "trading": trading(y, p, r, rr)}


def _f(v, fmt="%.4f"):
    return "-" if v is None else fmt % v


def format_report(report: dict) -> str:
    """The test report as plain text (printed and saved as test_report.txt)."""
    models = report["models"]
    lines = ["", "=" * 80,
             "TEST RESULT: %d trades, win rate %.1f%% (train win rate %.1f%%)"
             % (report["test_n"], 100 * report["test_win_rate"], 100 * report["train_win_rate"]),
             "=" * 80,
             "%-18s %8s %8s %8s %10s %8s %9s" % ("model", "logloss", "brier", "AUC", "AUC/coin", "acc", "p spread")]
    for name, m in models.items():
        lines.append("%-18s %8s %8s %8s %10s %8s %9s" % (
            name, _f(m["log_loss"]), _f(m["brier"]), _f(m["auc"]), _f(m["within_symbol_auc"]),
            _f(m["accuracy@0.5"]), _f(m["p_win_spread"])))
    lines += ["",
              "logloss / brier lower than base_rate = better probabilities; AUC 0.5 = no ranking skill.",
              "AUC/coin = AUC inside each coin, averaged: the number that counts when coins are mixed."]

    best = models["laya_finetuned"]
    if best["p_win_spread"] < 0.02:
        lines += ["",
                  "WARNING: laya_finetuned gives (almost) the same p(win) to every trade (spread %.4f):"
                  % best["p_win_spread"],
                  "  it collapsed to one answer, so its AUC and the R table below are noise.",
                  "  Try more data / epochs, --loss soft-ce, or a lower --encoder-lr."]
    lines += ["", "Per coin (laya_finetuned):"]
    for s, v in best["per_symbol"].items():
        lines.append("  %-12s n=%-5d win %5.1f%%  AUC %s" % (s, v["n"], 100 * v["win_rate"], _f(v["auc"])))

    lines += ["", "If only the trades the model likes were taken (gross R, fees NOT included):",
              "  %-24s %6s %8s %8s %8s" % ("", "trades", "win%", "mean R", "sum R")]
    labels = {"all_trades": "all trades (no filter)", "p_win_above_breakeven": "p(win) > 1/(1+RR)",
              "top_half_by_p": "top 50% by p(win)", "top_third_by_p": "top 33% by p(win)",
              "bottom_half_by_p": "bottom 50% by p(win)"}
    for key, label in labels.items():
        t = best["trading"][key]
        wr = None if t["win_rate"] is None else 100 * t["win_rate"]
        lines.append("  %-24s %6d %8s %8s %8s" % (label, t["n"], _f(wr, "%.1f"), _f(t["mean_r"], "%+.3f"),
                                                  _f(t["sum_r"], "%+.1f")))
    lines += ["=" * 80, ""]
    return "\n".join(lines)


# -------------------------------------------------------------------------- tabular baseline

TAB_CATEGORICAL = ("side", "signal_tf", "zone_type", "sl_source", "tp_source")


def _features(meta: List[dict], vocab: Optional[Dict[str, List[str]]], use_symbol: bool):
    cats = list(TAB_CATEGORICAL) + (["symbol"] if use_symbol else [])
    if vocab is None:
        vocab = {c: sorted({str(m.get(c)) for m in meta}) for c in cats}
    cols = []
    for m in meta:
        row = []
        for c in cats:
            v = str(m.get(c))
            row += [1.0 if v == k else 0.0 for k in vocab[c]]
        rr = float(m.get("rr") or 0.0)
        hour = int(str(m["entry_time"])[11:13]) if len(str(m["entry_time"])) >= 13 else 0
        row += [rr, math.log1p(max(rr, 0.0)), math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24)]
        cols.append(row)
    return np.asarray(cols, dtype=float), vocab


def logistic_baseline(train_meta: List[dict], test_meta: List[dict], use_symbol: bool,
                      l2: float = 1.0, iters: int = 2000, lr: float = 0.1) -> np.ndarray:
    """p(win) on test from an L2 logistic regression on the plain trade fields (no text)."""
    x_tr, vocab = _features(train_meta, None, use_symbol)
    x_te, _ = _features(test_meta, vocab, use_symbol)
    y = np.asarray([1.0 if m["label"] == "win" else 0.0 for m in train_meta])
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-9
    x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd
    w, b = np.zeros(x_tr.shape[1]), math.log((y.mean() + EPS) / (1 - y.mean() + EPS))
    n = len(y)
    for _ in range(iters):
        z = np.clip(x_tr @ w + b, -30, 30)
        g = 1 / (1 + np.exp(-z)) - y
        w -= lr * (x_tr.T @ g / n + l2 * w / n)
        b -= lr * g.mean()
    return 1 / (1 + np.exp(-np.clip(x_te @ w + b, -30, 30)))
