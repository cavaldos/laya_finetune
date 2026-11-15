"""Turn `src/evaluate/llm_dataset.py` output into Laya training rows.

Input:  one or more `dataset.jsonl` files (one row per closed backtest trade).
Output: Laya JSONL rows `{state, questions, expected}` + a metadata list used for evaluation.

Steps:
1. load every `dataset.jsonl` under the data folder (all coins together);
2. one global chronological split for all coins: train (oldest) / val / test (newest). No
   per-coin split: altcoins move with BTC, so a per-coin split leaks the future. Embargo: a
   trade still open when the next period starts is dropped;
3. optional text transforms: keep only some sections, anonymize the coin, coin group;
4. write `train.jsonl` / `val.jsonl` / `test.jsonl` in the Laya row format.

Only the review text goes to the model. Outcome fields (r, exit_time, label) never do.
"""

from __future__ import annotations

import gzip
import json
import math
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

QUESTION_ID = "outcome"
QUESTION = {
    "type": "choice",
    "instructions": ("A backtest trade is about to be opened as described. Predict its outcome: "
                     "does price reach the take profit before the stop loss?"),
    "criteria": {
        "win": "the take profit is hit first",
        "loss": "the stop loss is hit first",
    },
}
LABELS = ("win", "loss")

# Sections of the review text (the `## NAME` headings written by the live reviewer).
SECTION_PRESETS = {
    "all": None,
    # The summary sections: about 1-2k tokens instead of 6-7k.
    "compact": ["PLAN", "KEY FACTS", "SETUP", "TREND", "STRUCTURE"],
    "facts": ["PLAN", "KEY FACTS", "SETUP"],
}

# Lines that are identical in every message: they cost tokens and carry no signal.
BOILERPLATE_PREFIXES = (
    "How to read this message",
    "- Prices are exact",
    "Answer with the JSON object only",
)

# --coin-group: replace the coin name with a coarse bucket.
COIN_GROUPS = {
    "major": {"BTC", "ETH"},
    "large-cap alt": {"SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "LINK", "DOT", "TRX", "LTC",
                      "BCH", "TON", "SUI", "NEAR", "APT", "HBAR", "XLM", "UNI", "ATOM"},
}
DEFAULT_GROUP = "small-cap alt"

_NUM = r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])"
_TIME = re.compile(r"\b\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?\b|\b\d{2}:\d{2}(?::\d{2})?\b|\b\d{4}-\d{2}-\d{2}\b")
# A number directly followed by one of these is not a price (%, R, x, ATR, seconds, candles...).
_NOT_PRICE_AFTER = re.compile(r"\s*(%|R\b|x\b|ATR|s\b|candles?|min|h\b|m\b|d\b|of\b|levels?|obstacles?|more\b|closed|swings)")


@dataclass
class Trade:
    row: dict
    text: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def entry_time(self) -> datetime:
        return _ts(self.row["entry_time"])

    @property
    def exit_time(self) -> Optional[datetime]:
        v = self.row.get("exit_time")
        return _ts(v) if v else None


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("T", " ").replace("Z", ""))


# --------------------------------------------------------------------------------- loading

def find_dataset_files(data: Path) -> List[Path]:
    """Every dataset.jsonl (or dataset.jsonl.gz) under `data`, or `data` itself when it is a file."""
    if data.is_file():
        return [data]
    files = sorted(set(data.rglob("dataset.jsonl")) | set(data.rglob("dataset.jsonl.gz")))
    if not files:
        raise FileNotFoundError(f"no dataset.jsonl under {data}")
    return files


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def load_trades(files: Sequence[Path], symbols: Optional[Sequence[str]] = None) -> List[Trade]:
    wanted = {s.upper() for s in symbols} if symbols else None
    trades, seen = [], set()
    for f in files:
        with _open_text(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("label") not in LABELS or not row.get("text"):
                    continue
                if wanted and _base(row["symbol"]) not in wanted and row["symbol"].upper() not in wanted:
                    continue
                if row["id"] in seen:        # same trade exported twice
                    continue
                seen.add(row["id"])
                trades.append(Trade(row=row))
    trades.sort(key=lambda t: (t.entry_time, t.row["symbol"]))
    return trades


# ---------------------------------------------------------------------------------- split

def _cut(trades: List[Trade], frac: float, embargo: bool):
    """Split the time-sorted `trades` at one time: `(before, after, cutoff, dropped)`.
    With `embargo`, trades of the earlier part that close after the cutoff are dropped."""
    n_after = max(1, int(round(len(trades) * frac)))
    cutoff = trades[len(trades) - n_after].entry_time
    after = [t for t in trades if t.entry_time >= cutoff]
    before = [t for t in trades if t.entry_time < cutoff]
    kept = [t for t in before if t.exit_time is None or t.exit_time < cutoff] if embargo else before
    return kept, after, cutoff, len(before) - len(kept)


def chrono_split(trades: List[Trade], test_frac: float, val_frac: float = 0.0, embargo: bool = True):
    """`(train, val, test, info)`: oldest trades = train, then val, newest = test.

    One cutoff time for every coin (a per-coin split leaks: coins move together). `val_frac`
    is a share of all trades, taken just before the test period; it picks the best epoch and
    fits the probability temperature, so the test split is touched only once, at the end.
    """
    if not 0 < test_frac < 1 or not 0 <= val_frac < 1 or test_frac + val_frac >= 1:
        raise ValueError("need 0 < test_frac < 1, 0 <= val_frac < 1 and test_frac + val_frac < 1")
    rest, test, test_cut, dropped_test = _cut(trades, test_frac, embargo)
    val, val_cut, dropped_val = [], None, 0
    if val_frac > 0:
        rest, val, val_cut, dropped_val = _cut(rest, val_frac / (1 - test_frac), embargo)
    info = {"test_cutoff": str(test_cut), "val_cutoff": str(val_cut) if val_cut else None,
            "embargo_dropped": dropped_test + dropped_val}
    return rest, val, test, info


def sample(trades: List[Trade], n: Optional[int], seed: int) -> List[Trade]:
    """Random subset of `n` (kept in time order); stratified so both labels appear."""
    if not n or n >= len(trades):
        return trades
    rng = random.Random(seed)
    by_label = {lab: [t for t in trades if t.row["label"] == lab] for lab in LABELS}
    picked = []
    for lab, group in by_label.items():
        k = round(n * len(group) / len(trades))
        k = min(len(group), max(1 if group else 0, k))
        picked += rng.sample(group, k)
    picked.sort(key=lambda t: (t.entry_time, t.row["symbol"]))
    return picked


# ------------------------------------------------------------------------- text transforms

def _base(symbol: str) -> str:
    s = symbol.upper()
    for quote in ("USDT", "USDC", "USD", "PERP"):
        if s.endswith(quote) and len(s) > len(quote):
            return s[: -len(quote)]
    return s


def coin_group(symbol: str) -> str:
    base = _base(symbol)
    for name, members in COIN_GROUPS.items():
        if base in members:
            return name
    return DEFAULT_GROUP


def select_sections(text: str, sections: Optional[Sequence[str]]) -> str:
    """Keep the header (minus boilerplate) and the `## NAME` sections listed, in their original order."""
    out, keep = [], True
    for line in text.split("\n"):
        if line.startswith("## "):
            name = line[3:].strip().upper()
            keep = sections is None or any(name.startswith(s.upper()) for s in sections)
        if not keep or line.startswith(BOILERPLATE_PREFIXES):
            continue
        out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def anonymize(text: str, row: dict) -> str:
    """Hide which coin this is: name -> COIN, absolute prices -> signed R from the entry,
    absolute distances -> % of price. Heuristic: a number within +-30% of the entry price
    that is not followed by a unit (%, R, x, ATR, ...) is taken as a price."""
    entry = float(row["entry"])
    risk = abs(entry - float(row["stop_loss"])) or entry * 1e-3
    sign = 1.0 if str(row.get("side", "LONG")).upper() == "LONG" else -1.0
    sym, base = row["symbol"].upper(), _base(row["symbol"])

    text = re.sub(re.escape(sym), "COIN", text, flags=re.I)
    if len(base) >= 3:
        text = re.sub(rf"\b{re.escape(base)}\b", "COIN", text, flags=re.I)

    # Absolute distances: "R = ... = 240.424", "ATR = ... = 117.779", "601.061 from the entry".
    def pct(m):
        return f"{m.group(1)}{float(m.group(2)) / entry * 100:.3f}% of price"
    text = re.sub(r"^(- (?:R|ATR) = .*?= )(\d+(?:\.\d+)?)", pct, text, flags=re.M)
    text = re.sub(r"(?<![\w.])(\d+(?:\.\d+)?) from the entry",
                  lambda m: f"{float(m.group(1)) / entry * 100:.3f}% from the entry", text)

    # Mask times so their digits are never read as prices.
    masks: List[str] = []

    def mask(m):
        masks.append(m.group(0))
        return f"\x00{len(masks) - 1}\x00"
    text = _TIME.sub(mask, text)

    lo, hi = entry * 0.7, entry * 1.3
    rr_table = re.compile(r"^\s+(T\d+\s*)?\|")     # the "RR of each combination" grid

    def price(m, line):
        v = float(m.group(1))
        if not lo <= v <= hi or _NOT_PRICE_AFTER.match(line, m.end()):
            return m.group(0)
        return f"[{round(sign * (v - entry) / risk, 2) + 0.0:+.2f}R]"

    lines = []
    for line in text.split("\n"):
        if not rr_table.match(line):
            line = re.sub(_NUM, lambda m, ln=line: price(m, ln), line)
        lines.append(line)
    text = "\n".join(lines)
    return re.sub("\x00(\\d+)\x00", lambda m: masks[int(m.group(1))], text)


def build_text(trade: Trade, sections: Optional[Sequence[str]], anon: bool, group: bool) -> str:
    text = select_sections(trade.row["text"], sections)
    if anon:
        text = anonymize(text, trade.row)
    if group:
        # The name goes, the bucket stays. Use with --anonymize, or prices still give the coin away.
        text = re.sub(re.escape(trade.row["symbol"]), "COIN", text, flags=re.I)
        first, _, rest = text.partition("\n")
        text = f"{first} | coin group: {coin_group(trade.row['symbol'])}\n{rest}"
    return text


# ---------------------------------------------------------------------------------- export

META_FIELDS = ("id", "symbol", "side", "signal_tf", "entry_time", "exit_time", "label", "r", "rr",
               "zone_type", "sl_source", "tp_source")


def to_laya_row(trade: Trade) -> dict:
    return {"state": trade.text, "questions": {QUESTION_ID: QUESTION},
            "expected": {QUESTION_ID: trade.row["label"]}}


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


SPLITS = ("train", "val", "test")


def read_jsonl(path: Path) -> List[dict]:
    with _open_text(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def prepare(data: Path, out_dir: Path, *, test_frac: float, val_frac: float,
            sections: Optional[Sequence[str]], anon: bool, group: bool,
            symbols: Optional[Sequence[str]] = None, limits: Optional[Dict[str, Optional[int]]] = None,
            embargo: bool = True, seed: int = 0) -> dict:
    """Write `<split>.jsonl` (Laya rows) and `<split>_meta.jsonl` (trade info) for train / val / test."""
    files = find_dataset_files(data)
    trades = load_trades(files, symbols)
    if len(trades) < 6:
        raise ValueError(f"only {len(trades)} usable trades in {data}")
    train, val, test, info = chrono_split(trades, test_frac, val_frac, embargo)
    limits = limits or {}
    splits = {name: sample(group_, limits.get(name), seed + i)
              for i, (name, group_) in enumerate(zip(SPLITS, (train, val, test)))}
    for group_ in splits.values():
        for t in group_:
            t.text = build_text(t, sections, anon, group)
            t.meta = {k: t.row.get(k) for k in META_FIELDS}
    for name, group_ in splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", [to_laya_row(t) for t in group_])
        write_jsonl(out_dir / f"{name}_meta.jsonl", [t.meta for t in group_])

    def stats(group_: List[Trade]) -> dict:
        n = len(group_)
        wins = sum(t.row["label"] == "win" for t in group_)
        by_sym: Dict[str, int] = {}
        for t in group_:
            by_sym[t.row["symbol"]] = by_sym.get(t.row["symbol"], 0) + 1
        return {"n": n, "wins": wins, "win_rate": round(wins / n, 4) if n else math.nan,
                "from": str(group_[0].entry_time) if n else None,
                "to": str(group_[-1].entry_time) if n else None, "by_symbol": by_sym}

    everything = [t for g in splits.values() for t in g]
    return {"files": [str(f) for f in files], "total_trades": len(trades), **info,
            **{name: stats(g) for name, g in splits.items()},
            "chars_median": sorted(len(t.text) for t in everything)[len(everything) // 2]}
