"""
Price-direction dataset: what the chart showed up to a moment (price path + indicators) -> how
price moved AFTER it (up / down and by how many %).

Unlike llm_dataset.py (win / loss of one SMC trade, label depends on SL / TP), the label here is
the market itself: the close `horizon` signal candles later vs the price at the decision time.

Anchors (the decision moments):
  - trades (default): entry times of the backtest trades (any result), on their signal timeframe;
  - grid: every `--every` candles of `--tf` (more rows, no strategy bias).
Only candles closed at the decision time are used for the input (no look-ahead); the 1m candles
after it give the result.

Input, two modes (--mode text | image | both):
  - text: price path + indicators as text, every price as % from the current price (no absolute
    prices, no coin name unless --show-symbol), ~1.75k Laya tokens: KEY FACTS, INDICATORS (signal
    TF + 1h + 4h), STRUCTURE (SMC trend, BOS / CHoCH, sweeps, zones), CANDLES (1h, signal TF), DAY;
  - image: candle chart PNG of the last --image-bars signal candles with EMA 20/50/200, Bollinger,
    volume, RSI and MACD panels (matplotlib, no browser, no price axis).

Label of each row (main horizon = the first of --horizons):
    label        "up" / "down" (close after the horizon vs the price now; equal = down)
    ret_pct      % change after the horizon
    max_up_pct   highest high in the horizon, % from now   max_down_pct  lowest low, % from now
    atr_pct      ATR(14) of the signal TF in % of price (to compare moves between coins)
    targets      the same for every horizon in --horizons
    features     the main numbers of the text (for a tabular baseline)
    exit_time    end of the main horizon (embargo of the train / test split)

Output (default res/datasets/price_direction/<coin>/), one folder per coin:
    up/text/<id>.txt   up/image/<id>.png   down/...      (with --test > 0: train/up/..., test/up/...)
    dataset.jsonl      one row per anchor: id, text, image, label, ret_pct, targets, features, ...
    dataset.csv        same without the text
    build.json         settings of the dataset (horizons, timeframe...): a run with other settings
                       on the same folder stops (use another --out, or --rebuild)
Running again continues: anchors already in dataset.jsonl are skipped.

    .venv/bin/python -m src.evaluate.price_dataset            # no arguments: menu
    .venv/bin/python -m src.evaluate.price_dataset --symbol BTCUSDT ETHUSDT --limit 20
    ... --anchors grid --tf 15m --every 4                     # every hour, 15m chart
    ... --mode image  /  --mode both
    ... --horizons 12,4,24                                    # main label after 12 signal candles
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.core.timeframe import convert_timeframe, timeframe_to_minutes

TEXT_FORMAT = "price-v2"  # v2: candles as integers, 1h before the signal TF, DAY last (~1.75k tokens)
DEFAULT_OUT = "res/datasets/price_direction"
WARMUP_SIGNAL = 260        # signal candles before an anchor (EMA 200 + slack)
STRUCTURE_BARS = 300       # signal candles the SMC structure is read from
HTF_LONG_EMA = {"1h": 200, "4h": 100}  # higher timeframes in the text and their long EMA (4h EMA 200 = 33 days)
DAY_MINUTES = 48 * 60      # 1m candles for the day context (previous UTC day + last 24h)
SIGNAL_ROWS = 24           # signal candles written in the CANDLES section
HTF_ROWS = 16              # 1h candles written in the CANDLES section
CSV_FIELDS = (
    "id", "split", "label", "label_id", "ret_pct", "max_up_pct", "max_down_pct", "atr_pct", "symbol",
    "timeframe", "anchor", "side", "entry_time", "exit_time", "horizon_bars", "price", "image", "text_file",
)


# ==================== indicators (whole series once, causal) ====================

def _ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def _rma(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """EMA 20/50/200, RSI 14, MACD 12/26/9, ATR 14, Bollinger 20/2, ADX 14, volume ratio.
    Every value at row i uses rows <= i only."""
    df = df.reset_index(drop=True).copy()
    close, high, low = df["Close"], df["High"], df["Low"]
    for n in (20, 50, 100, 200):
        df[f"ema{n}"] = _ema(close, n)
    delta = close.diff()
    gain, loss = _rma(delta.clip(lower=0), 14), _rma(-delta.clip(upper=0), 14)
    df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    df.loc[loss == 0, "rsi"] = 100.0
    macd = _ema(close, 12) - _ema(close, 26)
    df["macd_hist"] = macd - _ema(macd, 9)
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    df["atr"] = _rma(tr, 14)
    mid, sd = close.rolling(20).mean(), close.rolling(20).std(ddof=0)
    df["bb_mid"], df["bb_up"], df["bb_low"] = mid, mid + 2 * sd, mid - 2 * sd
    df["bb_width"] = (4 * sd) / mid
    up_move, down_move = high.diff(), -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_adx = _rma(tr, 14)
    plus_di = 100 * _rma(pd.Series(plus_dm), 14) / atr_adx
    minus_di = 100 * _rma(pd.Series(minus_dm), 14) / atr_adx
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx"], df["plus_di"], df["minus_di"] = _rma(dx, 14), plus_di, minus_di
    df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    return df


def resample(m1: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Candles of `tf` from 1m candles (Time = open time), with indicators and close time."""
    df = m1 if tf == "1m" else convert_timeframe(m1, tf)
    df = add_indicators(df.dropna(subset=["Open", "Close"]))
    df["CloseTime"] = df["Time"] + pd.Timedelta(minutes=timeframe_to_minutes(tf))
    return df


def closed_index(frame: pd.DataFrame, decision: pd.Timestamp) -> int:
    """Row of the last candle closed at the decision time (-1 = none)."""
    return int(np.searchsorted(frame["CloseTime"].to_numpy(), np.datetime64(decision), side="right")) - 1


# ==================== one symbol's data ====================

@dataclass
class Anchor:
    symbol: str
    tf: str                     # signal timeframe of the chart
    decision: pd.Timestamp      # every candle closed at this time is known; the result starts here
    kind: str                   # "trade" | "grid"
    side: str = ""              # the trade's side (anchors=trades), never written into the text
    entry_time: Optional[pd.Timestamp] = None

    @property
    def id(self) -> str:
        return f"{self.decision:%Y%m%d-%H%M}_{self.symbol}_{self.tf}"


class Market:
    """1m candles of one symbol + the timeframes the text needs, indicators computed once."""

    def __init__(self, data_folder: str, symbol: str):
        from src.live.core.snapshot import _load_ohlcv, _resolve_data_file

        self.symbol = symbol
        m1 = _load_ohlcv(_resolve_data_file(data_folder, symbol, "1m"))
        self.m1 = m1[["Time", "Open", "High", "Low", "Close", "Volume"]].astype(
            {"Open": float, "High": float, "Low": float, "Close": float, "Volume": float})
        self.m1_close_time = (self.m1["Time"] + pd.Timedelta(minutes=1)).to_numpy()
        self.frames: dict[str, pd.DataFrame] = {}

    def frame(self, tf: str) -> pd.DataFrame:
        if tf not in self.frames:
            self.frames[tf] = resample(self.m1, tf)
        return self.frames[tf]

    @property
    def first(self) -> pd.Timestamp:
        return self.m1["Time"].iloc[0]

    @property
    def last_close(self) -> pd.Timestamp:
        return self.m1["Time"].iloc[-1] + pd.Timedelta(minutes=1)

    def price_at(self, decision: pd.Timestamp) -> Optional[float]:
        i = int(np.searchsorted(self.m1_close_time, np.datetime64(decision), side="right")) - 1
        if i < 0 or self.m1_close_time[i] < np.datetime64(decision - pd.Timedelta(minutes=5)):
            return None  # a gap in the data right before the decision
        return float(self.m1["Close"].iloc[i])

    def future(self, decision: pd.Timestamp, minutes: int) -> Optional[dict]:
        """Close / high / low of the 1m candles in (decision, decision + minutes]; None when the
        data does not cover the whole horizon."""
        start = int(np.searchsorted(self.m1_close_time, np.datetime64(decision), side="right"))
        end = int(np.searchsorted(self.m1_close_time, np.datetime64(decision + pd.Timedelta(minutes=minutes)),
                                  side="right"))
        if end <= start:
            return None
        last_close = self.m1_close_time[end - 1]
        if last_close < np.datetime64(decision + pd.Timedelta(minutes=minutes - 2)):
            return None  # data ends (or has a gap) before the horizon does
        part = self.m1.iloc[start:end]
        if len(part) < 0.9 * minutes:
            return None
        return {"close": float(part["Close"].iloc[-1]), "high": float(part["High"].max()),
                "low": float(part["Low"].min())}


def history_minutes(tf: str) -> int:
    """Minutes of 1m data the text of a `tf` chart needs before the decision."""
    tf_min = timeframe_to_minutes(tf)
    need = (WARMUP_SIGNAL + 1) * tf_min
    for htf, long in HTF_LONG_EMA.items():
        if timeframe_to_minutes(htf) > tf_min:
            need = max(need, (long + 12) * timeframe_to_minutes(htf))
    return max(need, DAY_MINUTES)


# ==================== text ====================

def pct(value: float, base: float) -> float:
    return (value / base - 1.0) * 100.0 if base else 0.0


def sp(value: Optional[float], digits: int = 2) -> str:
    """Signed % text: +0.42%."""
    if value is None or not np.isfinite(value):
        return "?"
    return f"{value:+.{digits}f}%"


def num(value: Optional[float], digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "?"
    return f"{value:.{digits}f}"


def _ago(bars: int) -> str:
    return "this candle" if bars <= 0 else f"{bars} candle{'s' if bars != 1 else ''} ago"


def indicator_block(frame: pd.DataFrame, i: int, tf: str, price: float, long: int = 200) -> tuple[list[str], dict]:
    """INDICATORS lines of one timeframe at row i + their numbers (EMA 20 / 50 / `long`)."""
    row = frame.iloc[i]
    close = frame["Close"]

    def ret(n: int) -> Optional[float]:
        return pct(price, float(close.iloc[i - n])) if i - n >= 0 else None

    def slope(col: str, n: int = 5) -> Optional[float]:
        if i - n < 0 or not np.isfinite(frame[col].iloc[i - n]):
            return None
        return pct(float(frame[col].iloc[i]), float(frame[col].iloc[i - n]))

    emas = {n: float(row[f"ema{n}"]) for n in (20, 50, long)}
    stack = "above all" if all(price > v for v in emas.values()) else \
        "below all" if all(price < v for v in emas.values()) else "mixed"
    order = f"20>50>{long} (bull stack)" if emas[20] > emas[50] > emas[long] else \
        f"20<50<{long} (bear stack)" if emas[20] < emas[50] < emas[long] else "tangled"
    lo100, hi100 = float(frame["Low"].iloc[max(0, i - 99):i + 1].min()), float(frame["High"].iloc[max(0, i - 99):i + 1].max())
    range_pos = (price - lo100) / (hi100 - lo100) * 100 if hi100 > lo100 else 50.0
    atr_pct = float(row["atr"]) / price * 100
    atr_med = float((frame["atr"] / frame["Close"]).iloc[max(0, i - 99):i + 1].median()) * 100
    bbw_med = float(frame["bb_width"].iloc[max(0, i - 99):i + 1].median())
    bb_pos = (price - float(row["bb_low"])) / (float(row["bb_up"]) - float(row["bb_low"])) * 100 \
        if float(row["bb_up"]) > float(row["bb_low"]) else 50.0
    rsi_prev = float(frame["rsi"].iloc[i - 5]) if i >= 5 else float("nan")
    hist = float(row["macd_hist"]) / price * 100
    hist_prev = float(frame["macd_hist"].iloc[i - 1]) / price * 100 if i >= 1 else float("nan")
    vol5 = float(frame["Volume"].iloc[max(0, i - 4):i + 1].mean())
    vol50 = float(frame["Volume"].iloc[max(0, i - 49):i + 1].mean())
    up_bars = int((frame["Close"].iloc[max(0, i - 9):i + 1] > frame["Open"].iloc[max(0, i - 9):i + 1]).sum())

    feats = {
        "ret_1": ret(1), "ret_4": ret(4), "ret_12": ret(12), "ret_48": ret(48),
        "ema20_pct": pct(price, emas[20]), "ema50_pct": pct(price, emas[50]), "ema_long_pct": pct(price, emas[long]),
        "ema20_slope": slope("ema20"), "rsi": float(row["rsi"]), "rsi_chg5": float(row["rsi"]) - rsi_prev,
        "macd_hist_pct": hist, "adx": float(row["adx"]), "di_diff": float(row["plus_di"] - row["minus_di"]),
        "atr_pct": atr_pct, "atr_vs_med": atr_pct / atr_med if atr_med else None, "bb_pos": bb_pos,
        "bb_width_vs_med": float(row["bb_width"]) / bbw_med if bbw_med else None, "range100_pos": range_pos,
        "vol_ratio": float(row["vol_ratio"]), "vol5_vs_50": vol5 / vol50 if vol50 else None, "up_bars_10": up_bars,
    }
    rsi = float(row["rsi"])
    rsi_zone = " overbought" if rsi >= 70 else " oversold" if rsi <= 30 else ""
    lines = [
        f"### {tf}",
        f"- Change: 1 candle {sp(feats['ret_1'])}, 4 {sp(feats['ret_4'])}, 12 {sp(feats['ret_12'])}, "
        f"48 {sp(feats['ret_48'])}; up candles in the last 10: {up_bars}",
        f"- EMA: price {stack}; EMA20 {sp(feats['ema20_pct'])}, EMA50 {sp(feats['ema50_pct'])}, "
        f"EMA{long} {sp(feats['ema_long_pct'])} from price; order {order}; EMA20 slope (5 candles) {sp(feats['ema20_slope'], 3)}",
        f"- RSI14 {num(rsi)}{rsi_zone} (5 candles ago {num(rsi_prev)}); MACD hist {sp(hist, 3)} of price, "
        f"{'rising' if hist > hist_prev else 'falling'}",
        f"- ADX14 {num(feats['adx'])}, +DI {num(float(row['plus_di']))} / -DI {num(float(row['minus_di']))}",
        f"- ATR14 {atr_pct:.3f}% of price ({num(feats['atr_vs_med'], 2)}x its 100-candle median); Bollinger "
        f"position {num(bb_pos, 0)}% (0 = lower band, 100 = upper), width {num(feats['bb_width_vs_med'], 2)}x median",
        f"- Range of the last 100 candles: high {sp(pct(hi100, price))}, low {sp(pct(lo100, price))}, "
        f"price at {num(range_pos, 0)}% of it",
        f"- Volume: last candle {num(feats['vol_ratio'], 2)}x its 20-candle average; last 5 vs last 50 "
        f"{num(feats['vol5_vs_50'], 2)}x",
    ]
    return lines, feats


def structure_block(frame: pd.DataFrame, i: int, tf: str, price: float, params: dict) -> tuple[list[str], dict]:
    """SMC structure of the signal TF (BOS / CHoCH, swings, sweeps, zones), prices as % from now."""
    from src.indicators.export import run_indicators
    from src.live.core.llm_structure import STRUCTURE_INDICATORS, read_structure

    window = frame.iloc[max(0, i - STRUCTURE_BARS + 1):i + 1][["Time", "Open", "High", "Low", "Close", "Volume"]]
    window = window.reset_index(drop=True)
    last = len(window) - 1
    st = read_structure(window, timeframe=tf, indicator_params=params, events=8, sweep_bars=40, chart_bars=0)
    ind = run_indicators(window, include=["bos", "choch", "swing"], **params)
    trend = {t.scope: t.direction for t in st.trends}
    lines = ["## STRUCTURE (" + tf + ", smart money concepts)",
             f"- Trend: swing {trend.get('swing', 'unclear')}, internal {trend.get('internal', 'unclear')}"]

    events = []
    for brk in ind.get("bos", []) + ind.get("choch", []):
        idx = brk.get("breakIndex")
        if idx is None:
            continue
        scope = str(brk.get("bosType") or brk.get("chochType") or "").lower()
        direction = "up" if brk["bias"] == "BULLISH" else "down"
        events.append((int(idx), f"{brk['type']} {direction} ({scope}) at {sp(pct(float(brk['level']), price))}"))
    for sw in ind.get("swing", []):
        if sw.get("index") is not None:
            events.append((int(sw["index"]), f"{sw.get('type', '')} swing {'high' if sw.get('isHigh') else 'low'} "
                                             f"{sp(pct(float(sw['price']), price))}"))
    events.sort(key=lambda e: e[0])
    if events:
        lines.append("- Last events (oldest first): " + "; ".join(f"{_ago(last - k)} {text}" for k, text in events[-8:]))
    if st.range_low and st.range_high:
        pos = (price - st.range_low) / (st.range_high - st.range_low) * 100
        lines.append(f"- Dealing range (last swing low -> high): {sp(pct(st.range_low, price))} -> "
                     f"{sp(pct(st.range_high, price))}, price at {pos:.0f}% "
                     f"({'premium' if pos > 50 else 'discount'})")
    times = window["Time"].to_numpy()
    if st.sweeps:
        sweeps = []
        for lv in st.sweeps[-4:]:
            bars = last - int(np.searchsorted(times, np.datetime64(pd.Timestamp(lv.touch_time, unit="s"))))
            sweeps.append(f"{lv.kind} {sp(pct(lv.price, price))} swept {_ago(bars)}")
        lines.append("- Liquidity sweeps (last 40 candles): " + "; ".join(sweeps))
    else:
        lines.append("- Liquidity sweeps (last 40 candles): none")

    def near(side: str) -> list[str]:
        found = []
        for lv in st.levels:
            mid = (lv.low + lv.high) / 2
            if side == "above" and lv.low <= price:
                continue
            if side == "below" and lv.high >= price:
                continue
            found.append((abs(mid - price), lv))
        found.sort(key=lambda x: x[0])
        out = []
        for _, lv in found[:4]:
            where = sp(pct(lv.low, price)) if lv.low == lv.high else f"{sp(pct(lv.low, price))}..{sp(pct(lv.high, price))}"
            out.append(f"{lv.title} {where}")
        return out

    inside = [lv.title for lv in st.levels if lv.low <= price <= lv.high and lv.low < lv.high]
    lines.append("- Above: " + (", ".join(near("above")) or "nothing open"))
    lines.append("- Below: " + (", ".join(near("below")) or "nothing open"))
    if inside:
        lines.append("- Price is inside: " + ", ".join(inside[:3]))
    feats = {"swing_trend": {"up": 1, "down": -1}.get(trend.get("swing"), 0),
             "internal_trend": {"up": 1, "down": -1}.get(trend.get("internal"), 0),
             "sweeps_high": sum(lv.is_high for lv in st.sweeps), "sweeps_low": sum(not lv.is_high for lv in st.sweeps)}
    return lines, feats


def candle_rows(frame: pd.DataFrame, i: int, tf: str, rows: int, price: float) -> list[str]:
    """Last candles as integers (few tokens): high, low, close in 0.01% (0.001% for tiny candles) from
    the current price, volume in % of its 20-candle average. Open is left out (= previous close)."""
    part = frame.iloc[max(0, i - rows + 1):i + 1]
    small = float(frame["atr"].iloc[i]) / price * 100 < 0.15
    scale, unit = (1000, "0.001%") if small else (100, "0.01%")
    lines = [f"## CANDLES {tf} (oldest first; candles ago: high low close in units of {unit} from the current "
             f"price, v = volume in % of its 20-candle average)"]
    n = len(part)

    def q(value: float) -> int:
        return int(round(pct(value, price) * scale))
    for k, (_, c) in enumerate(part.iterrows()):
        vol = float(c["vol_ratio"]) * 100 if np.isfinite(float(c["vol_ratio"])) else 0.0
        lines.append(f"{n - 1 - k}: {q(c['High'])} {q(c['Low'])} {q(c['Close'])} v{int(round(vol))}")
    return lines


def day_block(market: Market, decision: pd.Timestamp, price: float) -> tuple[list[str], dict]:
    from src.live.core.llm_structure import day_context

    m1 = market.m1
    start = int(np.searchsorted(m1["Time"].to_numpy(), np.datetime64(decision - pd.Timedelta(minutes=DAY_MINUTES))))
    end = int(np.searchsorted(market.m1_close_time, np.datetime64(decision), side="right"))
    ctx, _levels = day_context(m1.iloc[start:end], decision)
    lines = [f"## DAY (UTC {decision:%A %H:%M}, session {ctx.session})"]
    if ctx.high_24h is not None:
        pos = (price - ctx.low_24h) / (ctx.high_24h - ctx.low_24h) * 100 if ctx.high_24h > ctx.low_24h else 50.0
        lines.append(f"- Last 24h: change {sp(ctx.change_24h_pct)}, high {sp(pct(ctx.high_24h, price))}, "
                     f"low {sp(pct(ctx.low_24h, price))}, price at {pos:.0f}% of the range")
    if ctx.prev_high is not None:
        lines.append(f"- Previous UTC day: high {sp(pct(ctx.prev_high, price))}, low {sp(pct(ctx.prev_low, price))}")
    return lines, {"chg_24h": ctx.change_24h_pct, "hour": decision.hour, "weekday": decision.weekday()}


def build_text(market: Market, anchor: Anchor, price: float, params: dict, horizon_bars: int,
               show_symbol: bool, structure: bool) -> tuple[str, dict]:
    """(text, features) of one anchor. Raises ValueError when the history is too short."""
    sig = market.frame(anchor.tf)
    i = closed_index(sig, anchor.decision)
    if i < WARMUP_SIGNAL:
        raise ValueError(f"only {i + 1} {anchor.tf} candles before it (need {WARMUP_SIGNAL})")
    since_close = pct(price, float(sig["Close"].iloc[i]))
    minutes = timeframe_to_minutes(anchor.tf) * horizon_bars
    coin = anchor.symbol if show_symbol else "a crypto coin"
    lines = [
        f"# Chart of {coin} on {anchor.tf} at {anchor.decision:%Y-%m-%d %H:%M} UTC. "
        f"Every price is written as % from the current price.",
        f"Question: will price be higher or lower {horizon_bars} {anchor.tf} candles from now "
        f"({minutes // 60}h{minutes % 60:02d}m), and by how much?",
        "",
        "## KEY FACTS",
    ]
    feats: dict[str, Any] = {}
    sig_lines, sig_feats = indicator_block(sig, i, anchor.tf, price)
    feats.update({f"s_{k}": v for k, v in sig_feats.items()})
    lines.append(f"- Since the last {anchor.tf} close: {sp(since_close)}")
    lines.append(f"- ATR14 ({anchor.tf}) = {sig_feats['atr_pct']:.3f}% of price: an average candle moves that much")

    htf_lines: list[str] = []
    for tf, long in HTF_LONG_EMA.items():
        if timeframe_to_minutes(tf) <= timeframe_to_minutes(anchor.tf):
            continue
        frame = market.frame(tf)
        j = closed_index(frame, anchor.decision)
        if j < long + 10:
            raise ValueError(f"only {j + 1} {tf} candles before it (need {long + 10})")
        block, hf = indicator_block(frame, j, tf, price, long)
        htf_lines += block
        feats.update({f"{tf}_{k}": v for k, v in hf.items()})
        stack = "up" if hf["ema20_pct"] > 0 and hf["ema50_pct"] > 0 else \
            "down" if hf["ema20_pct"] < 0 and hf["ema50_pct"] < 0 else "mixed"
        lines.append(f"- {tf}: price vs EMA20/50 {stack}, RSI {num(hf['rsi'])}, 12-candle change {sp(hf['ret_12'])}")

    struct_lines: list[str] = []
    if structure:
        try:
            struct_lines, st_feats = structure_block(sig, i, anchor.tf, price, params)
            feats.update(st_feats)
        except Exception as exc:  # the text stays complete without it
            raise ValueError(f"structure unavailable: {exc}") from exc
        if struct_lines:
            lines.append(f"- {anchor.tf} SMC " + struct_lines[1][2:].lower())
    day_lines, day_feats = day_block(market, anchor.decision, price)
    feats.update(day_feats)

    lines += ["", "## INDICATORS", *sig_lines, *htf_lines]
    if struct_lines:
        lines += ["", *struct_lines]
    if "1h" in market.frames and timeframe_to_minutes(anchor.tf) < 60:
        lines += ["", *candle_rows(market.frame("1h"), closed_index(market.frame("1h"), anchor.decision), "1h",
                                   HTF_ROWS, price)]
    lines += ["", *candle_rows(sig, i, anchor.tf, SIGNAL_ROWS, price)]
    lines += ["", *day_lines]  # last: a text cut at the end loses the least here
    clean = {k: (round(float(v), 5) if v is not None and np.isfinite(float(v)) else None) for k, v in feats.items()}
    return "\n".join(lines), clean


# ==================== image ====================

class ChartImage:
    """Candle chart PNG: price + EMA 20/50/200 + Bollinger, volume, RSI 14, MACD histogram."""

    def __init__(self, bars: int = 120, size: int = 512):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.plt = plt
        self.bars = bars
        self.size = size

    def render(self, frame: pd.DataFrame, i: int) -> bytes:
        plt = self.plt
        part = frame.iloc[max(0, i - self.bars + 1):i + 1].reset_index(drop=True)
        x = np.arange(len(part))
        o, h, l, c = (part[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
        up = c >= o
        dpi = 100
        fig = plt.figure(figsize=(self.size / dpi, self.size / dpi), dpi=dpi)
        grid = fig.add_gridspec(4, 1, height_ratios=[5, 1, 1.3, 1.3], hspace=0.04,
                                left=0.01, right=0.99, top=0.99, bottom=0.01)
        ax, axv, axr, axm = (fig.add_subplot(grid[k]) for k in range(4))
        green, red = "#26a69a", "#ef5350"
        colors = np.where(up, green, red)
        ax.vlines(x, l, h, color=colors, linewidth=0.8)
        ax.bar(x, np.maximum(np.abs(c - o), (h.max() - l.min()) * 0.001), bottom=np.minimum(o, c),
               color=colors, width=0.7, linewidth=0)
        ax.fill_between(x, part["bb_low"], part["bb_up"], color="#90a4ae", alpha=0.15, linewidth=0)
        for col, color in (("ema20", "#ff9800"), ("ema50", "#2962ff"), ("ema200", "#7b1fa2")):
            ax.plot(x, part[col], color=color, linewidth=1.0)
        axv.bar(x, part["Volume"], color=colors, width=0.7, linewidth=0)
        axr.plot(x, part["rsi"], color="#7e57c2", linewidth=1.0)
        for level in (30, 50, 70):
            axr.axhline(level, color="#b0bec5", linewidth=0.6, linestyle="--" if level != 50 else ":")
        axr.set_ylim(0, 100)
        hist = part["macd_hist"].to_numpy(float)
        axm.bar(x, hist, color=np.where(hist >= 0, green, red), width=0.7, linewidth=0)
        axm.axhline(0, color="#b0bec5", linewidth=0.6)
        for a in (ax, axv, axr, axm):
            a.set_xlim(-1, len(part))
            a.set_xticks([])
            a.set_yticks([])
            for spine in a.spines.values():
                spine.set_color("#cfd8dc")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
        plt.close(fig)
        return buf.getvalue()


# ==================== anchors ====================

def trade_anchors(trades_sources: list[str], symbol: str, entry_minutes: int) -> list[Anchor]:
    from src.evaluate.llm_backtest.trades import load_trades

    seen: dict[str, Anchor] = {}
    for t in load_trades(trades_sources):
        if t.symbol != symbol:
            continue
        a = Anchor(symbol, t.signal_tf, t.entry_time + pd.Timedelta(minutes=entry_minutes), "trade",
                   side=t.side, entry_time=t.entry_time)
        seen.setdefault(a.id, a)  # a LONG and a SHORT at the same minute and TF: one chart
    return sorted(seen.values(), key=lambda a: a.decision)


def grid_anchors(market: Market, symbol: str, tf: str, every: int) -> list[Anchor]:
    frame = market.frame(tf)
    closes = frame["CloseTime"].iloc[::max(1, every)]
    return [Anchor(symbol, tf, pd.Timestamp(t), "grid") for t in closes]


# ==================== build ====================

def _rel(split: str, label: str, tid: str, ext: str) -> str:
    from src.evaluate.llm_dataset import _rel as rel

    return rel(split, label, tid, ext)


def build_symbol(args: argparse.Namespace, symbol: str, out: Path, params: dict, images: Optional[ChartImage]) -> str:
    from src.evaluate.llm_dataset import _assign_splits, _place_files, _rows

    out.mkdir(parents=True, exist_ok=True)
    settings = {"text_format": TEXT_FORMAT, "anchors": args.anchors, "tf": args.tf if args.anchors == "grid" else None,
                "every": args.every if args.anchors == "grid" else None, "horizons": args.horizons,
                "show_symbol": args.show_symbol, "structure": not args.no_structure}
    build_file = out / "build.json"
    if build_file.exists() and not args.rebuild:
        old = json.loads(build_file.read_text(encoding="utf-8"))
        changed = [k for k, v in settings.items() if old.get(k) != v]
        if changed:
            raise SystemExit(f"{out} was built with other settings ({', '.join(f'{k}: {old.get(k)} -> {settings[k]}' for k in changed)}). "
                             "Use another --out or add --rebuild.")
    build_file.write_text(json.dumps(settings, indent=2), encoding="utf-8")

    jsonl = out / "dataset.jsonl"
    if args.rebuild and jsonl.exists():
        jsonl.unlink()
    existing = _rows(jsonl)

    t0 = time.time()
    market = Market(args.data, symbol)
    if args.anchors == "trades":
        anchors = trade_anchors(args.trades, symbol, args.entry_minutes)
    else:
        anchors = grid_anchors(market, symbol, args.tf, args.every)
    main_h = args.horizons[0]
    # anchors without the history the text needs, or without the result yet, cannot be rows
    usable = [a for a in anchors if market.first + pd.Timedelta(minutes=history_minutes(a.tf)) <= a.decision
              and a.decision + pd.Timedelta(minutes=max(args.horizons) * timeframe_to_minutes(a.tf)) <= market.last_close]
    if len(usable) < len(anchors):
        print(f"{len(anchors) - len(usable)} anchors dropped: too close to the start (history) or the end (result)")
    anchors = usable
    todo = [a for a in anchors if a.id not in existing]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{symbol}: {len(market.m1)} 1m candles {market.first:%Y-%m-%d} -> {market.last_close:%Y-%m-%d}, "
          f"{len(anchors)} anchors ({args.anchors}), {len(existing)} already built, {len(todo)} to build "
          f"(load {time.time() - t0:.1f}s)")

    added = skipped = 0
    reasons: Counter = Counter()
    started = time.time()
    for count, a in enumerate(todo, 1):
        try:
            price = market.price_at(a.decision)
            if price is None:
                raise ValueError("no 1m candle right before the decision")
            tf_min = timeframe_to_minutes(a.tf)
            targets = {}
            for h in args.horizons:
                fut = market.future(a.decision, h * tf_min)
                if fut is None:
                    raise ValueError("data ends before the horizon")
                targets[f"h{h}"] = {"bars": h, "minutes": h * tf_min, "ret_pct": round(pct(fut["close"], price), 4),
                                    "max_up_pct": round(pct(fut["high"], price), 4),
                                    "max_down_pct": round(pct(fut["low"], price), 4)}
            text, feats = build_text(market, a, price, params, main_h, args.show_symbol, not args.no_structure)
        except ValueError as exc:
            skipped += 1
            reasons[re.sub(r"\b\d+\b", "N", str(exc).split(":")[0])] += 1
            if args.verbose:
                print(f"skip {a.id}: {exc}")
            continue
        main = targets[f"h{main_h}"]
        label = "up" if main["ret_pct"] > 0 else "down"
        old_split = ""
        text_rel = _rel(old_split, label, a.id, "txt") if args.mode in ("text", "both") else ""
        image_rel = ""
        if images is not None:
            sig = market.frame(a.tf)
            png = images.render(sig, closed_index(sig, a.decision))
            image_rel = _rel(old_split, label, a.id, "png")
            (out / image_rel).parent.mkdir(parents=True, exist_ok=True)
            (out / image_rel).write_bytes(png)
        if text_rel:
            (out / text_rel).parent.mkdir(parents=True, exist_ok=True)
            (out / text_rel).write_text(text, encoding="utf-8")
        record = {
            "id": a.id, "task": "direction", "text": text, "text_file": text_rel, "image": image_rel,
            "label": label, "label_id": 1 if label == "up" else 0,
            "ret_pct": main["ret_pct"], "max_up_pct": main["max_up_pct"], "max_down_pct": main["max_down_pct"],
            "atr_pct": feats.get("s_atr_pct"), "horizon_bars": main_h, "horizon_minutes": main["minutes"],
            "targets": targets, "symbol": symbol, "timeframe": a.tf, "anchor": a.kind, "side": a.side,
            "entry_time": f"{a.decision:%Y-%m-%d %H:%M:%S}",
            "exit_time": f"{a.decision + pd.Timedelta(minutes=main['minutes']):%Y-%m-%d %H:%M:%S}",
            "price": price, "split": "", "text_format": TEXT_FORMAT, "chars": len(text), "features": feats,
        }
        with jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        added += 1
        if count % args.log_every == 0 or count == len(todo):
            rate = (time.time() - started) / count
            print(f"[{count}/{len(todo)}] {a.id} {label} {main['ret_pct']:+.2f}%  ({rate:.2f}s/anchor, "
                  f"ETA {rate * (len(todo) - count) / 60:.1f} min)")

    rows = list(_rows(jsonl).values())
    _assign_splits(rows, 0.0, args.test, split=args.test > 0)
    moved = _place_files(out, rows)
    if moved:
        print(f"{moved} files moved into the split / label folders")
    with jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    with (out / "dataset.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    ups = sum(r["label_id"] for r in rows)
    rets = np.array([r["ret_pct"] for r in rows]) if rows else np.zeros(1)
    why = (" (" + ", ".join(f"{k}: {n}" for k, n in reasons.most_common(4)) + ")") if reasons else ""
    line = (f"{symbol}: {added} added, {skipped} skipped{why}. Dataset: {len(rows)} rows, {ups} up / {len(rows) - ups} down, "
            f"ret median {np.median(rets):+.3f}%, |ret| median {np.median(np.abs(rets)):.3f}% -> {out}")
    print(line)
    return line


# ==================== menu / main ====================

def menu(args: argparse.Namespace) -> None:
    from src.evaluate.llm_dataset import choose_many
    from src.evaluate.verify_orders import choose

    folder = Path(args.data)
    symbols = sorted(p.name.removesuffix("_1m.csv") for p in folder.glob("*_1m.csv"))
    if not symbols:
        raise SystemExit(f"No *_1m.csv in {folder}")
    print(f"backtest.yaml\n  data   (candles): {args.data}\n  output (trades) : {args.trades[0]}")
    args.symbol = choose_many("Coins (one dataset folder per coin):", [(s, "") for s in symbols])
    args.anchors = choose("Decision moments:", [
        ("trades", f"entry time of every backtest trade in {args.trades[0]} (its signal timeframe)"),
        ("grid", "every N candles of one timeframe (more rows, no strategy bias)"),
    ])
    if args.anchors == "grid":
        args.tf = choose("Chart timeframe:", [(tf, "") for tf in ("15m", "5m", "1h", "3m")])
        raw = input(f"Every how many {args.tf} candles? (Enter = {args.every}): ").strip()
        args.every = int(raw) if raw.isdigit() and int(raw) > 0 else args.every
    raw = input(f"Horizons in signal candles, first = label (Enter = {','.join(map(str, args.horizons))}): ").strip()
    if raw:
        args.horizons = [int(x) for x in raw.replace(" ", "").split(",") if x.isdigit() and int(x) > 0] or args.horizons
    args.mode = choose("Input:", [("text", "price + indicators as text (Laya)"),
                                  ("image", "candle chart PNG with indicators"), ("both", "text + PNG")])
    raw = input("Limit per coin (Enter = all, e.g. 20 for a quick test): ").strip()
    args.limit = int(raw) if raw.isdigit() else 0


def main() -> None:
    from src.config import load_backtest_config, load_strategy_config
    from src.indicators.export import strategy_indicator_params

    bt = load_backtest_config(include_strategy=False)
    parser = argparse.ArgumentParser(description="Price path + indicators before a moment -> up / down and % after it")
    parser.add_argument("--data", default=None, help="1m candle folder (default: backtest.yaml folder_path)")
    parser.add_argument("--trades", nargs="+", default=None, help="trades CSV files / folders for --anchors trades "
                                                                  "(default: backtest.yaml output_folder)")
    parser.add_argument("--symbol", nargs="+", default=["BTCUSDT"], help="one or more coins (one folder each); 'all'")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output folder; each coin in <out>/<coin>")
    parser.add_argument("--anchors", choices=("trades", "grid"), default="trades")
    parser.add_argument("--tf", default="15m", help="chart timeframe of --anchors grid")
    parser.add_argument("--every", type=int, default=4, help="--anchors grid: one anchor every N candles of --tf")
    parser.add_argument("--horizons", default="12,4,24",
                        help="signal candles after the decision; the first is the label (default 12,4,24)")
    parser.add_argument("--mode", choices=("text", "image", "both"), default="text")
    parser.add_argument("--image-bars", type=int, default=120, help="candles in the chart image")
    parser.add_argument("--image-size", type=int, default=512, help="image width = height in pixels")
    parser.add_argument("--show-symbol", action="store_true", help="write the coin name in the text")
    parser.add_argument("--no-structure", action="store_true", help="no SMC STRUCTURE section (faster)")
    parser.add_argument("--test", type=float, default=0.0, help="newest share in test/ (default 0 = no split folders; "
                                                                "laya_finetune splits by time itself)")
    parser.add_argument("--limit", type=int, default=0, help="only the first N new anchors per coin (0 = all)")
    parser.add_argument("--rebuild", action="store_true", help="start the coin folders from scratch")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--verbose", action="store_true", help="print every skipped anchor")
    args = parser.parse_args()
    args.trades = args.trades or [str(bt.get("output_folder") or "output")]
    args.data = args.data or str(bt.get("folder_path") or "data")
    args.horizons = [int(x) for x in str(args.horizons).split(",") if x.strip()]
    if len(sys.argv) == 1 and sys.stdin.isatty():
        menu(args)
    if [s.lower() for s in args.symbol] == ["all"]:
        args.symbol = sorted(p.name.removesuffix("_1m.csv") for p in Path(args.data).glob("*_1m.csv"))

    strategy_config = load_strategy_config()
    args.entry_minutes = int(timeframe_to_minutes(str(strategy_config.get("entry_timeframe") or "1m"), default=1) or 1)
    params = strategy_indicator_params(strategy_config)
    images = ChartImage(args.image_bars, args.image_size) if args.mode in ("image", "both") else None
    summary = []
    for symbol in dict.fromkeys(s.upper() for s in args.symbol):
        base = symbol.removesuffix("USDT").removesuffix("USDC").lower()
        summary.append(build_symbol(args, symbol, Path(args.out) / base, params, images))
    if len(summary) > 1:
        print("\n===== summary =====")
        print("\n".join(summary))


if __name__ == "__main__":
    main()
