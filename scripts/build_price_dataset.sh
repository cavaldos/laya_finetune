#!/usr/bin/env bash
# Build the price-direction datasets (Mac, inside the Bot_Trade project) into laya_finetune/data_price/.
# Each row: price path + indicators up to a moment (text) -> price up / down and % after the horizon.
#
#   bash laya_finetune/scripts/build_price_dataset.sh                   # every coin of the backtest data
#   bash laya_finetune/scripts/build_price_dataset.sh BTCUSDT ETHUSDT   # only these
#   LIMIT=50 bash laya_finetune/scripts/build_price_dataset.sh BTCUSDT  # quick test
#   EVERY=4 bash laya_finetune/scripts/build_price_dataset.sh           # one row per hour (more rows, overlapping)
#   ANCHORS=trades bash laya_finetune/scripts/build_price_dataset.sh    # at the backtest trade entries instead
#
# Default: a 15m chart every 12 candles (3h) with the label after 12 candles (3h): the results of two
# rows never overlap. ~1100 rows per coin on the 200k (139 days) data, ~0.01-0.05 s per row.
# Re-running continues; dataset.jsonl.gz is what gets committed / read on Kaggle and Colab.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"          # laya_finetune/
BOT_TRADE="${BOT_TRADE:-$(cd "$HERE/.." && pwd)}"                 # the Bot_Trade project
BT="${BT:-res/backtest/200k}"                                       # backtest folder (data/ [+ output/])
PY="${PY:-$BOT_TRADE/.venv/bin/python}"
OUT="${OUT:-$HERE/data_price}"
ANCHORS="${ANCHORS:-grid}"
TF="${TF:-15m}"
EVERY="${EVERY:-12}"
HORIZONS="${HORIZONS:-12,4,24}"                                     # first = label; others stored (train.py --horizon h4)
MODE="${MODE:-text}"                                                # text | image | both
LIMIT="${LIMIT:-0}"

cd "$BOT_TRADE"
[ -d "$BT/data" ] || { echo "missing $BOT_TRADE/$BT/data"; exit 1; }
if [ "$#" -gt 0 ]; then SYMBOLS=("$@"); else SYMBOLS=(all); fi

"$PY" -m src.evaluate.price_dataset \
  --data "$BT/data" --trades "$BT/output" --symbol "${SYMBOLS[@]}" \
  --anchors "$ANCHORS" --tf "$TF" --every "$EVERY" --horizons "$HORIZONS" \
  --mode "$MODE" --limit "$LIMIT" --out "$OUT" --log-every 200

for f in "$OUT"/*/dataset.jsonl; do
  gzip -kf "$f"                                                     # -> dataset.jsonl.gz (kept in git)
done
echo
echo "rows per coin:"
wc -l "$OUT"/*/dataset.jsonl
du -ch "$OUT"/*/dataset.jsonl.gz | tail -1
