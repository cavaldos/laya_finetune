#!/usr/bin/env bash
# Build the AI-review text datasets (Mac, inside the Bot_Trade project) into laya_finetune/data/.
#
#   bash laya_finetune/scripts/build_dataset.sh                 # every coin not built yet
#   bash laya_finetune/scripts/build_dataset.sh BTCUSDT ETHUSDT # only these coins
#   BT=res/backtest/100k bash laya_finetune/scripts/build_dataset.sh
#
# Text only (--no-image: the model never reads the chart image), no split (train.py makes one
# global time split over all coins). Re-running continues: trades already built are skipped.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"          # laya_finetune/
BOT_TRADE="${BOT_TRADE:-$(cd "$HERE/.." && pwd)}"                 # the Bot_Trade project
BT="${BT:-res/backtest/200k}"                                       # backtest folder (output/ + data/)
PY="${PY:-$BOT_TRADE/.venv/bin/python}"
OUT="$HERE/data"

cd "$BOT_TRADE"
[ -d "$BT/output" ] && [ -d "$BT/data" ] || { echo "missing $BOT_TRADE/$BT/output or /data"; exit 1; }

if [ "$#" -gt 0 ]; then
  SYMBOLS=("$@")
else
  # every coin with 1m candles in the backtest data folder, minus the ones already in data/
  SYMBOLS=()
  for f in "$BT"/data/*_1m.csv; do
    sym="$(basename "$f" _1m.csv)"
    base="$(echo "${sym%USDT}" | tr '[:upper:]' '[:lower:]')"
    if [ -f "$OUT/${base}_ai_review/dataset.csv" ]; then
      echo "skip $sym (already built: $OUT/${base}_ai_review)"
    else
      SYMBOLS+=("$sym")
    fi
  done
fi
[ "${#SYMBOLS[@]}" -gt 0 ] || { echo "nothing to build"; exit 0; }

echo "building ${SYMBOLS[*]} from $BT -> $OUT"
for sym in "${SYMBOLS[@]}"; do
  base="$(echo "${sym%USDT}" | tr '[:upper:]' '[:lower:]')"
  echo
  echo "===== $sym -> $OUT/${base}_ai_review ====="
  # one coin per call: llm_dataset writes a single coin straight into --out
  "$PY" -m src.evaluate.llm_dataset \
    --trades "$BT/output" --data "$BT/data" \
    --symbol "$sym" \
    --no-image --no-split --out "$OUT/${base}_ai_review"
done

echo
echo "done. Rows per coin:"
wc -l "$OUT"/*/dataset.jsonl
