#!/usr/bin/env bash
# Colab T4: unpack the data from Drive, train (or resume), score the test split.
# Runs and checkpoints go to Drive, so a disconnect loses at most SAVE_EVERY minutes:
# run the same command again (same RUN_NAME) to resume.
#
#   RUN_NAME=run1 bash scripts/colab_train.sh                       # defaults below
#   RUN_NAME=anon1 bash scripts/colab_train.sh --anonymize          # extra args go to train.py
#   RUN_NAME=quick bash scripts/colab_train.sh --limit-train 2000 --epochs 1
#
# Settings (environment variables):
DRIVE="${DRIVE:-/content/drive/MyDrive/laya}"                 # your folder on Google Drive
DATA_TAR="${DATA_TAR:-$DRIVE/laya_data.tar.gz}"               # made by scripts/pack_data.sh
OUT_DIR="${OUT_DIR:-$DRIVE/runs}"                             # runs (checkpoints, reports)
RUN_NAME="${RUN_NAME:-run1}"
MAX_HOURS="${MAX_HOURS:-0}"                                   # e.g. 3.5 to stop cleanly before a session limit
SAVE_EVERY="${SAVE_EVERY:-20}"                                # minutes between resume saves
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

[ -d /content/drive/MyDrive ] || { echo "Mount Google Drive first (notebook cell: drive.mount)"; exit 1; }
mkdir -p "$OUT_DIR"

if ! ls data/*/dataset.jsonl* >/dev/null 2>&1; then          # data is in the repo; Drive tar only as a fallback
  [ -f "$DATA_TAR" ] || { echo "no data/ in the repo and no $DATA_TAR"; exit 1; }
  tar xzf "$DATA_TAR" -C "$HERE"
fi
echo "data:"; ls -la data/*/dataset.jsonl* | awk '{print $5, $9}'

python train.py --profile colab \
  --name "$RUN_NAME" --out "$OUT_DIR" \
  --models-dir /content/laya_models \
  --max-hours "$MAX_HOURS" --save-every-min "$SAVE_EVERY" \
  "$@" 2>&1 | grep --line-buffered -v -E "FutureWarning|warnings.warn|gradient_checkpointing_enable"
