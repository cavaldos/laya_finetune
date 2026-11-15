#!/usr/bin/env bash
# Pack data/ for Colab: only dataset.jsonl + system_prompt.txt of each coin (the .txt copies,
# dataset.csv and images are not needed). ~5 MB per coin after gzip.
#
#   bash scripts/pack_data.sh            # -> laya_data.tar.gz in the laya_finetune folder
#
# Then upload laya_data.tar.gz to Google Drive, e.g. MyDrive/laya/laya_data.tar.gz
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
ls data/*/dataset.jsonl >/dev/null 2>&1 || { echo "no data/*/dataset.jsonl: run scripts/build_dataset.sh first"; exit 1; }

files=$(ls data/*/dataset.jsonl data/*/system_prompt.txt 2>/dev/null)
COPYFILE_DISABLE=1 tar czf laya_data.tar.gz $files     # COPYFILE_DISABLE: no macOS ._ files
echo "laya_data.tar.gz: $(du -h laya_data.tar.gz | cut -f1)"
wc -l data/*/dataset.jsonl
