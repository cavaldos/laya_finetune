#!/usr/bin/env bash
# Flow check on the Mac (MPS): 24 train / 12 val / 12 test trades, 1 epoch, 512 tokens, ~2 min.
# The numbers mean nothing; it only proves data -> train -> resume state -> test report works.
#
#   bash scripts/smoke_mac.sh                 # extra args go to train.py, e.g. --anonymize
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$HERE/../.venv/bin/python}"            # the Bot_Trade project venv
[ -x "$PY" ] || PY=python3

"$PY" -c "import laya" 2>/dev/null || "$PY" -m pip install -r "$HERE/requirements.txt"
"$PY" "$HERE/train.py" --profile smoke --name "smoke_$(date +%Y%m%d-%H%M%S)" "$@"
