#!/usr/bin/env bash
# Colab: install deps and check the GPU. Run from the repo folder (/content/laya_finetune).
#   bash scripts/colab_setup.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
  echo "No GPU: Runtime -> Change runtime type -> T4 GPU"; exit 1; }

# torch is preinstalled on Colab (with CUDA); keep it, install the rest.
pip install -q -r requirements.txt

python - <<'EOF'
import torch, transformers, laya
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| transformers", transformers.__version__, "| laya", laya.__version__)
assert torch.cuda.is_available(), "CUDA not available"
EOF
