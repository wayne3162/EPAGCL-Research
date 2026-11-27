#!/usr/bin/env bash
set -euo pipefail

# Place this script in EPAGCL/extension_experiments/reverse_drop_C/.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Run the original and reversed edge-dropping strategies with paired seeds.
exec python extension_experiments/reverse_drop_C/run.py \
  --method both \
  --dataset Cora \
  --device cuda:0 \
  --epoch 500 \
  --repeat 5 \
  --seed 20261119 \
  --eval 50 \
  --eval-epoch 3000 \
  --optim Adam \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --temperature 0.3 \
  --edge-add-rate 0.1 \
  --edge-drop-rate-1 0.2 \
  --edge-drop-rate-2 0.3 \
  --feat-drop-rate-1 0.1 \
  --feat-drop-rate-2 0.1 \
  --feat-dim 256 \
  --proj-hidden-dim 256 \
  --eval-optim Adam \
  --eval-lr 0.01 \
  --eval-weight-decay 0.0
