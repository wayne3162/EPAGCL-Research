#!/usr/bin/env bash

set -euo pipefail

cd ~/projects/EPAGCL

LOG_DIR="extension_experiments/formal_A/logs"
mkdir -p "$LOG_DIR"

# 剩餘四個正式實驗種子
for seed in 20261124 20261125 20261126 20261127; do

    for method in baseline vr; do

        echo ""
        echo "========================================"
        echo "Method: $method"
        echo "Seed:   $seed"
        echo "Epoch:  1000"
        echo "Start:  $(date)"
        echo "========================================"

        vr_args=()

        if [[ "$method" == "vr" ]]; then
            vr_args=(--vr-lambda 10 --vr-gamma 0.002)
        fi

        python -u extension_experiments/run.py \
            --method "$method" \
            "${vr_args[@]}" \
            --dataset Coauthor-Phy \
            --device cuda:0 \
            --epoch 1000 \
            --repeat 1 \
            --seed "$seed" \
            --eval 50 \
            --eval-epoch 3000 \
            --loss-log 1 \
            --save-model-every 100 \
            --no-shadow-stop \
            2>&1 | tee "$LOG_DIR/${method}_${seed}.log"

        echo ""
        echo "Completed: $method | seed=$seed"
        echo "End: $(date)"

    done
done

echo ""
echo "========================================"
echo "All remaining formal experiments finished."
echo "========================================"
