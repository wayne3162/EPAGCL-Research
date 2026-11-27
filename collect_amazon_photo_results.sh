#!/usr/bin/env bash

cd "$HOME/projects/EPAGCL" || exit 1

for name in RandomAdd AddBoth EPAGCL DropOnly RandomDrop AddOnly; do
    echo ""
    echo "================================================"
    echo "Amazon-Photo - $name"
    echo "================================================"

    dir="trained_models/Q3_${name}/Amazon-Photo"
    found=0

    for result in "$dir"/completed_*/result.txt; do
        [ -f "$result" ] || continue

        folder="$(dirname "$result")"
        args="$folder/args.txt"

        # 僅收集正式實驗：1000 Epoch × 5 次
        if [ ! -f "$args" ] ||
           ! grep -qx 'epoch: 1000' "$args" ||
           ! grep -qx 'repeat: 5' "$args"; then
            continue
        fi

        found=1

        echo ""
        echo "[正式實驗路徑]"
        echo "$folder"

        echo ""
        echo "[訓練結果]"
        cat "$result"

        echo ""
        echo "[實驗參數]"
        grep -E \
'^(epoch|repeat|lr|weight_decay|temperature|edge_add_rate|edge_drop_rate_1|edge_drop_rate_2|add_edge_random|random_drop_edge|add_single|not_add_edge|not_drop_edge|feat_drop_rate_1|feat_drop_rate_2):' \
            "$args"
    done

    if [ "$found" -eq 0 ]; then
        echo "尚未找到 1000 Epoch × 5 次的正式結果"
    fi
done
