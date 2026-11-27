#!/usr/bin/env bash
set -Eeuo pipefail

cd "$HOME/projects/EPAGCL"

PYTHON="$PWD/.venv/bin/python"
LOGDIR="$PWD/logs"

mkdir -p "$LOGDIR"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: 找不到 .venv/bin/python"
    exit 1
fi

trap 'status=$?; echo "[ERROR] 實驗佇列停止，exit code: $status" >&2' ERR

# 保存程式版本
git rev-parse HEAD > "$LOGDIR/WikiCS_Q3_base_commit.txt"
git diff > "$LOGDIR/WikiCS_Q3_source_changes.patch"

# 所有 Q3 實驗共用參數
COMMON=(
    --dataset WikiCS
    --batch_compute
    --feat_drop_rate_1 0
    --feat_drop_rate_2 0
    --lr 0.001
    --weight_decay 0.0001
    --temperature 0.3
    --edge_add_rate 0.1
    --edge_drop_rate_1 0.2
    --edge_drop_rate_2 0.3
)

echo "========================================"
echo "WikiCS Q3 自動實驗開始"
echo "時間：$(date)"
echo "========================================"

# ========================================
# 執行正式實驗
# ========================================

run_formal() {
    local name="$1"
    shift

    local done_file="$LOGDIR/WikiCS_${name}_formal.done"
    local old_done_file="$LOGDIR/WikiCS_${name}.done"
    local log_file="$LOGDIR/WikiCS_${name}_formal_$(date +%Y%m%d_%H%M%S).log"

    if [[ -f "$done_file" || -f "$old_done_file" ]]; then
        echo "[SKIP] $name 已有完成標記，跳過"
        return 0
    fi

    echo ""
    echo "========================================"
    echo "[START] WikiCS - $name"
    echo "開始時間：$(date)"
    echo "========================================"

    "$PYTHON" -u main.py \
        "${COMMON[@]}" \
        "$@" \
        --epoch 3000 \
        --repeat 5 \
        --eval 50 \
        --eval_epoch 3000 \
        --path "./trained_models/Q3_${name}/" \
        2>&1 | tee "$log_file"

    date -Is > "$done_file"

    echo "[DONE] WikiCS - $name"
    echo "完成時間：$(date)"
}

# ========================================
# 新增邊短測試
# ========================================

run_test() {
    local name="$1"
    shift

    local done_file="$LOGDIR/WikiCS_${name}_test.done"
    local log_file="$LOGDIR/WikiCS_${name}_test.log"

    if [[ -f "$done_file" ]]; then
        echo "[SKIP] $name 短測試已通過"
        return 0
    fi

    echo ""
    echo "========================================"
    echo "[TEST] WikiCS - $name (10 Epoch)"
    echo "開始時間：$(date)"
    echo "========================================"

    "$PYTHON" -u main.py \
        "${COMMON[@]}" \
        "$@" \
        --epoch 10 \
        --repeat 1 \
        --eval 10 \
        --eval_epoch 30 \
        --path "./trained_models/Q3_${name}_Test/" \
        2>&1 | tee "$log_file"

    date -Is > "$done_file"

    echo "[PASS] $name 短測試成功"
}

# ========================================
# 1. Random Drop
# ========================================

run_formal RandomDrop \
    --not_add_edge \
    --random_drop_edge

# ========================================
# 2. Add to Both Views
# ========================================

run_test AddBoth

run_formal AddBoth

# ========================================
# 3. EPAGCL
# ========================================

run_formal EPAGCL \
    --add_single

# ========================================
# 4. Add Only
# ========================================

run_formal AddOnly \
    --add_single \
    --not_drop_edge

# ========================================
# 5. Random Add
# ========================================

# 隨機新增邊使用不同函式，另外短測試
run_test RandomAdd \
    --add_edge_random \
    --random_drop_edge

run_formal RandomAdd \
    --add_edge_random \
    --random_drop_edge

echo ""
echo "========================================"
echo "WikiCS Q3 全部剩餘實驗已完成"
echo "結束時間：$(date)"
echo "========================================"
