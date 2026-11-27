#!/usr/bin/env bash
set -Eeuo pipefail

# Coauthor-Physics Q3, all six ablations, sequential execution.
# Run from the EPAGCL project directory with:
#   bash run_coauthor_phy_q3_all.sh

cd "$HOME/projects/EPAGCL"
mkdir -p logs

PYTHON="$PWD/.venv/bin/python"
LOGDIR="$PWD/logs"
DATASET="Coauthor-Phy"

if [[ ! -x "$PYTHON" ]]; then
    echo "[ERROR] Project Python not found: $PYTHON" >&2
    exit 1
fi

# Prevent two copies of this queue from running at the same time.
exec 9>"$LOGDIR/Coauthor-Phy_queue.lock"
if ! flock -n 9; then
    echo "[ERROR] Coauthor-Physics queue is already running." >&2
    exit 1
fi

trap 'rc=$?; echo "[STOP] Coauthor-Physics queue failed (exit=$rc) at $(date -Is)" >&2' ERR

# Record the exact code version and changes before the queue starts.
git rev-parse HEAD > "$LOGDIR/Coauthor-Phy_base_commit.txt"
git diff > "$LOGDIR/Coauthor-Phy_code_changes.patch"
cp "$0" "$LOGDIR/Coauthor-Phy_queue_script_snapshot.sh"

COMMON=(
    --dataset "$DATASET"
    --batch_compute
    --feat_drop_rate_1 0
    --feat_drop_rate_2 0
    --lr 0.01
    --weight_decay 0.001
    --temperature 0.5
    --edge_add_rate 0.1
    --edge_drop_rate_1 0.1
    --edge_drop_rate_2 0.4
)

run_test() {
    local name="$1"
    shift
    local done_file="$LOGDIR/Coauthor-Phy_${name}_test.done"
    local log_file="$LOGDIR/Coauthor-Phy_${name}_test_$(date +%Y%m%d_%H%M%S).log"

    if [[ -f "$done_file" ]]; then
        echo "[SKIP TEST] $name has already passed."
        return 0
    fi

    echo "[TEST] $name | start=$(date -Is)"
    "$PYTHON" -u main.py \
        "${COMMON[@]}" "$@" \
        --epoch 10 --repeat 1 --eval 10 --eval_epoch 30 \
        --path "./trained_models/Q3_${name}_Test/" \
        2>&1 | tee "$log_file"

    date -Is > "$done_file"
    echo "[TEST PASS] $name | finish=$(date -Is)"
}

run_formal() {
    local name="$1"
    shift
    local done_file="$LOGDIR/Coauthor-Phy_${name}_formal.done"
    local log_file="$LOGDIR/Coauthor-Phy_${name}_formal_$(date +%Y%m%d_%H%M%S).log"

    if [[ -f "$done_file" ]]; then
        echo "[SKIP FORMAL] $name is already marked completed."
        return 0
    fi

    echo "[FORMAL] $name | start=$(date -Is)"
    "$PYTHON" -u main.py \
        "${COMMON[@]}" "$@" \
        --epoch 1000 --repeat 5 --eval 50 --eval_epoch 3000 \
        --path "./trained_models/Q3_${name}/" \
        2>&1 | tee "$log_file"

    # Do not mark success unless the program wrote a completed result summary.
    local result_root="trained_models/Q3_${name}/$DATASET"
    local found=0
    local args_file result_file
    for args_file in "$result_root"/completed_*/args.txt; do
        [[ -f "$args_file" ]] || continue
        result_file="$(dirname "$args_file")/result.txt"
        if grep -qx 'epoch: 1000' "$args_file" \
            && grep -qx 'repeat: 5' "$args_file" \
            && [[ -s "$result_file" ]] \
            && grep -q '^mean:' "$result_file" \
            && grep -q '^mean:.*std:' "$result_file"; then
            found=1
            echo "[RESULT] $result_file"
            tail -n 2 "$result_file"
        fi
    done
    if [[ "$found" -ne 1 ]]; then
        echo "[ERROR] No valid 1000-epoch x 5-run result found for $name" >&2
        return 1
    fi

    date -Is > "$done_file"
    echo "[FORMAL DONE] $name | finish=$(date -Is)"
}

printf '\n[START] Coauthor-Physics Q3 queue | %s\n' "$(date -Is)"

# 1. Random Drop, first as requested in the previous WikiCS workflow.
run_test RandomDrop --not_add_edge --random_drop_edge
run_formal RandomDrop --not_add_edge --random_drop_edge

# 2. Drop Only.
run_test DropOnly --not_add_edge
run_formal DropOnly --not_add_edge

# 3. Adaptive adding to both views. This tests the >30,000-node path.
run_test AddBoth
run_formal AddBoth

# 4. EPAGCL: add to one view; adaptive dropping in both.
run_formal EPAGCL --add_single
# 5. Add Only: adaptive adding to one view, no edge dropping.
run_formal AddOnly --add_single --not_drop_edge

# 6. Random Add: random addition and random dropping.
run_test RandomAdd --add_edge_random --random_drop_edge
run_formal RandomAdd --add_edge_random --random_drop_edge

printf '\n[ALL DONE] Coauthor-Physics Q3 queue | %s\n' "$(date -Is)"
