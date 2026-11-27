"""E3 odd/even timing diagnostic: dynamic augmentation versus fixed graph views.

Place in EPAGCL/extension_experiments/efficiency_B/ and run from project root:
    python -u extension_experiments/efficiency_B/diagnose_e3_parity_control.py \
        --variant b2x8 --seed 20261128 --steps 16 --device cuda:0

This is NOT a formal speed benchmark or model-quality evaluation. Fixed-view
training deliberately changes the data augmentation schedule, and is used only
as a control to localize an observed odd/even step-time anomaly.
No test labels, linear evaluation, splits, or checkpoints are read/modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch

from train_e3 import (
    B_ROOT,
    REPO_ROOT,
    FIXED,
    model_for,
    set_seed,
    sync,
    get_dataset,
    get_optim,
    add_edge_weights,
    drop_edge_weights,
    make_augmented_views,
    variance_floor_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E3 dynamic/fixed-view parity diagnostic")
    parser.add_argument("--variant", choices=("b0", "b2x8"), default="b2x8")
    parser.add_argument("--seed", type=int, default=20261128)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset", default="Coauthor-Phy")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.steps < 8:
        parser.error("--steps must be at least 8 for parity comparison")
    if args.device != "cuda:0":
        parser.error("Use cuda:0 to match the prior RTX 2070 experiments")
    if args.dataset != "Coauthor-Phy":
        parser.error("Use Coauthor-Phy to match the prior experiments")
    for name, value in FIXED.items():
        setattr(args, name, value)
    return args


def digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def median_metric(rows: list[dict], metric: str, parity: int) -> float:
    # Exclude first two steps because CUDA init can dominate them.
    values = [float(r[metric]) for r in rows if r["epoch"] >= 3 and r["epoch"] % 2 == parity]
    return statistics.median(values)


def run_condition(label: str, fixed: bool, args, data, add_weights,
                  adding_edge, drop_weights, device: torch.device) -> list[dict]:
    # An independent model and optimizer for each condition. No previous state reused.
    set_seed(args.seed)
    model = model_for(args.variant, data.num_features, args, device)
    optimizer = get_optim(model.parameters(), args)

    cached_views = None
    if fixed:
        cached_views = make_augmented_views(data, add_weights, adding_edge, drop_weights, args)
        sync(device)

    print(f"\n=== {label}: {'one fixed pair of views' if fixed else 'new views every step'} ===", flush=True)
    results = []
    for epoch in range(1, args.steps + 1):
        phase = {}

        def measure(name, action):
            sync(device)
            start = perf_counter()
            value = action()
            sync(device)
            phase[name] = perf_counter() - start
            return value

        model.train()
        optimizer.zero_grad()
        if fixed:
            (x1, edge1), (x2, edge2) = cached_views
            phase["augmentation_s"] = 0.0  # cached views, not regenerated
        else:
            (x1, edge1), (x2, edge2) = measure(
                "augmentation_s",
                lambda: make_augmented_views(data, add_weights, adding_edge, drop_weights, args),
            )

        out1, z1 = measure("forward_1_s", lambda: model(x1, edge1))
        out2, z2 = measure("forward_2_s", lambda: model(x2, edge2))
        contrastive = measure("loss_s", lambda: model.loss(z1, z2, args.batch_compute))
        vr = measure("vr_s", lambda: variance_floor_loss(out1, out2, gamma=args.vr_gamma))
        total = contrastive + args.vr_lambda * vr
        if not torch.isfinite(total).item():
            raise FloatingPointError(f"{label} step={epoch} has a non-finite loss")
        measure("backward_s", total.backward)
        measure("optimizer_s", optimizer.step)
        # No linear evaluation and no access to data.y anywhere.
        phase_total = sum(phase.values())
        row = {
            "condition": label,
            "epoch": epoch,
            "edge1_count": int(edge1.size(1)),
            "edge2_count": int(edge2.size(1)),
            "forward_s": phase["forward_1_s"] + phase["forward_2_s"],
            "sum_s": phase_total,
            **phase,
        }
        results.append(row)
        print(
            f"{label:9s} step={epoch:02d} total={phase_total:7.4f}s "
            f"aug={phase['augmentation_s']:6.4f}s "
            f"fwd={row['forward_s']:6.4f}s bwd={phase['backward_s']:6.4f}s "
            f"edges=({row['edge1_count']},{row['edge2_count']})",
            flush=True,
        )
    return results


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(args.device)
    gpu = torch.cuda.get_device_name(device)
    if gpu != "NVIDIA GeForce RTX 2070":
        raise RuntimeError(f"Use the original RTX 2070 for this comparison, got {gpu!r}")

    out = Path(args.out).expanduser().resolve() if args.out else (
        B_ROOT / "outputs" /
        f"{datetime.now():%Y%m%d_%H%M%S_%f}_E3_parity_{args.variant}_seed_{args.seed}"
    )
    out.mkdir(parents=True, exist_ok=False)
    dataset_path = Path(args.dataset_path).expanduser().resolve() if args.dataset_path else (
        REPO_ROOT.parent / "datasets" / args.dataset
    ).resolve()

    print("E3 dynamic-vs-fixed graph timing CONTROL (not a speed benchmark)")
    print(f"variant={args.variant}; seed={args.seed}; GPU={gpu}; steps/condition={args.steps}")
    print(f"dataset={dataset_path}")
    print(f"output={out}", flush=True)

    set_seed(args.seed)
    data = get_dataset(str(dataset_path), args.dataset).to(device)
    add_weights, adding_edge = add_edge_weights(data.edge_index, args.edge_add_rate)
    drop_weights = drop_edge_weights(data.edge_index)
    sync(device)

    # Dynamic -> Fixed -> Dynamic guards against a one-off time/order artifact.
    all_rows = []
    by_condition = {}
    for label, fixed in (("dynamic_1", False), ("fixed", True), ("dynamic_2", False)):
        rows = run_condition(label, fixed, args, data, add_weights,
                             adding_edge, drop_weights, device)
        all_rows.extend(rows)
        by_condition[label] = rows

    fields = (
        "condition", "epoch", "edge1_count", "edge2_count", "augmentation_s",
        "forward_1_s", "forward_2_s", "forward_s", "loss_s", "vr_s",
        "backward_s", "optimizer_s", "sum_s",
    )
    with (out / "parity_steps.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {}
    print("\n=== Median even/odd ratio (after excluding steps 1-2) ===")
    for label, rows in by_condition.items():
        summary[label] = {}
        for metric in ("forward_s", "backward_s", "sum_s", "augmentation_s"):
            odd = median_metric(rows, metric, 1)
            even = median_metric(rows, metric, 0)
            ratio = even / odd if odd > 0 else None
            summary[label][metric] = {"odd_median_s": odd, "even_median_s": even,
                                      "even_over_odd": ratio}
        e1 = [r["edge1_count"] for r in rows]
        e2 = [r["edge2_count"] for r in rows]
        summary[label]["edge1_min_max"] = [min(e1), max(e1)]
        summary[label]["edge2_min_max"] = [min(e2), max(e2)]
        print(
            f"{label:9s} fwd ratio={summary[label]['forward_s']['even_over_odd']:.2f}x; "
            f"bwd ratio={summary[label]['backward_s']['even_over_odd']:.2f}x; "
            f"total ratio={summary[label]['sum_s']['even_over_odd']:.2f}x; "
            f"edge ranges=({min(e1)}-{max(e1)}, {min(e2)}-{max(e2)})",
            flush=True,
        )
    meta = {
        "purpose": "Localize odd-even time alternation, NOT a speed benchmark",
        "variant": args.variant,
        "seed": args.seed,
        "steps_per_condition": args.steps,
        "gpu": gpu,
        "torch_version": torch.__version__,
        "dataset": args.dataset,
        "conditions": summary,
        "source_sha256": {
            str(p.relative_to(REPO_ROOT)): digest(p)
            for p in (
                REPO_ROOT / "Model.py",
                REPO_ROOT / "Augmentation.py",
                REPO_ROOT / "extension_experiments" / "run.py",
                REPO_ROOT / "extension_experiments" / "efficiency_B" / "train_e3.py",
                REPO_ROOT / "extension_experiments" / "efficiency_B" / "model_b1.py",
                REPO_ROOT / "extension_experiments" / "efficiency_B" / "model_b2.py",
                Path(__file__).resolve(),
            ) if p.is_file()
        },
        "warning": "Fixed-view training deliberately changes augmentation. "
                   "The condition is diagnostic only and must not be used as a formal quality or speed result. "
                   "Every phase is CUDA-synchronized and changes execution overhead.",
    }
    (out / "parity_summary.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nCSV: {out / 'parity_steps.csv'}")
    print(f"Summary: {out / 'parity_summary.json'}")
    print("No labels, splits or checkpoints accessed. Training state is discarded.")


if __name__ == "__main__":
    main()
