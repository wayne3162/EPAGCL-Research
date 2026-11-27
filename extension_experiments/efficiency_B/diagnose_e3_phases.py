"""E3 non-destructive, per-phase, CUDA-synchronized diagnostic.

Copy alongside train_e3.py, model_b1.py, and model_b2.py in
  EPAGCL/extension_experiments/efficiency_B/

Example (from EPAGCL root):
  python -u extension_experiments/efficiency_B/diagnose_e3_phases.py \
      --variant b2x8 --seed 20261128 --steps 20 --device cuda:0

IMPORTANT: Every stage synchronizes CUDA for localization. These timings are
instrumented diagnostics and MUST NOT be used to report B0/B2 speedups.
This script does NOT run a linear probe, touch test labels, modify source files,
load/save checkpoints, or modify existing experiment outputs.
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


def get_args():
    p = argparse.ArgumentParser(description="E3 CUDA stage localization (not a benchmark)")
    p.add_argument("--variant", choices=("b0", "b2x8"), default="b2x8")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20261128)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--dataset", default="Coauthor-Phy")
    p.add_argument("--dataset-path", default=None)
    p.add_argument("--output-root", default=None)
    args = p.parse_args()
    if args.steps <= 0:
        p.error("--steps must be positive")
    if not args.device.startswith("cuda:"):
        p.error("CUDA device required for comparison with previous measurements")
    if args.dataset != "Coauthor-Phy":
        p.error("Use Coauthor-Phy to match previous measurements")
    for k, v in FIXED.items():
        setattr(args, k, v)
    return args


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    args = get_args()
    device = torch.device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    gpu_name = torch.cuda.get_device_name(device)
    if gpu_name != "NVIDIA GeForce RTX 2070":
        raise RuntimeError(f"Expected original RTX 2070, observed {gpu_name!r}")

    out_root = Path(args.output_root).expanduser().resolve() if args.output_root else B_ROOT / "outputs"
    out_root.mkdir(parents=True, exist_ok=True)
    output = out_root / (datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                         + f"_E3_stage_diagnostic_{args.variant}_seed_{args.seed}")
    output.mkdir(parents=True, exist_ok=False)
    dataset_path = Path(args.dataset_path) if args.dataset_path else REPO_ROOT.parent / "datasets" / args.dataset
    dataset_path = dataset_path.expanduser().resolve()

    print("E3 instrumented stage diagnostic (not formal timing)", flush=True)
    print(f"variant={args.variant} seed={args.seed} steps={args.steps} gpu={gpu_name}", flush=True)
    print("Output:", output, flush=True)
    print("Preprocessing dataset once, not included in per-step timing.", flush=True)

    set_seed(args.seed)
    sync(device)
    prep_start = perf_counter()
    data = get_dataset(str(dataset_path), args.dataset).to(device)
    add_weights, adding_edge = add_edge_weights(data.edge_index, args.edge_add_rate)
    drop_weights = drop_edge_weights(data.edge_index)
    sync(device)
    prep_seconds = perf_counter() - prep_start

    # Exactly the same reset and model setup as train_e3.py.
    set_seed(args.seed)
    model = model_for(args.variant, data.num_features, args, device)
    optim = get_optim(model.parameters(), args)
    torch.cuda.reset_peak_memory_stats(device)

    labels = (
        "setup", "graph_augmentation", "GCN_view_1", "GCN_view_2",
        "contrastive_loss", "variance_penalty", "backward", "optimizer", "scalar_logging",
    )
    rows = []
    for epoch in range(1, args.steps + 1):
        phase_times = {}

        def measure(name, fn):
            sync(device)
            t0 = perf_counter()
            value = fn()
            sync(device)
            phase_times[name] = perf_counter() - t0
            return value

        def setup():
            model.train()
            optim.zero_grad()

        measure("setup", setup)
        (x1, edge1), (x2, edge2) = measure(
            "graph_augmentation",
            lambda: make_augmented_views(data, add_weights, adding_edge, drop_weights, args),
        )
        out1, z1 = measure("GCN_view_1", lambda: model(x1, edge1))
        out2, z2 = measure("GCN_view_2", lambda: model(x2, edge2))
        contrastive = measure("contrastive_loss", lambda: model.loss(z1, z2, args.batch_compute))
        vr = measure("variance_penalty", lambda: variance_floor_loss(out1, out2, gamma=args.vr_gamma))
        total_loss = contrastive + args.vr_lambda * vr
        if not bool(torch.isfinite(total_loss).item()):
            raise FloatingPointError(f"Non-finite loss at step {epoch}")
        measure("backward", total_loss.backward)
        measure("optimizer", optim.step)
        measure("scalar_logging", lambda: (contrastive.item(), vr.item(), total_loss.item()))
        total = sum(phase_times.values())
        row = {"epoch": epoch, **phase_times, "sum_diagnostic_seconds": total}
        rows.append(row)
        print(f"Epoch {epoch:2d} | sum={total:.4f}s "
              f"| augment={phase_times['graph_augmentation']:.4f}s "
              f"| forwards={phase_times['GCN_view_1'] + phase_times['GCN_view_2']:.4f}s "
              f"| loss={phase_times['contrastive_loss']:.4f}s "
              f"| backward={phase_times['backward']:.4f}s", flush=True)

    csv_file = output / "stage_times.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=("epoch", *labels, "sum_diagnostic_seconds"))
        w.writeheader()
        w.writerows(rows)

    summary = {
        "variant": args.variant,
        "seed": args.seed,
        "steps": args.steps,
        "dataset": args.dataset,
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "preprocessing_seconds_excluded": prep_seconds,
        "stage_medians_seconds": {
            label: statistics.median(row[label] for row in rows) for label in labels
        },
        "peak_allocated_memory_mib": torch.cuda.max_memory_allocated(device) / 1048576,
        "source_sha256": {
            str(p.relative_to(REPO_ROOT)): digest(p) for p in (
                REPO_ROOT / "Model.py", REPO_ROOT / "Augmentation.py",
                REPO_ROOT / "extension_experiments" / "run.py",
                REPO_ROOT / "extension_experiments" / "efficiency_B" / "model_b1.py",
                REPO_ROOT / "extension_experiments" / "efficiency_B" / "model_b2.py",
                Path(__file__).resolve(),
            ) if p.exists()
        },
        "note": "Extra CUDA synchronize calls BETWEEN stages; for bottleneck localization only, not B0/B2 speed comparison.",
        "has_linear_evaluation": False,
        "uses_test_labels": False,
    }
    (output / "stage_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\nDone. Stage CSV:", csv_file, flush=True)
    print("Summary:", output / "stage_summary.json", flush=True)
    print("Reminder: these stage timings are NOT formal speedup estimates.", flush=True)


if __name__ == "__main__":
    main()
