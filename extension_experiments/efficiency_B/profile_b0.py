"""B0 diagnostic profiler for the *unchanged* VR-EPAGCL training computation.

Place this file at:
    ~/projects/EPAGCL/extension_experiments/efficiency_B/profile_b0.py

Run from the EPAGCL project root in the existing Python environment:
    python -u extension_experiments/efficiency_B/profile_b0.py

It never edits original EPAGCL source, model checkpoints, or formal_A results.
No test labels or linear evaluation are used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT / "extension_experiments"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENT_ROOT))

from Load_data import get_dataset  # noqa: E402
from Augmentation import add_edge_weights, drop_edge_weights  # noqa: E402
from Model import MyModel  # noqa: E402
from Utils import get_optim  # noqa: E402
from run import make_augmented_views, set_seed  # noqa: E402
from vr_regularization import variance_floor_loss  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B0: VR-EPAGCL GPU/CPU training-step profiler")
    parser.add_argument("--dataset", default="Coauthor-Phy")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timed", type=int, default=20)
    parser.add_argument("--profile-active", type=int, default=5)
    parser.add_argument("--skip-profiler", action="store_true")
    parser.add_argument("--export-trace", action="store_true", help="Optional; trace can be large")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    if min(args.warmup, args.timed, args.profile_active) < 1:
        parser.error("warmup, timed and profile-active must be positive")
    if not (PROJECT_ROOT / "Model.py").is_file() or not (EXPERIMENT_ROOT / "run.py").is_file():
        parser.error("Place this script under EPAGCL/extension_experiments/efficiency_B/")
    return args


def device_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def source_hashes() -> dict[str, str]:
    files = [
        "Model.py", "Augmentation.py", "Load_data.py", "Utils.py",
        "extension_experiments/run.py", "extension_experiments/vr_regularization.py",
        "extension_experiments/efficiency_B/profile_b0.py",
    ]
    result = {}
    for file in files:
        path = PROJECT_ROOT / file
        if path.exists():
            result[file] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main() -> None:
    args = arguments()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use the existing EPAGCL GPU environment")
    if device.type != "cuda":
        raise RuntimeError("This experiment is intended for the same CUDA GPU as formal_A")

    output = Path(args.out) if args.out else (
        EXPERIMENT_ROOT / "efficiency_B" / "outputs" /
        (datetime.now().strftime("%Y%m%d_%H%M%S") + "_B0_seed_" + str(args.seed))
    )
    output.mkdir(parents=True, exist_ok=False)

    dataset_path = Path(args.dataset_path) if args.dataset_path else (
        PROJECT_ROOT.parent / "datasets" / args.dataset
    )
    dataset_path = dataset_path.expanduser().resolve()
    print("B0 profiler (unchanged VR loss; no evaluation/test labels)", flush=True)
    print("Output:", output, flush=True)
    print("Dataset:", dataset_path, flush=True)
    print("GPU:", torch.cuda.get_device_name(device), flush=True)
    print("PyTorch:", torch.__version__, flush=True)
    print("Seed:", args.seed, flush=True)
    print("Profiler instrumentation is diagnostic; its timings are NOT used as speedup benchmarks.")

    # Replicate the actual run.py one-time data and augmentation preparation.
    set_seed(args.seed)
    device_sync(device)
    t0 = perf_counter()
    data = get_dataset(str(dataset_path), args.dataset).to(device)
    add_weights, adding_edge = add_edge_weights(data.edge_index, 0.1)
    drop_weights = drop_edge_weights(data.edge_index)
    device_sync(device)
    prep_s = perf_counter() - t0

    train_args = SimpleNamespace(
        optim="Adam", lr=0.01, weight_decay=0.001,
        edge_drop_rate_1=0.1, edge_drop_rate_2=0.4,
        feat_drop_rate_1=0.0, feat_drop_rate_2=0.0,
    )
    set_seed(args.seed)
    model = MyModel(data.num_features, 256, 256, 0.5).to(device)
    optimizer = get_optim(model.parameters(), train_args)

    def step(trace: bool = False) -> None:
        mark = torch.profiler.record_function if trace else (lambda name: nullcontext())
        model.train()
        optimizer.zero_grad()
        with mark("B0/1_graph_augmentation"):
            (x1, edge1), (x2, edge2) = make_augmented_views(
                data, add_weights, adding_edge, drop_weights, train_args
            )
        with mark("B0/2_GCN_projection_view_1"):
            out1, z1 = model(x1, edge1)
        with mark("B0/3_GCN_projection_view_2"):
            out2, z2 = model(x2, edge2)
        with mark("B0/4_contrastive_loss"):
            contrastive = model.loss(z1, z2, True)
        with mark("B0/5_variance_penalty"):
            vr = variance_floor_loss(out1, out2, gamma=0.002)
            loss = contrastive + 10.0 * vr
        with mark("B0/6_backward"):
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            loss.backward()
        with mark("B0/7_optimizer_update"):
            optimizer.step()
        with mark("B0/8_scalar_logging"):
            # The formal runner writes these three scalars every epoch.
            _ = (float(contrastive.item()), float(vr.item()), float(loss.item()))

    # Wall clock: avoid profiler overhead and any extra synchronization inside phases.
    for _ in range(args.warmup):
        step(trace=False)
    device_sync(device)
    torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for _ in range(args.timed):
        device_sync(device)
        start = perf_counter()
        step(trace=False)
        device_sync(device)
        timings.append(perf_counter() - start)
    memory_mib = torch.cuda.max_memory_allocated(device) / 1048576
    sorted_time = sorted(timings)
    n = len(sorted_time)
    metrics = {
        "benchmark": "B0 VR-EPAGCL unchanged loss",
        "seed": args.seed,
        "dataset": args.dataset,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": str(torch.__version__),
        "warmup_steps": args.warmup,
        "timed_steps": args.timed,
        "preprocessing_seconds": prep_s,
        "mean_seconds_per_step": statistics.mean(timings),
        "median_seconds_per_step": statistics.median(timings),
        "p90_seconds_per_step": sorted_time[min(n - 1, (9 * n + 9) // 10 - 1)],
        "min_seconds_per_step": min(timings),
        "max_seconds_per_step": max(timings),
        "peak_allocated_memory_mib": memory_mib,
        "timing_scope": "One complete main-model VR training step; no linear evaluation, preprocessing excluded",
        "source_sha256": source_hashes(),
    }
    (output / "timing.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print("\n=== UNPROFILED WALL-CLOCK (use for performance comparisons) ===")
    for k in ["preprocessing_seconds", "mean_seconds_per_step", "median_seconds_per_step",
              "p90_seconds_per_step", "peak_allocated_memory_mib"]:
        print(f"{k}: {metrics[k]:.4f}", flush=True)

    # Diagnostic-only per-stage wall-clock timings. Synchronizing after each
    # phase changes execution characteristics; these are NOT speedup timings.
    stage_samples = {}
    def phase(label, callback):
        device_sync(device)
        start = perf_counter()
        value = callback()
        device_sync(device)
        stage_samples.setdefault(label, []).append(perf_counter() - start)
        return value

    for _ in range(3):
        model.train()
        optimizer.zero_grad()
        (x1, edge1), (x2, edge2) = phase(
            "1_graph_augmentation",
            lambda: make_augmented_views(data, add_weights, adding_edge, drop_weights, train_args),
        )
        out1, z1 = phase("2_GCN_projection_view_1", lambda: model(x1, edge1))
        out2, z2 = phase("3_GCN_projection_view_2", lambda: model(x2, edge2))
        contrastive = phase("4_contrastive_loss", lambda: model.loss(z1, z2, True))
        penalty = phase("5_variance_penalty", lambda: variance_floor_loss(out1, out2, gamma=0.002))
        total = contrastive + 10.0 * penalty
        phase("6_backward", total.backward)
        phase("7_optimizer_update", optimizer.step)
        phase("8_scalar_logging", lambda: (contrastive.item(), penalty.item(), total.item()))

    stage_means = {name: statistics.mean(times) for name, times in stage_samples.items()}
    diagnostic_sum = sum(stage_means.values())
    with (output / "stage_diagnostic.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(("phase", "mean_seconds_3_samples", "pct_of_diagnostic_phases"))
        for label, seconds in stage_means.items():
            writer.writerow((label, f"{seconds:.6f}", f"{100.0 * seconds / diagnostic_sum:.2f}"))
    print("\n=== SYNCHRONIZED PHASE DIAGNOSTICS (NOT speedup timings) ===")
    for label, seconds in stage_means.items():
        print(f"{label:30s} {seconds:.4f} s, diagnostic share {100 * seconds / diagnostic_sum:.1f}%")

    if not args.skip_profiler:
        try:
            activity = [torch.profiler.ProfilerActivity.CPU]
            if torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities():
                activity.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(
                activities=activity,
                schedule=torch.profiler.schedule(wait=1, warmup=1,
                                                 active=args.profile_active, repeat=1),
                record_shapes=False, profile_memory=False, with_stack=False,
            ) as prof:
                for _ in range(args.profile_active + 2):
                    step(trace=True)
                    prof.step()
            sort_key = "self_cuda_time_total" if len(activity) > 1 else "self_cpu_time_total"
            table = prof.key_averages().table(sort_by=sort_key, row_limit=40)
            (output / "profiler_top_ops.txt").write_text(table + "\n", encoding="utf-8")
            print("\n=== PROFILED TOP OPERATIONS (diagnostic ONLY) ===\n" + table)
            if args.export_trace:
                prof.export_chrome_trace(str(output / "profiler_trace.json"))
                print("Chrome trace exported; this file may be large.")
        except Exception as exc:
            (output / "profiler_error.txt").write_text(repr(exc) + "\n", encoding="utf-8")
            print("Profiler was unavailable; UNPROFILED timing.json is still valid:", repr(exc))

    print("\nCompleted B0 diagnostics. Output:", output, flush=True)
    print("No formal training checkpoints or splits were modified.", flush=True)


if __name__ == "__main__":
    main()
