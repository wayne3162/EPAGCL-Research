"""Paired B0/B1 GPU runtime benchmark (no profiler or linear evaluation).

Requires a passing verify_b1.py equivalence.json from the *same* code and GPU.
Uses the B0 step body and defaults from the user-provided profile_b0.py.
Does not modify original project files, checkpoints, or formal_A outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT / "extension_experiments"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Load_data import get_dataset  # noqa: E402
from Augmentation import add_edge_weights, drop_edge_weights  # noqa: E402
from Model import MyModel  # noqa: E402
from model_b1 import MyModelB1  # noqa: E402
from Utils import get_optim  # noqa: E402
from run import make_augmented_views, set_seed  # noqa: E402
from vr_regularization import variance_floor_loss  # noqa: E402
from verify_b1 import SOURCE_FILES  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="B0/B1 paired performance benchmark")
    parser.add_argument("--verification-json", required=True, help="Path to passing B1 equivalence.json")
    parser.add_argument("--dataset", default="Coauthor-Phy")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timed", type=int, default=20)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if min(args.rounds, args.warmup, args.timed) <= 0:
        parser.error("rounds, warmup, timed must be positive")
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        parser.error("Run this comparison on the same CUDA GPU as B0 and the verification")
    return args


def file_hashes():
    names = list(SOURCE_FILES) + [
        "Augmentation.py", "Load_data.py", "Utils.py",
        "extension_experiments/run.py",
        "extension_experiments/efficiency_B/benchmark_b01.py",
    ]
    return {p: hashlib.sha256((PROJECT_ROOT / p).read_bytes()).hexdigest() for p in names}


def check_verification(path, device):
    record = json.loads(Path(path).read_text())
    if not record.get("passed"):
        raise RuntimeError("B1 equivalence verification has NOT passed; do not benchmark")
    if record.get("small_only") or 34493 not in [r.get("nodes") for r in record.get("checks", []) if r.get("kind") == "embedding_and_input_gradient"]:
        raise RuntimeError("Required 34493-node gradient check is missing")
    if record.get("batch_size") != 256:
        raise RuntimeError("Equivalence check must use the target batch size 256")
    if record.get("gpu_name") != torch.cuda.get_device_name(device) or record.get("device") != str(device):
        raise RuntimeError("GPU or CUDA device differs from successful verification")
    for name, expected_hash in record["source_sha256"].items():
        current = hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
        if current != expected_hash:
            raise RuntimeError(f"Source changed after verification: {name}")
    return record


def synchronize(device):
    torch.cuda.synchronize(device)


def nth_p90(arr):
    vals = sorted(arr)
    return vals[min(len(vals) - 1, (9 * len(vals) + 9) // 10 - 1)]


def run_trial(variant, seed, device, data, add_weights, adding_edge, drop_weights, train_args, warmup, timed):
    # Each member of a paired round gets the same model initialization seed and
    # augmentation/batching RNG start; no other model is resident simultaneously.
    set_seed(seed)
    cls = MyModel if variant == "B0" else MyModelB1
    model = cls(data.num_features, 256, 256, 0.5).to(device)
    optimizer = get_optim(model.parameters(), train_args)

    def step():
        model.train()
        optimizer.zero_grad()
        (x1, edge1), (x2, edge2) = make_augmented_views(
            data, add_weights, adding_edge, drop_weights, train_args
        )
        out1, z1 = model(x1, edge1)
        out2, z2 = model(x2, edge2)
        contrastive = model.loss(z1, z2, True)
        vr = variance_floor_loss(out1, out2, gamma=0.002)
        loss = contrastive + 10.0 * vr
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss")
        loss.backward()
        optimizer.step()
        # Exactly the three scalar reads present in original B0 step.
        _ = (float(contrastive.item()), float(vr.item()), float(loss.item()))

    try:
        for _ in range(warmup):
            step()
        synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        times = []
        for _ in range(timed):
            synchronize(device)
            start = perf_counter()
            step()
            synchronize(device)
            times.append(perf_counter() - start)
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        return {
            "variant": variant,
            "seed": seed,
            "status": "ok",
            "warmup_steps": warmup,
            "timed_steps": timed,
            "mean_seconds_per_step": statistics.mean(times),
            "median_seconds_per_step": statistics.median(times),
            "p90_seconds_per_step": nth_p90(times),
            "peak_allocated_memory_mib": peak_mem,
            "individual_times_seconds": times,
        }
    finally:
        del optimizer, model
        synchronize(device)
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    device = torch.device(args.device)
    verification = check_verification(args.verification_json, device)
    output = Path(args.out) if args.out else (
        EXPERIMENT_ROOT / "efficiency_B" / "outputs" /
        (datetime.now().strftime("%Y%m%d_%H%M%S") + "_B01_compare")
    )
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "comparison.json"
    gpu = torch.cuda.get_device_name(device)
    print("B0/B1 paired benchmark on:", gpu, flush=True)
    print("Output:", output, flush=True)
    dataset_path = Path(args.dataset_path) if args.dataset_path else (
        PROJECT_ROOT.parent / "datasets" / args.dataset
    )
    dataset_path = dataset_path.expanduser().resolve()

    set_seed(args.seed)
    synchronize(device)
    start = perf_counter()
    data = get_dataset(str(dataset_path), args.dataset).to(device)
    add_weights, adding_edge = add_edge_weights(data.edge_index, 0.1)
    drop_weights = drop_edge_weights(data.edge_index)
    synchronize(device)
    preprocess_seconds = perf_counter() - start
    print(f"Preprocess (excluded from step timings): {preprocess_seconds:.3f}s", flush=True)
    train_args = SimpleNamespace(
        optim="Adam", lr=0.01, weight_decay=0.001,
        edge_drop_rate_1=0.1, edge_drop_rate_2=0.4,
        feat_drop_rate_1=0.0, feat_drop_rate_2=0.0,
    )
    report = {
        "benchmark": "paired B0/B1 unprofiled main-model VR train step",
        "device": str(device), "gpu_name": gpu, "torch_version": torch.__version__,
        "dataset": args.dataset, "dataset_path": str(dataset_path),
        "base_seed": args.seed, "rounds": args.rounds,
        "warmup_steps_each_trial": args.warmup,
        "timed_steps_each_trial": args.timed,
        "preprocessing_seconds_excluded": preprocess_seconds,
        "evaluation": "none; no test labels; no linear probe",
        "source_sha256": file_hashes(),
        "equivalence_verification": str(Path(args.verification_json).resolve()),
        "verification_gpu": verification["gpu_name"],
        "trials": [], "completed": False,
    }
    try:
        for r in range(args.rounds):
            order = ("B0", "B1") if r % 2 == 0 else ("B1", "B0")
            seed = args.seed + 1000 * r
            for variant in order:
                print(f"Round {r + 1}/{args.rounds}: {variant}; seed={seed}", flush=True)
                result = run_trial(variant, seed, device, data, add_weights, adding_edge,
                                   drop_weights, train_args, args.warmup, args.timed)
                result["round"] = r + 1
                report["trials"].append(result)
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                print(f"  median={result['median_seconds_per_step']:.4f}s, "
                      f"mean={result['mean_seconds_per_step']:.4f}s, "
                      f"peak={result['peak_allocated_memory_mib']:.1f}MiB", flush=True)

        all_b0 = [s for t in report["trials"] if t["variant"] == "B0" for s in t["individual_times_seconds"]]
        all_b1 = [s for t in report["trials"] if t["variant"] == "B1" for s in t["individual_times_seconds"]]
        ratio = statistics.median(all_b0) / statistics.median(all_b1)
        paired = []
        for r in range(1, args.rounds + 1):
            b0 = next(t for t in report["trials"] if t["round"] == r and t["variant"] == "B0")
            b1 = next(t for t in report["trials"] if t["round"] == r and t["variant"] == "B1")
            paired.append({"round": r, "B0_median_s": b0["median_seconds_per_step"],
                           "B1_median_s": b1["median_seconds_per_step"],
                           "paired_speedup_ratio": b0["median_seconds_per_step"] / b1["median_seconds_per_step"],
                           "B0_peak_MiB": b0["peak_allocated_memory_mib"],
                           "B1_peak_MiB": b1["peak_allocated_memory_mib"]})
        report["summary"] = {
            "B0_pooled_median_seconds": statistics.median(all_b0),
            "B1_pooled_median_seconds": statistics.median(all_b1),
            "B0_pooled_mean_seconds": statistics.mean(all_b0),
            "B1_pooled_mean_seconds": statistics.mean(all_b1),
            "B0_pooled_p90_seconds": nth_p90(all_b0),
            "B1_pooled_p90_seconds": nth_p90(all_b1),
            "speedup_ratio_B0_median_over_B1_median": ratio,
            "reduction_percent": 100 * (1 - 1 / ratio),
            "rounds_with_faster_B1_median": sum(x["paired_speedup_ratio"] > 1 for x in paired),
            "paired_rounds": paired,
            "B0_peak_allocated_memory_mib_max_over_rounds": max(t["peak_allocated_memory_mib"] for t in report["trials"] if t["variant"] == "B0"),
            "B1_peak_allocated_memory_mib_max_over_rounds": max(t["peak_allocated_memory_mib"] for t in report["trials"] if t["variant"] == "B1"),
            "preliminary_10_percent_median_reduction_met": 100 * (1 - 1 / ratio) >= 10,
        }
        report["completed"] = True
    except Exception as exc:
        report["error"] = repr(exc)
        print("Benchmark failed (partial results preserved):", repr(exc), flush=True)
    finally:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print("Results:", report_path, flush=True)
        if "summary" in report:
            print("B0 pooled median:", f"{report['summary']['B0_pooled_median_seconds']:.4f}s")
            print("B1 pooled median:", f"{report['summary']['B1_pooled_median_seconds']:.4f}s")
            print("B1 step-time reduction:", f"{report['summary']['reduction_percent']:.2f}%")
    if not report["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
