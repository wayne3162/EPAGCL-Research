"""Compare B0, B1 and B2 grouped-loss variants in paired GPU training steps.

Requires passing full B1 and B2 equivalence JSON from this same GPU and
unchanged source files. No profiler, no test labels, no linear evaluation.
Uses the original profile_b0.py training step and hyperparameters.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT / "extension_experiments"
B_DIR = Path(__file__).resolve().parent
for d in (PROJECT_ROOT, EXPERIMENT_ROOT, B_DIR):
    sys.path.insert(0, str(d))

from Load_data import get_dataset  # noqa: E402
from Augmentation import add_edge_weights, drop_edge_weights  # noqa: E402
from Model import MyModel  # noqa: E402
from model_b1 import MyModelB1  # noqa: E402
from model_b2 import MyModelB2  # noqa: E402
from Utils import get_optim  # noqa: E402
from run import make_augmented_views, set_seed  # noqa: E402
from vr_regularization import variance_floor_loss  # noqa: E402

VARIANTS = ("B0", "B1", "B2x2", "B2x4", "B2x8")
HASH_FILES = (
    "Model.py", "Augmentation.py", "Load_data.py", "Utils.py",
    "extension_experiments/run.py", "extension_experiments/vr_regularization.py",
    "extension_experiments/efficiency_B/profile_b0.py",
    "extension_experiments/efficiency_B/model_b1.py",
    "extension_experiments/efficiency_B/model_b2.py",
    "extension_experiments/efficiency_B/verify_b1.py",
    "extension_experiments/efficiency_B/verify_b2.py",
    "extension_experiments/efficiency_B/benchmark_b012.py",
)


def parse_args():
    p = argparse.ArgumentParser(description="B0/B1/B2 grouped-loss paired benchmark")
    p.add_argument("--b1-verification-json", required=True)
    p.add_argument("--b2-verification-json", required=True)
    p.add_argument("--dataset", default="Coauthor-Phy")
    p.add_argument("--dataset-path", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20261123)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--timed", type=int, default=20)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if min(args.rounds, args.warmup, args.timed) < 1:
        p.error("rounds, warmup, timed must be positive")
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        p.error("Run on the CUDA GPU used for B1/B2 verification")
    return args


def validate_verification(json_path, device, required_groups=None):
    record = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if not record.get("passed") or record.get("small_only"):
        raise RuntimeError(f"Incomplete/unpassed verification: {json_path}")
    if record.get("device") != str(device) or record.get("gpu_name") != torch.cuda.get_device_name(device):
        raise RuntimeError(f"Verification GPU does not match: {json_path}")
    if record.get("batch_size") != 256:
        raise RuntimeError("Formal target batch size is 256")
    sizes = {r.get("nodes") for r in record.get("checks", [])
             if r.get("kind") == "embedding_and_input_gradient" and r.get("passed")}
    if 34493 not in sizes:
        raise RuntimeError("Missing passing 34493-node input-gradient check")
    if required_groups is not None:
        found = set(record.get("groups", []))
        if not set(required_groups).issubset(found):
            raise RuntimeError("B2 verification does not cover 2/4/8 grouping")
        for row in record.get("checks", []):
            if set(g.get("groups_per_call") for g in row.get("groups", []) if g.get("passed")) != set(required_groups):
                raise RuntimeError("B2 verification has missing/failed group checks")
    for path, old_hash in record.get("source_sha256", {}).items():
        new_hash = hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()
        if new_hash != old_hash:
            raise RuntimeError(f"Verified file changed since verification: {path}")
    return record


def synchronize(device):
    torch.cuda.synchronize(device)


def p90(arr):
    vals = sorted(arr)
    return vals[min(len(vals) - 1, (9 * len(vals) + 9) // 10 - 1)]


def model_for_variant(name):
    if name == "B0":
        return MyModel
    if name == "B1":
        return MyModelB1
    if name in ("B2x2", "B2x4", "B2x8"):
        return partial(MyModelB2, groups_per_call=int(name[-1]))
    raise ValueError(f"Unsupported variant: {name}")


def run_trial(variant, seed, device, data, add_weights, adding_edge, drop_weights,
              train_args, warmup, timed):
    set_seed(seed)
    model = None
    optimizer = None
    try:
        model = model_for_variant(variant)(data.num_features, 256, 256, 0.5).to(device)
        optimizer = get_optim(model.parameters(), train_args)

        def step():
            model.train()
            optimizer.zero_grad()
            (x1, edge1), (x2, edge2) = make_augmented_views(
                data, add_weights, adding_edge, drop_weights, train_args)
            out1, z1 = model(x1, edge1)
            out2, z2 = model(x2, edge2)
            contrastive = model.loss(z1, z2, True)
            vr = variance_floor_loss(out1, out2, gamma=0.002)
            loss = contrastive + 10.0 * vr
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            loss.backward()
            optimizer.step()
            _ = (float(contrastive.item()), float(vr.item()), float(loss.item()))

        for _ in range(warmup):
            step()
        synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        timings = []
        for _ in range(timed):
            synchronize(device)
            start = perf_counter()
            step()
            synchronize(device)
            timings.append(perf_counter() - start)
        peak = torch.cuda.max_memory_allocated(device) / 1048576
        return {"variant": variant, "seed": seed, "status": "ok",
                "warmup_steps": warmup, "timed_steps": timed,
                "individual_times_seconds": timings,
                "mean_seconds_per_step": statistics.mean(timings),
                "median_seconds_per_step": statistics.median(timings),
                "p90_seconds_per_step": p90(timings),
                "peak_allocated_memory_mib": peak}
    finally:
        del optimizer, model
        gc.collect()
        synchronize(device)
        torch.cuda.empty_cache()


def summarize(trials):
    overall = {}
    for name in VARIANTS:
        vt = [x for x in trials if x["variant"] == name and x["status"] == "ok"]
        if not vt:
            overall[name] = {"status": "no_successful_trials"}
            continue
        steps = [dt for row in vt for dt in row["individual_times_seconds"]]
        overall[name] = {
            "successful_rounds": len(vt),
            "pooled_median_seconds": statistics.median(steps),
            "pooled_mean_seconds": statistics.mean(steps),
            "pooled_p90_seconds": p90(steps),
            "peak_allocated_memory_mib_max": max(r["peak_allocated_memory_mib"] for r in vt),
            "round_medians_seconds": [r["median_seconds_per_step"] for r in vt],
        }
    paired = []
    for r in sorted({x["round"] for x in trials}):
        one = {x["variant"]: x for x in trials if x["round"] == r and x["status"] == "ok"}
        record = {"round": r, "seed": next(x["seed"] for x in trials if x["round"] == r)}
        for name in VARIANTS:
            if name in one:
                record[name + "_median_seconds"] = one[name]["median_seconds_per_step"]
        for name in VARIANTS[1:]:
            if "B0" in one and name in one:
                a = one["B0"]["median_seconds_per_step"]
                b = one[name]["median_seconds_per_step"]
                record[name + "_vs_B0_reduction_percent"] = (1.0 - b / a) * 100.0
            if "B1" in one and name in one and name.startswith("B2"):
                a = one["B1"]["median_seconds_per_step"]
                b = one[name]["median_seconds_per_step"]
                record[name + "_vs_B1_reduction_percent"] = (1.0 - b / a) * 100.0
        paired.append(record)
    for name in VARIANTS[1:]:
        for baseline in ("B0", "B1"):
            if name == baseline:
                continue
            if "pooled_median_seconds" in overall[name] and "pooled_median_seconds" in overall[baseline]:
                a = overall[baseline]["pooled_median_seconds"]
                b = overall[name]["pooled_median_seconds"]
                overall[name][f"pooled_reduction_vs_{baseline}_percent"] = (1.0 - b / a) * 100.0
    return {"overall_by_variant": overall, "paired_rounds": paired,
            "note": "Outliers retained. Profiler and linear probe excluded. Check each paired round, not just pooled medians."}


def main():
    args = parse_args()
    device = torch.device(args.device)
    validate_verification(args.b1_verification_json, device)
    validate_verification(args.b2_verification_json, device, required_groups=(2, 4, 8))

    out = Path(args.out) if args.out else (B_DIR / "outputs" /
        (datetime.now().strftime("%Y%m%d_%H%M%S") + "_B012_compare"))
    out.mkdir(parents=True, exist_ok=False)
    report_path = out / "comparison_b012.json"
    dataset_path = Path(args.dataset_path) if args.dataset_path else PROJECT_ROOT.parent / "datasets" / args.dataset
    dataset_path = dataset_path.expanduser().resolve()

    print("B0/B1/B2 paired CUDA benchmark:", torch.cuda.get_device_name(device), flush=True)
    print("Output:", out, flush=True)
    set_seed(args.seed)
    synchronize(device)
    t0 = perf_counter()
    data = get_dataset(str(dataset_path), args.dataset).to(device)
    add_weights, adding_edge = add_edge_weights(data.edge_index, 0.1)
    drop_weights = drop_edge_weights(data.edge_index)
    synchronize(device)
    preprocess_seconds = perf_counter() - t0
    print(f"Preprocess excluded from timing: {preprocess_seconds:.3f} s", flush=True)

    train_args = SimpleNamespace(optim="Adam", lr=0.01, weight_decay=0.001,
        edge_drop_rate_1=0.1, edge_drop_rate_2=0.4,
        feat_drop_rate_1=0.0, feat_drop_rate_2=0.0)
    report = {
        "study": "B0 B1 B2x2 B2x4 B2x8 main-model training-step comparison",
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": str(torch.__version__), "dataset": args.dataset,
        "dataset_path": str(dataset_path), "base_seed": args.seed,
        "rounds": args.rounds, "variants": list(VARIANTS),
        "warmup_steps_each_trial": args.warmup, "timed_steps_each_trial": args.timed,
        "preprocessing_seconds_excluded": preprocess_seconds,
        "evaluation": "none; no labels, no linear probe; no profiler",
        "source_sha256": {p: hashlib.sha256((PROJECT_ROOT / p).read_bytes()).hexdigest() for p in HASH_FILES},
        "b1_verification": str(Path(args.b1_verification_json).resolve()),
        "b2_verification": str(Path(args.b2_verification_json).resolve()),
        "trials": [], "completed": False,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    try:
        for r in range(args.rounds):
            order = VARIANTS if r % 2 == 0 else tuple(reversed(VARIANTS))
            seed = args.seed + 1000 * r
            for name in order:
                print(f"Round {r+1}/{args.rounds}: {name}, seed={seed}", flush=True)
                try:
                    trial = run_trial(name, seed, device, data, add_weights, adding_edge,
                                      drop_weights, train_args, args.warmup, args.timed)
                    print(f"  median={trial['median_seconds_per_step']:.4f}s, "
                          f"mean={trial['mean_seconds_per_step']:.4f}s, "
                          f"peak={trial['peak_allocated_memory_mib']:.1f} MiB", flush=True)
                except torch.cuda.OutOfMemoryError as exc:
                    trial = {"variant": name, "seed": seed, "status": "OOM", "error": str(exc)}
                    print("  OOM: recorded and continuing", flush=True)
                    torch.cuda.empty_cache()
                trial["round"] = r + 1
                report["trials"].append(trial)
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        report["summary"] = summarize(report["trials"])
        report["completed"] = True
    except Exception as exc:
        report["error"] = repr(exc)
        print("Comparison stopped; partial data preserved:", repr(exc), flush=True)
    finally:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print("Results:", report_path, flush=True)
        if "summary" in report:
            for name, row in report["summary"]["overall_by_variant"].items():
                if "pooled_median_seconds" in row:
                    extra = (" | vs B0: " + f"{row['pooled_reduction_vs_B0_percent']:.2f}%") if "pooled_reduction_vs_B0_percent" in row else ""
                    print(f"{name}: pooled median={row['pooled_median_seconds']:.4f}s, "
                          f"peak={row['peak_allocated_memory_mib_max']:.1f}MiB{extra}", flush=True)
    if not report["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
