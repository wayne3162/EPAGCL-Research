"""E3 diagnostic: locate the first observed cross-replay differences.

This script compares two independent *same-seed* runs WITHIN each model variant.
It tracks exact CPU/CUDA/Python/NumPy RNG state hashes, full augmented edge-index
hashes, and full trainable parameter hashes after every optimizer update.

It intentionally changes execution timing by synchronizing/copying GPU tensors.
It is NOT a speed benchmark, deterministic-mode experiment, formal E3 run,
or a replacement for the original five-seed results. No probes or labels used.

Place in extension_experiments/efficiency_B/ alongside train_e3.py.
Run from the project root with:
    python -u extension_experiments/efficiency_B/diagnose_e3_first_divergence.py \
        --seed 20261129 --steps 40 --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from train_e3 import (
    B_ROOT,
    REPO_ROOT,
    FIXED,
    add_edge_weights,
    drop_edge_weights,
    get_dataset,
    get_optim,
    make_augmented_views,
    model_for,
    set_seed,
    source_hashes,
    sync,
    variance_floor_loss,
)


def sha_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def fingerprint_tensor(t: torch.Tensor) -> str:
    """Full, exact hash. CPU copy/sync is intentional and makes this diagnostic invasive."""
    x = t.detach().contiguous().to(device="cpu")
    h = hashlib.sha256()
    h.update(str(x.dtype).encode("ascii"))
    h.update(str(tuple(x.shape)).encode("ascii"))
    h.update(x.numpy().tobytes(order="C"))
    return h.hexdigest()


def parameter_fingerprint(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    # Hash *all* parameters and buffers, not just a small sampled subset.
    # Avoid torch.save since a serialization archive is not a canonical byte hash.
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(fingerprint_tensor(t).encode("ascii"))
    return h.hexdigest()


def rng_fingerprints(device: torch.device) -> dict[str, str]:
    return {
        "torch_cpu": fingerprint_tensor(torch.get_rng_state()),
        "torch_cuda": fingerprint_tensor(torch.cuda.get_rng_state(device)),
        "python_random": sha_bytes(pickle.dumps(random.getstate(), protocol=4)),
        "numpy_random": sha_bytes(pickle.dumps(np.random.get_state(), protocol=4)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261129)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--variants", default="b0,b2x8",
                        help="Comma-separated subset of b0,b2x8")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--replay-reference", default=None,
                        help="Existing replay_comparison.json; source hashes are checked")
    args = parser.parse_args()
    if args.steps < 3 or args.steps > 150:
        parser.error("--steps must be between 3 and 150")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    variants = args.variants.split(",")
    if len(variants) != len(set(variants)) or not variants or any(
        v not in ("b0", "b2x8") for v in variants
    ):
        parser.error("--variants must be b0, b2x8, or b0,b2x8")
    args.variants = variants
    if args.device != "cuda:0":
        parser.error("This experiment is intended for the same original cuda:0 GPU")
    return args


def check_reference(args: argparse.Namespace, hashes: dict[str, str]) -> str:
    p = (Path(args.replay_reference).expanduser().resolve()
         if args.replay_reference else
         B_ROOT / "outputs" / "e3_same_seed_replay_20261129" / "replay_comparison.json")
    if not p.is_file():
        raise FileNotFoundError(
            f"Required provenance reference not found: {p}. "
            "Pass --replay-reference /actual/path/replay_comparison.json"
        )
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("seed") != args.seed:
        raise ValueError("The reference seed is different from --seed")
    reference_hashes = doc.get("shared_original_source_sha256", {})
    missing = set(hashes) - set(reference_hashes)
    drift = [name for name, digest in hashes.items()
             if reference_hashes.get(name) != digest]
    if missing or drift:
        raise RuntimeError(
            f"Historical-vs-current source version mismatch. "
            f"Missing={sorted(missing)}, changed={sorted(drift)}. "
            "Do not compare training traces from different source versions."
        )
    return str(p)


def one_replay(
    variant: str,
    repeat: int,
    steps: int,
    seed: int,
    args_model: SimpleNamespace,
    data,
    add_weights,
    adding_edge,
    drop_weights,
    device: torch.device,
    writer: csv.DictWriter,
    file_handle,
) -> tuple[str, list[dict]]:
    # Match train_e3: reset RNG immediately before model + optimizer construction.
    set_seed(seed)
    model = model_for(variant, data.num_features, args_model, device)
    optimizer = get_optim(model.parameters(), args_model)
    sync(device)
    init_digest = parameter_fingerprint(model)
    rows = []
    print(f"\n[{variant} repeat {repeat}] initial model SHA256={init_digest[:16]}...", flush=True)

    for epoch in range(1, steps + 1):
        model.train()
        optimizer.zero_grad()
        before = rng_fingerprints(device)
        (x1, edge1), (x2, edge2) = make_augmented_views(
            data, add_weights, adding_edge, drop_weights, args_model
        )
        sync(device)
        after_aug = rng_fingerprints(device)
        edge1_sha = fingerprint_tensor(edge1)
        edge2_sha = fingerprint_tensor(edge2)

        out1, z1 = model(x1, edge1)
        out2, z2 = model(x2, edge2)
        contrastive = model.loss(z1, z2, args_model.batch_compute)
        vr = variance_floor_loss(out1, out2, gamma=args_model.vr_gamma)
        total = contrastive + args_model.vr_lambda * vr
        if not bool(torch.isfinite(total).item()):
            raise FloatingPointError(f"Non-finite loss at {variant} repeat {repeat} epoch {epoch}")
        after_loss = rng_fingerprints(device)
        total.backward()
        optimizer.step()
        sync(device)
        digest = parameter_fingerprint(model)
        row = {
            "variant": variant,
            "repeat": repeat,
            "epoch": epoch,
            "contrastive_loss": float(contrastive.detach().item()),
            "total_loss": float(total.detach().item()),
            "edge1_count": int(edge1.size(1)),
            "edge2_count": int(edge2.size(1)),
            "edge1_sha256": edge1_sha,
            "edge2_sha256": edge2_sha,
            "parameters_sha256": digest,
        }
        for prefix, fingerprints in (("before", before),
                                     ("after_aug", after_aug),
                                     ("after_loss", after_loss)):
            for k, v in fingerprints.items():
                row[f"{prefix}_{k}_sha256"] = v
        writer.writerow(row)
        rows.append(row)
        if epoch % 5 == 0 or epoch == 1 or epoch == steps:
            file_handle.flush()
            print(
                f"[{variant} #{repeat}] epoch={epoch:03d} "
                f"loss={row['contrastive_loss']:.8f} "
                f"params={digest[:10]} edge1={edge1_sha[:10]}",
                flush=True,
            )
    return init_digest, rows


def first_difference(a: list[dict], b: list[dict], key: str) -> int | None:
    if len(a) != len(b):
        raise RuntimeError("The two repeated runs do not have the same length")
    return next((x["epoch"] for x, y in zip(a, b)
                 if x[key] != y[key]), None)


def compare(variant: str, run1: tuple[str, list[dict]],
            run2: tuple[str, list[dict]]) -> dict:
    initial1, a = run1
    initial2, b = run2
    fields = [
        "edge1_sha256", "edge2_sha256", "parameters_sha256",
        "before_torch_cpu_sha256", "before_torch_cuda_sha256",
        "after_aug_torch_cpu_sha256", "after_aug_torch_cuda_sha256",
        "after_loss_torch_cpu_sha256", "after_loss_torch_cuda_sha256",
        "before_python_random_sha256", "before_numpy_random_sha256",
        "after_aug_python_random_sha256", "after_aug_numpy_random_sha256",
        "after_loss_python_random_sha256", "after_loss_numpy_random_sha256",
    ]
    result = {
        "initial_model_hashes_equal": initial1 == initial2,
        "first_exact_difference_epoch": {
            field: first_difference(a, b, field) for field in fields
        },
        "first_abs_loss_difference_over_threshold_epoch": {},
    }
    for threshold in (1e-8, 1e-6, 1e-4, 1e-2, 0.1):
        first = next((x["epoch"] for x, y in zip(a, b)
                      if abs(x["contrastive_loss"] - y["contrastive_loss"]) > threshold), None)
        result["first_abs_loss_difference_over_threshold_epoch"][str(threshold)] = first
    print(f"\n=== {variant} repeated-run comparison ===", flush=True)
    print("Initial parameter state equal:", result["initial_model_hashes_equal"], flush=True)
    for field, epoch in result["first_exact_difference_epoch"].items():
        print(f"  {field:<40} first difference epoch: {epoch}", flush=True)
    for threshold, epoch in result["first_abs_loss_difference_over_threshold_epoch"].items():
        print(f"  |loss difference| > {threshold:<9} first epoch: {epoch}", flush=True)
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device(args.device)
    gpu = torch.cuda.get_device_name(device)
    if gpu != "NVIDIA GeForce RTX 2070":
        raise RuntimeError(f"Expected the original RTX 2070, observed {gpu!r}")
    hashes = source_hashes()
    reference = check_reference(args, hashes)
    dataset_path = (Path(args.dataset_path).expanduser().resolve()
                    if args.dataset_path else (REPO_ROOT.parent / "datasets" / "Coauthor-Phy").resolve())
    if not dataset_path.parent.is_dir():
        raise FileNotFoundError(dataset_path.parent)
    out_root = (Path(args.output_root).expanduser().resolve()
                if args.output_root else B_ROOT / "outputs")
    out_root.mkdir(parents=True, exist_ok=True)
    output = out_root / (
        f"{datetime.now():%Y%m%d_%H%M%S_%f}_E3_rng_param_trace_seed_{args.seed}"
    )
    output.mkdir(parents=False, exist_ok=False)
    print("E3 early-divergence diagnostic; NOT a formal result or speed benchmark")
    print(f"GPU={gpu} seed={args.seed} steps={args.steps} variants={args.variants}")
    print(f"Original historical provenance: {reference}")
    print(f"Output: {output}", flush=True)

    # Preprocess once. The dataset and precomputed edge scores are held fixed
    # across *both* independently seeded repeats for each variant.
    set_seed(args.seed)
    data = get_dataset(str(dataset_path), "Coauthor-Phy").to(device)
    add_weights, adding_edge = add_edge_weights(data.edge_index, FIXED["edge_add_rate"])
    drop_weights = drop_edge_weights(data.edge_index)
    sync(device)
    model_args = SimpleNamespace(**FIXED)
    records = {}
    fields = [
        "variant", "repeat", "epoch", "contrastive_loss", "total_loss",
        "edge1_count", "edge2_count", "edge1_sha256", "edge2_sha256",
        "parameters_sha256",
    ] + [f"{stage}_{kind}_sha256"
         for stage in ("before", "after_aug", "after_loss")
         for kind in ("torch_cpu", "torch_cuda", "python_random", "numpy_random")]
    with (output / "divergence_trace.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for variant in args.variants:
            r1 = one_replay(variant, 1, args.steps, args.seed, model_args,
                            data, add_weights, adding_edge, drop_weights,
                            device, writer, fp)
            r2 = one_replay(variant, 2, args.steps, args.seed, model_args,
                            data, add_weights, adding_edge, drop_weights,
                            device, writer, fp)
            records[variant] = compare(variant, r1, r2)
    metadata = {
        "purpose": "Locate earliest observable same-seed divergence, not formal quality or speed",
        "seed": args.seed,
        "steps_per_replay": args.steps,
        "replays_per_variant": 2,
        "variants": args.variants,
        "gpu": gpu,
        "torch_version": torch.__version__,
        "reference": reference,
        "source_sha256": hashes,
        "summary": records,
        "limitations": [
            "Full CPU/GPU fingerprinting synchronizes and copies tensors, so execution differs from E3.",
            "A first exact parameter-hash difference means at least one bit differs; it does not quantify magnitude.",
            "Identical RNG states cannot by themselves prove identical random graph outputs; edge hashes are recorded.",
            "Only within-variant replay comparisons are valid for tracing nondeterminism.",
            "This diagnostic does not access labels, splits, validation, test, or checkpoints.",
            "Do not merge any result into the original five-seed formal experiment.",
        ],
    }
    (output / "divergence_summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nTrace CSV: {output / 'divergence_trace.csv'}")
    print(f"Summary: {output / 'divergence_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
