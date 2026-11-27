"""E3: isolated full-training comparison of B0 VR-EPAGCL and B2x8.

Place this alongside model_b1.py and model_b2.py under
    extension_experiments/efficiency_B/train_e3.py

It imports the ORIGINAL run.py's graph augmentations, evaluation, diagnostics,
seeding, and data loader; it does not modify original source or past outputs.

Phase `sanity` is a 20-epoch connectivity test, not a formal experiment.
Phases `smoke` and `formal` use the fixed 1000-epoch E1 protocol.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

B_ROOT = Path(__file__).resolve().parent
EXT_ROOT = B_ROOT.parent
REPO_ROOT = EXT_ROOT.parent
# model_b2.py uses `from model_b1 import ...`; both live in B_ROOT.
for path in (B_ROOT, EXT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Model import MyModel  # noqa: E402
from model_b2 import MyModelB2  # noqa: E402
from Augmentation import add_edge_weights, drop_edge_weights  # noqa: E402
from Load_data import get_dataset  # noqa: E402
from Utils import get_optim  # noqa: E402
from run import (  # noqa: E402
    fit_linear_probe,
    load_or_create_split,
    make_augmented_views,
    output_diagnostics,
    score_test_with_selected_probe,
    set_seed,
    sync,
    to_json,
)
from vr_regularization import variance_floor_loss  # noqa: E402

# Freeze the E1 VR protocol. The only experimental variable is the model loss.
FIXED = dict(
    method="vr",
    optim="Adam",
    lr=0.01,
    weight_decay=0.001,
    eval_optim="Adam",
    eval_lr=0.01,
    eval_weight_decay=0.0,
    temperature=0.5,
    edge_add_rate=0.1,
    edge_drop_rate_1=0.1,
    edge_drop_rate_2=0.4,
    feat_drop_rate_1=0.0,
    feat_drop_rate_2=0.0,
    feat_dim=256,
    proj_hidden_dim=256,
    batch_compute=True,
    vr_lambda=10.0,
    vr_gamma=0.002,
    loss_log=1,
    save_model_every=0,
    shadow_stop=False,
    shadow_threshold=6.22,
)


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E3: validate-only selected B0 VR vs Fast VR B2x8, no early stop"
    )
    parser.add_argument("--variant", choices=("b0", "b2x8"), required=True)
    parser.add_argument("--phase", choices=("sanity", "smoke", "formal"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset", default="Coauthor-Phy")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--expected-gpu", default="NVIDIA GeForce RTX 2070",
                        help="Exact GPU name for comparable experiments; use empty string only for CPU sanity testing")
    parser.add_argument("--output-root", default=None,
                        help="Override new-output ROOT only; historical results are never overwritten")
    args = parser.parse_args(argv)
    if args.dataset != "Coauthor-Phy":
        parser.error("E3 currently fixes dataset to Coauthor-Phy for comparable evidence")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.phase in ("smoke", "formal") and not args.device.startswith("cuda:"):
        parser.error("smoke/formal must use the same CUDA GPU as the benchmark")
    args.epoch = 20 if args.phase == "sanity" else 1000
    args.eval = 10 if args.phase == "sanity" else 50
    args.eval_epoch = 30 if args.phase == "sanity" else 3000
    args.repeat = 1  # one explicit run per process; simplify paired-run audit
    for key, value in FIXED.items():
        setattr(args, key, value)
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    names = (
        "Model.py", "Augmentation.py", "Load_data.py", "Utils.py",
        "extension_experiments/run.py",
        "extension_experiments/fixed_eval.py",
        "extension_experiments/vr_regularization.py",
        "extension_experiments/efficiency_B/model_b1.py",
        "extension_experiments/efficiency_B/model_b2.py",
        "extension_experiments/efficiency_B/train_e3.py",
    )
    hashes = {}
    for name in names:
        p = REPO_ROOT / name
        if not p.is_file():
            raise FileNotFoundError(f"Required source is missing: {p}")
        hashes[name] = sha256(p)
    return hashes


def model_for(variant: str, num_features: int, args, device: torch.device):
    model_args = (num_features, args.feat_dim, args.proj_hidden_dim, args.temperature)
    if variant == "b0":
        model = MyModel(*model_args)
    elif variant == "b2x8":
        model = MyModelB2(*model_args, groups_per_call=8)
    else:
        raise ValueError(f"Unsupported variant: {variant}")
    return model.to(device)


def percentile90(times: list[float]) -> float:
    sorted_times = sorted(times)
    return sorted_times[min(len(sorted_times) - 1, (9 * len(sorted_times) + 9) // 10 - 1)]


def train_one_run(args, data, add_weights, adding_edge, drop_weights,
                  device: torch.device, run_dir: Path, split) -> dict:
    """Mimic run.py's model, loss, VR, probe, and test policy with per-phase timing.

    GPU synchronizations at epoch/probe boundaries are identical in both variants.
    Because of this extra instrumentation, these timings must be compared only
    against another E3 run (not the historical E1 elapsed_seconds directly).
    """
    set_seed(args.seed)
    model = model_for(args.variant, data.num_features, args, device)
    optimizer = get_optim(model.parameters(), args)
    best_val = -1.0
    best_path = run_dir / "best_by_validation.pt"
    run_start = perf_counter()
    main_step_times = []
    probe_times = []
    loss_by_epoch = {}
    shadow_decision = None  # Explicitly OFF: no early stop or shadow decision
    if device.type == "cuda":
        sync(device)
        torch.cuda.reset_peak_memory_stats(device)

    with (run_dir / "loss.csv").open("w", newline="", encoding="utf-8") as fp_loss, \
         (run_dir / "validation.csv").open("w", newline="", encoding="utf-8") as fp_eval, \
         (run_dir / "epoch_timing.csv").open("w", newline="", encoding="utf-8") as fp_timing:
        log_loss = csv.writer(fp_loss)
        log_val = csv.writer(fp_eval)
        log_time = csv.writer(fp_timing)
        log_loss.writerow(("epoch", "contrastive_loss", "variance_loss", "total_loss", "elapsed_seconds"))
        log_val.writerow(("encoder_epoch", "best_probe_epoch", "validation_accuracy", "is_best", "elapsed_seconds"))
        log_time.writerow(("epoch", "main_step_seconds", "linear_probe_seconds"))

        for epoch in range(1, args.epoch + 1):
            sync(device)
            step_start = perf_counter()
            model.train()
            optimizer.zero_grad()
            (x1, edge1), (x2, edge2) = make_augmented_views(
                data, add_weights, adding_edge, drop_weights, args
            )
            out1, z1 = model(x1, edge1)
            out2, z2 = model(x2, edge2)
            # B0 uses original MyModel.loss; B2x8 changes ONLY this loss implementation.
            contrastive = model.loss(z1, z2, args.batch_compute)
            vr_penalty = variance_floor_loss(out1, out2, gamma=args.vr_gamma)
            total_loss = contrastive + args.vr_lambda * vr_penalty
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"Non-finite total training loss at epoch {epoch}")
            total_loss.backward()
            optimizer.step()

            # Match E1 original loss logs (per epoch and checkpoint 50/75/100).
            if epoch % args.loss_log == 0 or epoch in (50, 75, 100) or epoch == args.epoch:
                log_loss.writerow((
                    epoch,
                    f"{contrastive.item():.8f}",
                    f"{vr_penalty.item():.8f}",
                    f"{total_loss.item():.8f}",
                    f"{perf_counter() - run_start:.2f}",
                ))
            if epoch in (50, 75, 100):
                loss_by_epoch[epoch] = float(contrastive.detach().item())

            sync(device)
            step_seconds = perf_counter() - step_start
            main_step_times.append(step_seconds)
            probe_seconds = 0.0

            if epoch % args.eval == 0 or epoch == args.epoch:
                # Identical validation-only probe/checkpoint selection as run.py.
                sync(device)
                probe_start = perf_counter()
                model.eval()
                with torch.no_grad():
                    out = model(data.x, data.edge_index)[0].detach()
                probe = fit_linear_probe(
                    out, data.y, split,
                    eval_epochs=args.eval_epoch,
                    eval_optimizer=args.eval_optim,
                    eval_lr=args.eval_lr,
                    eval_weight_decay=args.eval_weight_decay,
                    eval_seed=args.seed * 1009 + epoch,
                )
                is_best = probe.val_accuracy > best_val
                if is_best:
                    best_val = probe.val_accuracy
                    torch.save({
                        "encoder_state": model.state_dict(),
                        "classifier_state": probe.classifier_state,
                        "encoder_epoch": epoch,
                        "probe_epoch": probe.best_epoch,
                        "validation_accuracy": probe.val_accuracy,
                        "seed": args.seed,
                    }, best_path)
                log_val.writerow((
                    epoch, probe.best_epoch, f"{probe.val_accuracy:.6f}",
                    int(is_best), f"{perf_counter() - run_start:.2f}"
                ))
                print(
                    f"[seed={args.seed}][{args.variant}] epoch={epoch} "
                    f"contrastive={contrastive.item():.4f} "
                    f"vr={vr_penalty.item():.6f} val={probe.val_accuracy:.3f} "
                    f"best_val={best_val:.3f}", flush=True
                )
                sync(device)
                probe_seconds = perf_counter() - probe_start
                probe_times.append(probe_seconds)

            if args.save_model_every > 0 and epoch % args.save_model_every == 0:
                torch.save(model.state_dict(), run_dir / f"ckpt_epoch{epoch}.pt")
            log_time.writerow((epoch, f"{step_seconds:.8f}", f"{probe_seconds:.8f}"))
            if epoch % 10 == 0:
                fp_loss.flush()
                fp_eval.flush()
                fp_timing.flush()

    # No testing until encoder and classifier are selected solely by validation.
    sync(device)
    final_start = perf_counter()
    chosen = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(chosen["encoder_state"])
    model.eval()
    with torch.no_grad():
        embedding = model(data.x, data.edge_index)[0].detach()
    test_accuracy = score_test_with_selected_probe(
        embedding, data.y, split, chosen["classifier_state"]
    )
    diagnostics = output_diagnostics(embedding)
    sync(device)
    finalization_seconds = perf_counter() - final_start
    total_elapsed = perf_counter() - run_start
    main_total = sum(main_step_times)
    eval_total = sum(probe_times)
    peak_mem = (
        torch.cuda.max_memory_allocated(device) / 1048576
        if device.type == "cuda" else None
    )
    result = {
        "method": "vr",
        "variant": args.variant,
        "phase": args.phase,
        "seed": args.seed,
        "completed_epochs": args.epoch,
        "probe_count": len(probe_times),
        "encoder_selected_epoch": int(chosen["encoder_epoch"]),
        "classifier_selected_epoch": int(chosen["probe_epoch"]),
        "selected_validation_accuracy_pct": float(chosen["validation_accuracy"]),
        "final_test_accuracy_pct": float(test_accuracy),
        "elapsed_seconds": total_elapsed,
        "main_model_seconds": main_total,
        "linear_probe_seconds": eval_total,
        "finalization_seconds": finalization_seconds,
        "mean_seconds_per_main_step": statistics.mean(main_step_times),
        "median_seconds_per_main_step": statistics.median(main_step_times),
        "p90_seconds_per_main_step": percentile90(main_step_times),
        "peak_allocated_memory_mib": peak_mem,
        "shadow_candidate_epoch_100": shadow_decision,
        "shadow_threshold_historical": None,
        "diagnostics": diagnostics,
        "final_contrastive_loss": float(contrastive.detach().item()),
        "final_variance_loss": float(vr_penalty.detach().item()),
        "final_total_loss": float(total_loss.detach().item()),
        "note": (
            "No early stopping. Test only after validation selection. "
            "Per-main-step timing includes CUDA sync at epoch boundaries; "
            "compare B0/B2x8 within this runner only."
        ),
    }
    to_json(run_dir / "final.json", result)
    return result


def main(argv: list[str] | None = None) -> None:
    args = arguments(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable in this Python environment")
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        if args.expected_gpu and device_name != args.expected_gpu:
            raise RuntimeError(
                f"GPU mismatch: expected {args.expected_gpu!r}, observed {device_name!r}. "
                "Use exactly the same GPU as B0/B2 benchmarks."
            )
    else:
        if args.phase != "sanity":
            raise RuntimeError("CPU is permitted only for sanity validation")
        device_name = "CPU"

    source_hash = source_hashes()
    dataset_path = Path(args.dataset_path) if args.dataset_path else (
        REPO_ROOT.parent / "datasets" / args.dataset
    )
    dataset_path = dataset_path.expanduser().resolve()
    if not dataset_path.parent.exists():
        raise FileNotFoundError(f"Dataset parent directory missing: {dataset_path.parent}")
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else B_ROOT / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    experiment_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + f"_E3_{args.phase}_{args.variant}_{args.dataset}_seed_{args.seed}"
    )
    output = output_root / experiment_id
    output.mkdir(parents=True, exist_ok=False)
    run_dir = output / f"run_01_seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=False)

    to_json(output / "args.json", {
        **vars(args), "dataset_path_resolved": str(dataset_path),
        "repo_root": str(REPO_ROOT),
        "protocol": "original E1 VR fixed split and validation-only selection",
    })
    to_json(output / "source_sha256.json", source_hash)
    print(f"E3 {args.phase} {args.variant}  |  seed={args.seed}  |  GPU={device_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output}", flush=True)

    set_seed(args.seed)
    sync(device)
    preprocess_start = perf_counter()
    data = get_dataset(str(dataset_path), args.dataset).to(device)
    add_weights, adding_edge = add_edge_weights(data.edge_index, args.edge_add_rate)
    drop_weights = drop_edge_weights(data.edge_index)
    sync(device)
    preprocess_seconds = perf_counter() - preprocess_start
    print(f"Preprocessing: {preprocess_seconds:.2f}s, nodes={data.x.size(0)}", flush=True)

    split_path = EXT_ROOT / "splits" / args.dataset / f"seed_{args.seed}_nodes_{data.x.size(0)}.pt"
    split = load_or_create_split(split_path, data.x.size(0), args.seed)
    split_hash = sha256(split_path)
    print(f"Fixed split: {split_path} (sha256 {split_hash[:12]}...)", flush=True)
    to_json(output / "environment.json", {
        "device": str(device),
        "gpu_name": device_name,
        "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
        "dataset_nodes": int(data.x.size(0)),
        "dataset_features": int(data.x.size(1)),
        "preprocessing_seconds": preprocess_seconds,
        "split_path": str(split_path),
        "split_sha256": split_hash,
        "timing_policy": (
            "CUDA synchronized wall time per epoch: augmentation + two forwards + "
            "contrastive + original VR + backward + optimizer + scalar log; "
            "probe excluded. Linear probe timed separately."
        ),
        "full_run_policy": "elapsed_seconds includes 1000 main epochs, validation probes, and final test; excludes data preprocessing",
    })

    try:
        result = train_one_run(
            args, data, add_weights, adding_edge, drop_weights, device, run_dir, split
        )
    except BaseException as exc:
        to_json(output / "FAILED.json", {"exception_type": type(exc).__name__, "error": str(exc)})
        raise
    result["split_path"] = str(split_path)
    result["split_sha256"] = split_hash
    to_json(run_dir / "final.json", result)
    summary = {
        "variant": args.variant,
        "phase": args.phase,
        "seed": args.seed,
        "test_mean_pct": result["final_test_accuracy_pct"],
        "test_std_pct_sample": None,
        "test_values_pct": [result["final_test_accuracy_pct"]],
        "preprocessing_seconds": preprocess_seconds,
        "total_run_seconds_excluding_preprocessing": result["elapsed_seconds"],
        "main_model_seconds": result["main_model_seconds"],
        "linear_probe_seconds": result["linear_probe_seconds"],
        "peak_allocated_memory_mib": result["peak_allocated_memory_mib"],
        "evaluation_policy": "fixed split per seed; validation-only selection; test once at end",
        "early_stop_policy": "disabled, no shadow-stop decision",
    }
    to_json(output / "summary.json", summary)
    print("\n=== E3 COMPLETED ===")
    print(f"variant={args.variant}, phase={args.phase}, seed={args.seed}")
    print(f"final_test_accuracy_pct={result['final_test_accuracy_pct']:.4f}")
    print(f"selected_validation_accuracy_pct={result['selected_validation_accuracy_pct']:.4f}")
    print(f"main_model_seconds={result['main_model_seconds']:.3f}")
    print(f"mean_seconds_per_main_step={result['mean_seconds_per_main_step']:.5f}")
    print(f"linear_probe_seconds={result['linear_probe_seconds']:.3f}")
    print(f"elapsed_seconds={result['elapsed_seconds']:.3f}")
    print(f"peak_allocated_memory_mib={result['peak_allocated_memory_mib']}")
    print(f"Results: {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
