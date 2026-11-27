"""Isolated EPAGCL extension experiment runner.

Run from the EPAGCL repository root:
    python extension_experiments/run.py --method baseline --epoch 20 --eval 10 --eval-epoch 30

Only reads original Model, Augmentation, Load_data, and Utils modules.
All new outputs are stored below extension_experiments/outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_ROOT.parent
# Import only shared, unchanged components from the original repository.
sys.path.insert(0, str(REPO_ROOT))

from Augmentation import (  # noqa: E402
    add_edges,
    add_edge_weights,
    drop_edges,
    drop_edge_weights,
    drop_feature_random,
)
from Load_data import get_dataset  # noqa: E402
from Model import MyModel  # noqa: E402
from Utils import get_optim  # noqa: E402

from fixed_eval import (  # noqa: E402
    fit_linear_probe,
    load_or_create_split,
    score_test_with_selected_probe,
)
from vr_regularization import shadow_should_stop, variance_floor_loss  # noqa: E402


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isolated EPAGCL extension experiments")
    parser.add_argument("--method", choices=("baseline", "vr"), default="baseline")
    parser.add_argument("--dataset", default="Coauthor-Phy")
    parser.add_argument("--dataset-path", default=None,
                        help="Full dataset directory; default is ../datasets/<dataset> from project root")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--eval", type=int, default=50, help="Probe interval in encoder epochs")
    parser.add_argument("--eval-epoch", type=int, default=3000)
    parser.add_argument("--loss-log", type=int, default=1)
    parser.add_argument("--save-model-every", type=int, default=0,
                        help="0 means best checkpoint only; otherwise also save periodic checkpoints")
    parser.add_argument("--optim", choices=("Adam",), default="Adam")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--eval-optim", choices=("Adam", "SGD"), default="Adam")
    parser.add_argument("--eval-lr", type=float, default=0.01)
    parser.add_argument("--eval-weight-decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--edge-add-rate", type=float, default=0.1)
    parser.add_argument("--edge-drop-rate-1", type=float, default=0.1)
    parser.add_argument("--edge-drop-rate-2", type=float, default=0.4)
    parser.add_argument("--feat-drop-rate-1", type=float, default=0.0)
    parser.add_argument("--feat-drop-rate-2", type=float, default=0.0)
    parser.add_argument("--feat-dim", type=int, default=256)
    parser.add_argument("--proj-hidden-dim", type=int, default=256)
    parser.add_argument("--batch-compute", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vr-lambda", type=float, default=None,
                        help="Required with --method vr; choose using validation/diagnostics only")
    parser.add_argument("--vr-gamma", type=float, default=None,
                        help="Required with --method vr; depends on encoder output scale")
    parser.add_argument("--shadow-stop", action=argparse.BooleanOptionalAction, default=True,
                        help="Only flags potential early stop; NEVER terminates training")
    parser.add_argument("--shadow-threshold", type=float, default=6.22,
                        help="Historical Coauthor-Physics threshold; not validated on new seeds")
    args = parser.parse_args()
    if args.epoch <= 0 or args.repeat <= 0 or args.eval <= 0 or args.eval_epoch <= 0:
        parser.error("--epoch, --repeat, --eval and --eval-epoch must all be positive")
    if args.loss_log <= 0:
        parser.error("--loss-log must be positive")
    if args.method == "vr" and (
        args.vr_lambda is None or args.vr_lambda < 0 or
        args.vr_gamma is None or args.vr_gamma <= 0
    ):
        parser.error("VR requires --vr-lambda >= 0 and --vr-gamma > 0")
    if args.method == "baseline" and (args.vr_lambda is not None or args.vr_gamma is not None):
        parser.error("Do not supply VR parameters for baseline")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def hash_sources() -> dict:
    names = ("Model.py", "Augmentation.py", "Load_data.py", "Utils.py")
    result = {}
    for name in names:
        path = REPO_ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing original dependency: {path}")
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def make_augmented_views(data, add_weights, adding_edge, drop_weights, args):
    # Same EPAGCL (single added view, both views adaptively dropped)
    # path as the original main.py with --add_single enabled.
    e1 = drop_edges(data.edge_index, drop_weights, args.edge_drop_rate_1)
    e2 = drop_edges(data.edge_index, drop_weights, args.edge_drop_rate_2)
    edge1 = add_edges(add_weights, adding_edge, e1)
    edge2 = e2
    x1 = drop_feature_random(data.x, args.feat_drop_rate_1)
    x2 = drop_feature_random(data.x, args.feat_drop_rate_2)
    return (x1, edge1), (x2, edge2)


def output_diagnostics(z: torch.Tensor) -> dict:
    """Diagnostics are label-free; they do not inspect held-out test labels."""
    with torch.no_grad():
        std = z.detach().std(dim=0, unbiased=False).mean().item()
        n = z.size(0)
        if n < 2:
            return {"mean_feature_std": std, "mean_pairwise_cosine": None}
        # Fixed, evenly-spaced sample, independent of training and evaluation RNG.
        sample_idx = torch.linspace(0, n - 1, steps=min(512, n), device=z.device).long()
        vectors = torch.nn.functional.normalize(z.index_select(0, sample_idx), p=2, dim=1)
        sim = vectors @ vectors.t()
        size = vectors.size(0)
        off_diag_mean = ((sim.sum() - sim.diag().sum()) / (size * (size - 1))).item()
        return {"mean_feature_std": std, "mean_pairwise_cosine": off_diag_mean}


def to_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def train_one_run(args, data, add_weights, adding_edge, drop_weights, device, seed, run_dir, split):
    set_seed(seed)
    model = MyModel(data.num_features, args.feat_dim, args.proj_hidden_dim, args.temperature).to(device)
    optimizer = get_optim(model.parameters(), args)
    best_val = -1.0
    best_path = run_dir / "best_by_validation.pt"
    loss_by_epoch = {}
    shadow_decision = None
    run_start = perf_counter()

    with (run_dir / "loss.csv").open("w", newline="", encoding="utf-8") as fp_loss, \
         (run_dir / "validation.csv").open("w", newline="", encoding="utf-8") as fp_eval:
        log_loss = csv.writer(fp_loss)
        log_val = csv.writer(fp_eval)
        log_loss.writerow(("epoch", "contrastive_loss", "variance_loss", "total_loss", "elapsed_seconds"))
        log_val.writerow(("encoder_epoch", "best_probe_epoch", "validation_accuracy", "is_best", "elapsed_seconds"))
        for epoch in range(1, args.epoch + 1):
            model.train()
            optimizer.zero_grad()
            (x1, edge1), (x2, edge2) = make_augmented_views(data, add_weights, adding_edge, drop_weights, args)
            out1, z1 = model(x1, edge1)
            out2, z2 = model(x2, edge2)
            contrastive = model.loss(z1, z2, args.batch_compute)
            if args.method == "vr":
                vr_penalty = variance_floor_loss(out1, out2, gamma=args.vr_gamma)
                total_loss = contrastive + args.vr_lambda * vr_penalty
            else:
                vr_penalty = contrastive.new_zeros(())
                total_loss = contrastive
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            total_loss.backward()
            optimizer.step()

            if epoch % args.loss_log == 0 or epoch in (50, 75, 100) or epoch == args.epoch:
                log_loss.writerow((epoch, f"{contrastive.item():.8f}",
                                   f"{vr_penalty.item():.8f}", f"{total_loss.item():.8f}",
                                   f"{perf_counter() - run_start:.2f}"))
            if epoch in (50, 75, 100):
                # This is raw contrastive loss, NOT contrastive + VR penalty.
                loss_by_epoch[epoch] = float(contrastive.detach().item())
                if epoch == 100 and args.shadow_stop:
                    shadow_decision = shadow_should_stop(loss_by_epoch, args.shadow_threshold)
                    print(f"[seed={seed}] shadow stop candidate at epoch 100: {shadow_decision}; "
                          "monitoring only, training continues")
            if epoch % args.eval == 0 or epoch == args.epoch:
                model.eval()
                with torch.no_grad():
                    out = model(data.x, data.edge_index)[0].detach()
                probe = fit_linear_probe(
                    out, data.y, split,
                    eval_epochs=args.eval_epoch,
                    eval_optimizer=args.eval_optim,
                    eval_lr=args.eval_lr,
                    eval_weight_decay=args.eval_weight_decay,
                    eval_seed=seed * 1009 + epoch,
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
                        "seed": seed,
                    }, best_path)
                log_val.writerow((epoch, probe.best_epoch, f"{probe.val_accuracy:.6f}",
                                  int(is_best), f"{perf_counter() - run_start:.2f}"))
                print(f"[seed={seed}] epoch={epoch} contrastive={contrastive.item():.4f} "
                      f"vr={vr_penalty.item():.6f} val={probe.val_accuracy:.3f} "
                      f"best_val={best_val:.3f}")
            if args.save_model_every > 0 and epoch % args.save_model_every == 0:
                torch.save(model.state_dict(), run_dir / f"ckpt_epoch{epoch}.pt")
            if epoch % 10 == 0:
                fp_loss.flush()
                fp_eval.flush()

    # The test labels are accessed for the FIRST time only here, after
    # both the encoder and linear classifier were selected on validation.
    chosen = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(chosen["encoder_state"])
    model.eval()
    with torch.no_grad():
        embedding = model(data.x, data.edge_index)[0].detach()
    test_accuracy = score_test_with_selected_probe(embedding, data.y, split, chosen["classifier_state"])
    sync(device)
    elapsed = perf_counter() - run_start
    result = {
        "method": args.method,
        "seed": seed,
        "encoder_selected_epoch": int(chosen["encoder_epoch"]),
        "classifier_selected_epoch": int(chosen["probe_epoch"]),
        "selected_validation_accuracy_pct": float(chosen["validation_accuracy"]),
        "final_test_accuracy_pct": float(test_accuracy),
        "elapsed_seconds": elapsed,
        "shadow_candidate_epoch_100": shadow_decision,
        "shadow_threshold_historical": args.shadow_threshold if args.shadow_stop else None,
        "diagnostics": output_diagnostics(embedding),
        "note": "No actual early stopping; test was held out until after validation selection.",
    }
    to_json(run_dir / "final.json", result)
    return result


def main() -> None:
    args = get_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but not available in this Python environment")
    source_hash = hash_sources()
    dataset_path = Path(args.dataset_path) if args.dataset_path else (REPO_ROOT.parent / "datasets" / args.dataset)
    dataset_path = dataset_path.expanduser().resolve()
    if not dataset_path.parent.exists():
        raise FileNotFoundError(f"Dataset parent directory does not exist: {dataset_path.parent}")
    # Do not modify the historical trained_models directory.
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + args.method + "_" + args.dataset
    experiment_dir = EXPERIMENT_ROOT / "outputs" / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args))
    config.update({"dataset_path_resolved": str(dataset_path), "project_root": str(REPO_ROOT)})
    to_json(experiment_dir / "args.json", config)
    to_json(experiment_dir / "original_source_sha256.json", source_hash)
    print(f"Experiment output: {experiment_dir}")
    set_seed(args.seed)
    sync(device)
    preprocess_start = perf_counter()
    data = get_dataset(str(dataset_path), args.dataset).to(device)
    add_weights, adding_edge = add_edge_weights(data.edge_index, args.edge_add_rate)
    drop_weights = drop_edge_weights(data.edge_index)
    sync(device)
    prep_seconds = perf_counter() - preprocess_start
    print(f"Preprocessing time: {prep_seconds:.2f}s, nodes={data.x.size(0)}")
    results = []
    for index in range(args.repeat):
        seed = int(args.seed + index)
        run_dir = experiment_dir / f"run_{index + 1:02d}_seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=False)
        split_path = EXPERIMENT_ROOT / "splits" / args.dataset / f"seed_{seed}_nodes_{data.x.size(0)}.pt"
        split = load_or_create_split(split_path, data.x.size(0), seed)
        print(f"Run {index + 1}/{args.repeat}, seed={seed}, fixed split={split_path}")
        result = train_one_run(args, data, add_weights, adding_edge, drop_weights,
                               device, seed, run_dir, split)
        result["split_path"] = str(split_path)
        results.append(result)
        to_json(experiment_dir / "partial_summary.json", {"completed": results})
    accuracies = np.array([r["final_test_accuracy_pct"] for r in results])
    summary = {
        "method": args.method,
        "dataset": args.dataset,
        "runs": len(results),
        "test_mean_pct": float(np.mean(accuracies)),
        "test_std_pct_population": float(np.std(accuracies)),
        "test_values_pct": [float(value) for value in accuracies],
        "preprocessing_seconds": prep_seconds,
        "total_run_seconds_excluding_preprocessing": sum(r["elapsed_seconds"] for r in results),
        "evaluation_policy": "fixed split per seed; validation-only checkpoint selection; test once at end",
        "early_stop_policy": "shadow/monitoring only; no training was stopped",
    }
    to_json(experiment_dir / "summary.json", summary)
    print(f"Finished. Test mean={summary['test_mean_pct']:.3f}, std={summary['test_std_pct_population']:.3f}")
    print(f"Results saved under: {experiment_dir}")


if __name__ == "__main__":
    main()
