"""Experiment C: paired comparison of original and reversed drop weights.

Run from EPAGCL root. Original modules and shared fixed_eval.py are read-only.
Results go exclusively to reverse_drop_C/outputs; split files are shared.
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
SHARED_ROOT = EXPERIMENT_ROOT.parent
REPO_ROOT = SHARED_ROOT.parent
sys.path.insert(0, str(SHARED_ROOT))
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
from augmentation_c import (  # noqa: E402
    reverse_drop_edge_weights, prepare_edge_alignment, validate_weights, expected_drop_rate,
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment C: reverse edge-dropping weights")
    parser.add_argument("--method", choices=("both", "baseline", "reverse_drop"), default="both")
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
    args = parser.parse_args()
    if args.epoch <= 0 or args.repeat <= 0 or args.eval <= 0 or args.eval_epoch <= 0:
        parser.error("--epoch, --repeat, --eval and --eval-epoch must all be positive")
    if args.loss_log <= 0:
        parser.error("--loss-log must be positive")
    for name in ("edge_add_rate", "edge_drop_rate_1", "edge_drop_rate_2",
                 "feat_drop_rate_1", "feat_drop_rate_2"):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.edge_add_rate == 0:
        parser.error("--edge-add-rate must be positive for this add+drop comparison")
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
    # Count before adding edges; these are directed edge_index columns.
    original_count = data.edge_index.size(1)
    drop_counts = (original_count - e1.size(1), original_count - e2.size(1))
    return (x1, edge1), (x2, edge2), drop_counts


def to_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def train_one_run(args, data, add_weights, adding_edge, drop_weights, device, seed, run_dir, split):
    set_seed(seed)
    model = MyModel(data.num_features, args.feat_dim, args.proj_hidden_dim, args.temperature).to(device)
    optimizer = get_optim(model.parameters(), args)
    best_val = -1.0
    best_path = run_dir / "best_by_validation.pt"
    total_dropped_1 = 0
    total_dropped_2 = 0
    original_count = data.edge_index.size(1)
    run_start = perf_counter()

    with (run_dir / "loss.csv").open("w", newline="", encoding="utf-8") as fp_loss, \
         (run_dir / "validation.csv").open("w", newline="", encoding="utf-8") as fp_eval, \
         (run_dir / "edge_drop.csv").open("w", newline="", encoding="utf-8") as fp_drop:
        log_loss = csv.writer(fp_loss)
        log_val = csv.writer(fp_eval)
        log_drop = csv.writer(fp_drop)
        log_drop.writerow(("epoch", "dropped_view1", "dropped_view2", "original_edge_columns", "drop_rate_view1", "drop_rate_view2"))
        log_loss.writerow(("epoch", "contrastive_loss", "elapsed_seconds"))
        log_val.writerow(("encoder_epoch", "best_probe_epoch", "validation_accuracy", "is_best", "elapsed_seconds"))
        for epoch in range(1, args.epoch + 1):
            model.train()
            optimizer.zero_grad()
            (x1, edge1), (x2, edge2), (dropped1, dropped2) = make_augmented_views(data, add_weights, adding_edge, drop_weights, args)
            total_dropped_1 += dropped1
            total_dropped_2 += dropped2
            log_drop.writerow((epoch, dropped1, dropped2, original_count,
                               dropped1 / original_count, dropped2 / original_count))
            _, z1 = model(x1, edge1)
            _, z2 = model(x2, edge2)
            contrastive = model.loss(z1, z2, args.batch_compute)
            # Both groups use only the original EPAGCL contrastive loss.
            if not torch.isfinite(contrastive):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            contrastive.backward()
            optimizer.step()

            if epoch % args.loss_log == 0 or epoch == args.epoch:
                log_loss.writerow((epoch, f"{contrastive.item():.8f}",
                                   f"{perf_counter() - run_start:.2f}"))
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
                      f"val={probe.val_accuracy:.3f} "
                      f"best_val={best_val:.3f}")
            if args.save_model_every > 0 and epoch % args.save_model_every == 0:
                torch.save(model.state_dict(), run_dir / f"ckpt_epoch{epoch}.pt")
            if epoch % 10 == 0:
                fp_loss.flush()
                fp_eval.flush()
                fp_drop.flush()

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
        "actual_drop_rate_view1": total_dropped_1 / (original_count * args.epoch),
        "actual_drop_rate_view2": total_dropped_2 / (original_count * args.epoch),
        "note": "Test was held out until after validation selection.",
    }
    to_json(run_dir / "final.json", result)
    return result


def summarize_group(method, results, prep_seconds, expected_rates):
    accuracies = np.array([r["final_test_accuracy_pct"] for r in results])
    return {
        "method": method,
        "runs": len(results),
        "test_mean_pct": float(np.mean(accuracies)),
        "test_std_pct_population": float(np.std(accuracies)),
        "test_values_pct": accuracies.tolist(),
        "seeds": [r["seed"] for r in results],
        "expected_drop_rate_view1": expected_rates[0],
        "expected_drop_rate_view2": expected_rates[1],
        "mean_actual_drop_rate_view1": float(np.mean([r["actual_drop_rate_view1"] for r in results])),
        "mean_actual_drop_rate_view2": float(np.mean([r["actual_drop_rate_view2"] for r in results])),
        "preprocessing_seconds": prep_seconds,
        "total_run_seconds_excluding_preprocessing": sum(r["elapsed_seconds"] for r in results),
        "evaluation_policy": "fixed split per seed; validation-only checkpoint selection; test once at end",
    }


def main() -> None:
    args = get_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this environment")
    source_hash = hash_sources()
    source_hash["extension_experiments/fixed_eval.py"] = hashlib.sha256((SHARED_ROOT / "fixed_eval.py").read_bytes()).hexdigest()
    for name in ("run.py", "augmentation_c.py"):
        source_hash[f"reverse_drop_C/{name}"] = hashlib.sha256((EXPERIMENT_ROOT / name).read_bytes()).hexdigest()
    dataset_path = Path(args.dataset_path) if args.dataset_path else (REPO_ROOT.parent / "datasets" / args.dataset)
    dataset_path = dataset_path.expanduser().resolve()
    if not dataset_path.parent.exists():
        raise FileNotFoundError(f"Dataset parent directory does not exist: {dataset_path.parent}")

    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + args.method + "_" + args.dataset
    experiment_dir = EXPERIMENT_ROOT / "outputs" / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args))
    config.update({"dataset_path_resolved": str(dataset_path), "project_root": str(REPO_ROOT)})
    to_json(experiment_dir / "args.json", config)
    to_json(experiment_dir / "source_sha256.json", source_hash)
    print(f"Experiment output: {experiment_dir}")

    set_seed(args.seed)
    sync(device)
    preprocess_start = perf_counter()
    data = get_dataset(str(dataset_path), args.dataset).to(device)
    canonical_edges, weight_index, alignment = prepare_edge_alignment(data.edge_index)
    to_json(experiment_dir / "edge_alignment.json", alignment)
    print(f"Edge alignment: {alignment}")
    if alignment["duplicate_original_entries"] or alignment["reverse_entries_added_by_to_undirected"]:
        print("WARNING: input graph is not a unique symmetric edge list. "
              "Original sampling columns are preserved; both groups use degrees from "
              "the canonical undirected graph. Interpret comparisons accordingly.")
    add_weights, adding_edge = add_edge_weights(data.edge_index, args.edge_add_rate)
    methods = ("baseline", "reverse_drop") if args.method == "both" else (args.method,)
    drop_weights = {}
    expected = {}
    for method in methods:
        weight_fn = drop_edge_weights if method == "baseline" else reverse_drop_edge_weights
        canonical_weights = weight_fn(canonical_edges)
        validate_weights(canonical_weights, canonical_edges, method)
        # Original sampling uses data.edge_index, not the canonical edge order.
        # Map by endpoints for BOTH methods; keep all original graph columns.
        weights = canonical_weights[weight_index]
        validate_weights(weights, data.edge_index, method)
        drop_weights[method] = weights
        expected[method] = tuple(expected_drop_rate(weights, p) for p in (args.edge_drop_rate_1, args.edge_drop_rate_2))
        print(f"[{method}] expected drop rates after clipping: {expected[method]}")
    sync(device)
    prep_seconds = perf_counter() - preprocess_start
    print(f"Shared preprocessing time: {prep_seconds:.2f}s, nodes={data.x.size(0)}")

    results = {method: [] for method in methods}
    groups = {}
    for method in methods:
        print(f"[{method}] Starting all {args.repeat} runs", flush=True)
        for index in range(args.repeat):
            seed = int(args.seed + index)
            split_path = SHARED_ROOT / "splits" / args.dataset / f"seed_{seed}_nodes_{data.x.size(0)}.pt"
            split = load_or_create_split(split_path, data.x.size(0), seed)
            method_args = argparse.Namespace(**vars(args))
            method_args.method = method
            run_dir = experiment_dir / method / f"run_{index + 1:02d}_seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=False)
            print(f"[{method}] Run {index + 1}/{args.repeat}, seed={seed}, fixed split={split_path}")
            result = train_one_run(method_args, data, add_weights, adding_edge,
                                   drop_weights[method], device, seed, run_dir, split)
            result["split_path"] = str(split_path)
            to_json(run_dir / "final.json", result)
            results[method].append(result)
            to_json(experiment_dir / "partial_summary.json", {"completed": results})

        # Finish and save the entire group before starting the next method.
        group = summarize_group(method, results[method], prep_seconds, expected[method])
        group["dataset"] = args.dataset
        groups[method] = group
        to_json(experiment_dir / method / "summary.json", group)
        print(f"[{method}] Test mean={group['test_mean_pct']:.3f}, std={group['test_std_pct_population']:.3f}")
        print(f"[{method}] Completed all {args.repeat}/{args.repeat} runs", flush=True)
    summary = {"dataset": args.dataset, "groups": groups, "shared_preprocessing_seconds": prep_seconds, "edge_alignment": alignment}
    if args.method == "both":
        paired = []
        for base, reverse in zip(results["baseline"], results["reverse_drop"]):
            if base["seed"] != reverse["seed"] or base["split_path"] != reverse["split_path"]:
                raise RuntimeError("Paired experiment seed/split mismatch")
            a, b = base["final_test_accuracy_pct"], reverse["final_test_accuracy_pct"]
            paired.append({"seed": base["seed"], "baseline_pct": a,
                           "reverse_drop_pct": b, "difference_percentage_points": b - a})
        summary["paired_results"] = paired
        summary["mean_paired_difference_percentage_points"] = float(np.mean([r["difference_percentage_points"] for r in paired]))
        with (experiment_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)
        print(f"Reverse minus baseline (percentage points): {summary['mean_paired_difference_percentage_points']:+.3f}")
    to_json(experiment_dir / "summary.json", summary)
    print(f"Results saved under: {experiment_dir}")


if __name__ == "__main__":
    main()
