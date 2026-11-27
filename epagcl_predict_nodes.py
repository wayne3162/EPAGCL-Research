#!/usr/bin/env python3
"""Show real node-classification predictions from a trained EPAGCL encoder.

Place this file in the root of the official EPAGCL GitHub repository.
It imports the author's Model.MyModel and Load_data.get_dataset.

Example (run from the EPAGCL repository root):
    python epagcl_predict_nodes.py \
        --dataset Cora --data-root ../datasets \
        --checkpoint trained_models/Cora/completed_XXXX/1/ckpt_epoch500.pth

This is an additional, transparent linear-probe evaluation, not the official
paper's complete five-run reproduction. No class labels are used to train the
EPAGCL encoder; training labels are used only by the downstream classifier.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained EPAGCL encoder with node-level predictions"
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to an EPAGCL encoder .pth checkpoint")
    parser.add_argument("--dataset", default="Cora",
                        help="Dataset name understood by official Load_data.get_dataset")
    parser.add_argument("--data-root", type=Path, default=Path("../datasets"),
                        help="Parent directory containing a folder named after the dataset")
    parser.add_argument("--feat-dim", type=int, default=256,
                        help="Must match encoder training's --feat_dim")
    parser.add_argument("--proj-hidden-dim", type=int, default=256,
                        help="Must match encoder training's --proj_hidden_dim")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Same hyperparameter used to instantiate MyModel")
    parser.add_argument("--eval-epochs", type=int, default=3000,
                        help="Number of epochs for the downstream linear classifier")
    parser.add_argument("--eval-lr", type=float, default=0.01)
    parser.add_argument("--eval-weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for a fixed 10/10/80 train/valid/test split")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--show", type=int, default=20,
                        help="Number of individual test predictions to print")
    parser.add_argument("--compare-raw", action="store_true",
                        help="Also evaluate original node features using the SAME split")
    parser.add_argument("--out-dir", type=Path, default=Path("node_prediction_results"))
    return parser.parse_args()


def get_device(requested: str):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but CUDA is unavailable. Try --device cpu.")
    return torch.device(requested)


def make_splits(n: int, seed: int, device: torch.device):
    """Fixed random 10% training / 10% validation / 80% testing by node.

    All nodes remain present in the graph for transductive encoding. Only the
    training labels reach the downstream classifier's optimization step.
    """
    if n < 20:
        raise ValueError("At least 20 nodes are needed for a 10/10/80 split.")
    rng = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randperm(n, generator=rng).to(device)
    train_n = int(n * 0.1)
    valid_n = int(n * 0.1)
    return ids[:train_n], ids[train_n:train_n + valid_n], ids[train_n + valid_n:]


def train_and_evaluate_linear(features, labels, train_ids, valid_ids, test_ids,
                              classes: int, epochs: int, lr: float, weight_decay: float):
    """Multiclass logistic regression with validation-only model selection.

    Does not inspect test labels or test accuracy while optimizing/selecting
    the classifier. Returns the final test predictions after selection.
    """
    if epochs <= 0:
        raise ValueError("--eval-epochs must be a positive integer")
    if features.ndim != 2 or labels.ndim != 1:
        raise ValueError("Expected features [N, D] and labels [N].")
    classifier = torch.nn.Linear(features.shape[1], classes).to(features.device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    best_valid = -1.0
    best_epoch = -1
    best_state = None

    for epoch in range(1, epochs + 1):
        classifier.train()
        optimizer.zero_grad()
        loss = criterion(classifier(features[train_ids]), labels[train_ids])
        loss.backward()
        optimizer.step()

        classifier.eval()
        with torch.no_grad():
            valid_pred = classifier(features[valid_ids]).argmax(dim=-1)
            valid_acc = (valid_pred == labels[valid_ids]).float().mean().item()
        # IMPORTANT: no test-set evaluation for selection, including on ties.
        if valid_acc > best_valid:
            best_valid = valid_acc
            best_epoch = epoch
            best_state = copy.deepcopy(classifier.state_dict())

    assert best_state is not None
    classifier.load_state_dict(best_state)
    classifier.eval()
    with torch.no_grad():
        probabilities = classifier(features[test_ids]).softmax(dim=-1)
        confidences, predictions = probabilities.max(dim=-1)
        true_labels = labels[test_ids]
        accuracy = (predictions == true_labels).float().mean().item()
    return {
        "accuracy": accuracy,
        "best_valid_accuracy": best_valid,
        "best_epoch": best_epoch,
        "ids": test_ids.cpu().tolist(),
        "truth": true_labels.cpu().tolist(),
        "prediction": predictions.cpu().tolist(),
        "confidence": confidences.cpu().tolist(),
    }


def save_report(results: dict, output_dir: Path, dataset_name: str,
                checkpoint: Path, seed: int, classes: int, n: int,
                train_count: int, valid_count: int, baseline: dict | None):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "test_predictions.csv"
    rows = []
    for node_id, truth, prediction, confidence in zip(
        results["ids"], results["truth"], results["prediction"], results["confidence"]
    ):
        rows.append({
            "node_id": node_id,
            "true_class_id": truth,
            "predicted_class_id": prediction,
            "correct": int(truth == prediction),
            "confidence": f"{confidence:.4f}",
        })
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    confusion = [[0] * classes for _ in range(classes)]
    for row in rows:
        confusion[row["true_class_id"]][row["predicted_class_id"]] += 1
    summary = {
        "dataset": dataset_name,
        "encoder_checkpoint": str(checkpoint.resolve()),
        "seed": seed,
        "num_nodes": n,
        "num_classes": classes,
        "train_count": train_count,
        "validation_count": valid_count,
        "test_count": len(rows),
        "test_accuracy": results["accuracy"],
        "validation_accuracy_at_selected_epoch": results["best_valid_accuracy"],
        "classifier_selected_epoch": results["best_epoch"],
        "confusion_matrix_rows_true_columns_predicted": confusion,
        "raw_feature_baseline_accuracy": None if baseline is None else baseline["accuracy"],
        "note": (
            "This is one reproducible linear-evaluation split, not a five-run paper reproduction. "
            "Model selection for the classifier uses validation labels only. "
            "Classes are integer dataset IDs (not paper titles)."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path, output_dir / "summary.json", rows


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Encoder checkpoint not found: {args.checkpoint}")

    # The two imports below come from the ORIGINAL author's repository.
    # Save this script alongside Model.py and Load_data.py.
    try:
        from Model import MyModel
        from Load_data import get_dataset
    except ImportError as e:
        raise RuntimeError(
            "Put this script inside the official EPAGCL repository (next to "
            "Model.py and Load_data.py), with its dependencies already available."
        ) from e

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = get_device(args.device)

    # Mirror main.py: get_dataset(args.dataset_path, args.dataset), where
    # args.dataset_path is the root plus the dataset name in Arguments.py.
    data = get_dataset(str(args.data_root / args.dataset), args.dataset).to(device)
    labels = data.y.reshape(-1).long()
    if labels.numel() != data.x.shape[0]:
        raise ValueError("Mismatch between node features and label counts")
    if (labels < 0).any():
        raise ValueError("This demo expects labels 0..C-1 without missing labels")
    classes = int(labels.max().item()) + 1

    encoder = MyModel(
        data.num_features, args.feat_dim, args.proj_hidden_dim, args.temperature
    ).to(device)
    try:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except TypeError:  # for older torch versions without weights_only
        state = torch.load(args.checkpoint, map_location=device)
    encoder.load_state_dict(state, strict=True)
    encoder.eval()

    # Important: use the ORIGINAL, unaugmented graph here, matching main.py.
    with torch.no_grad():
        embeddings = encoder(data.x, data.edge_index)[0].detach()
    if embeddings.ndim != 2 or embeddings.shape[0] != labels.shape[0]:
        raise ValueError("Unexpected encoder output shape; check checkpoint and Model.py")

    train_ids, valid_ids, test_ids = make_splits(len(labels), args.seed, device)
    print(f"Dataset: {args.dataset}; device: {device}")
    print(f"Nodes={len(labels)}, classes={classes}, embeddings={tuple(embeddings.shape)}")
    print(f"Train={len(train_ids)}, validation={len(valid_ids)}, test={len(test_ids)}")
    print("Training ONLY a new linear classifier; the EPAGCL encoder stays frozen.")

    results = train_and_evaluate_linear(
        embeddings, labels, train_ids, valid_ids, test_ids,
        classes, args.eval_epochs, args.eval_lr, args.eval_weight_decay
    )
    baseline = None
    if args.compare_raw:
        torch.manual_seed(args.seed + 1)
        baseline = train_and_evaluate_linear(
            data.x.float(), labels, train_ids, valid_ids, test_ids,
            classes, args.eval_epochs, args.eval_lr, args.eval_weight_decay
        )

    csv_path, summary_path, rows = save_report(
        results, args.out_dir, args.dataset, args.checkpoint, args.seed,
        classes, len(labels), len(train_ids), len(valid_ids), baseline
    )

    print(f"\nEPAGCL + linear classifier TEST accuracy: {results['accuracy'] * 100:.2f}%")
    print(f"Best validation accuracy: {results['best_valid_accuracy'] * 100:.2f}% "
          f"(linear epoch {results['best_epoch']})")
    if baseline is not None:
        print(f"Raw node features + SAME linear classifier: {baseline['accuracy'] * 100:.2f}%")
    print("\nIndividual TEST node predictions (class numbers are dataset category IDs):")
    print(f"{'Node':>7} {'True':>7} {'Pred':>7} {'Conf':>8}  Match")
    for row in rows[:max(0, args.show)]:
        match = "YES" if row["correct"] else "NO"
        print(f"{row['node_id']:>7} {row['true_class_id']:>7} "
              f"{row['predicted_class_id']:>7} {row['confidence']:>8}  {match}")
    print(f"\nSaved complete test predictions: {csv_path}")
    print(f"Saved summary and confusion matrix: {summary_path}")


if __name__ == "__main__":
    main()
