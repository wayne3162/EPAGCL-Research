"""Fixed-split, validation-only linear probing for the EPAGCL extension.

The held-out test labels are never used by fit_linear_probe. Only
score_test_with_selected_probe accesses the test split, after encoder selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass
class ProbeResult:
    val_accuracy: float
    best_epoch: int
    classifier_state: dict


class LinearClassifier(nn.Module):
    """Identical architecture to the original single-FC classifier."""

    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def make_split(num_nodes: int, seed: int) -> dict:
    """Exactly the original 10/10/80 floor-count policy, with a fixed seed."""
    if num_nodes < 10:
        raise ValueError("At least 10 nodes are required for a 10/10/80 split")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    idx = torch.randperm(num_nodes, generator=generator)
    n_train = int(0.1 * num_nodes)
    n_val = int(0.1 * num_nodes)
    result = {
        "train": idx[:n_train].clone(),
        "valid": idx[n_train:n_train + n_val].clone(),
        "test": idx[n_train + n_val:].clone(),
        "num_nodes": int(num_nodes),
        "seed": int(seed),
    }
    validate_split(result, num_nodes)
    return result


def validate_split(split: dict, num_nodes: int) -> None:
    indices = [split[k].long().cpu() for k in ("train", "valid", "test")]
    all_indices = torch.cat(indices)
    if (all_indices.numel() != num_nodes or
            torch.unique(all_indices).numel() != num_nodes or
            int(all_indices.min()) != 0 or
            int(all_indices.max()) != num_nodes - 1):
        raise ValueError("Split must partition all nodes without duplication")


def load_or_create_split(path: Path, num_nodes: int, seed: int) -> dict:
    """Shared across baseline and VR for the same dataset/seed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        split = torch.load(path, map_location="cpu", weights_only=True)
        if split.get("num_nodes") != num_nodes or split.get("seed") != seed:
            raise ValueError(f"Split metadata mismatch: {path}")
        validate_split(split, num_nodes)
        return split
    split = make_split(num_nodes, seed)
    torch.save(split, path)
    return split


def _class_accuracy(classifier: nn.Module, features: torch.Tensor, labels: torch.Tensor) -> float:
    with torch.no_grad():
        prediction = classifier(features).argmax(dim=1)
        return 100.0 * (prediction == labels).float().mean().item()


def _optimizer(classifier: nn.Module, name: str, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    if name.lower() == "adam":
        return torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    if name.lower() == "sgd":
        return torch.optim.SGD(classifier.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"Unsupported eval optimizer: {name}")


def fit_linear_probe(
    z: torch.Tensor,
    y: torch.Tensor,
    split: dict,
    *,
    eval_epochs: int,
    eval_optimizer: str,
    eval_lr: float,
    eval_weight_decay: float,
    eval_seed: int,
) -> ProbeResult:
    """Train/select a probe on train/valid only; NEVER inspect test labels."""
    device = z.device
    z = z.detach()
    y = y.view(-1)
    train_idx = split["train"].to(device)
    valid_idx = split["valid"].to(device)
    train_x, train_y = z.index_select(0, train_idx), y.index_select(0, train_idx)
    valid_x, valid_y = z.index_select(0, valid_idx), y.index_select(0, valid_idx)

    # Isolate probe RNG from augmentation/model-training RNG. The same
    # seed/encoder epoch gets the same classifier initialization per method.
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(eval_seed))
        # Never inspect held-out test labels, even to infer class count.
        known_labels = torch.cat((train_y, valid_y))
        num_classes = int(known_labels.max().item()) + 1
        classifier = LinearClassifier(z.size(1), num_classes).to(device)
        optimizer = _optimizer(classifier, eval_optimizer, eval_lr, eval_weight_decay)
        criterion = nn.CrossEntropyLoss()
        best_val = -1.0
        best_epoch = -1
        best_state = None
        for epoch in range(int(eval_epochs)):
            classifier.train()
            optimizer.zero_grad()
            train_loss = criterion(classifier(train_x), train_y)
            train_loss.backward()
            optimizer.step()
            classifier.eval()
            val_acc = _class_accuracy(classifier, valid_x, valid_y)
            # Strictly validation-only, first occurrence wins ties.
            if val_acc > best_val:
                best_val = val_acc
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}
        if best_state is None:
            raise RuntimeError("No linear probe epochs were performed")
    return ProbeResult(best_val, best_epoch, best_state)


def score_test_with_selected_probe(
    z: torch.Tensor,
    y: torch.Tensor,
    split: dict,
    classifier_state: dict,
) -> float:
    """Call ONCE after all encoder checkpoint selection has finished."""
    device = z.device
    num_classes = int(classifier_state["fc.weight"].shape[0])
    classifier = LinearClassifier(z.size(1), num_classes).to(device)
    classifier.load_state_dict(classifier_state)
    classifier.eval()
    with torch.no_grad():
        test_idx = split["test"].to(device)
        test_features = z.detach().index_select(0, test_idx)
        test_labels = y.view(-1).index_select(0, test_idx)
        return _class_accuracy(classifier, test_features, test_labels)
