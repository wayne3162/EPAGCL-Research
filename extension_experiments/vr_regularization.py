"""An experimental variance-floor penalty on EPAGCL encoder features."""

import torch


def variance_floor_loss(out1: torch.Tensor, out2: torch.Tensor, gamma: float, eps: float = 1e-8) -> torch.Tensor:
    """Average relu(gamma - std) over features and both graph views.

    Note: this cannot guarantee recovery from an already exact collapse.
    Lambda/gamma must be chosen without looking at held-out test accuracy.
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if out1.ndim != 2 or out2.ndim != 2 or out1.shape != out2.shape:
        raise ValueError("out1 and out2 must be matching [nodes, features] tensors")
    std1 = torch.sqrt(out1.var(dim=0, unbiased=False) + eps)
    std2 = torch.sqrt(out2.var(dim=0, unbiased=False) + eps)
    return 0.5 * (torch.relu(float(gamma) - std1).mean() + torch.relu(float(gamma) - std2).mean())


def shadow_should_stop(loss_by_epoch: dict, threshold: float = 6.22) -> bool | None:
    """Historical Coauthor-Physics candidate, monitoring-only.

    Returns None until losses at epochs 50, 75, and 100 exist.
    This is calibrated on older runs, not an independently verified rule.
    """
    checkpoints = (50, 75, 100)
    if any(e not in loss_by_epoch for e in checkpoints):
        return None
    return all(float(loss_by_epoch[e]) >= float(threshold) for e in checkpoints)
