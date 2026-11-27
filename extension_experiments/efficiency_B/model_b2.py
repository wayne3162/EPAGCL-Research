"""B2: group 2/4/8 independent B1 contrastive batches into batched matmuls.

The Model.py baseline, the B1 pre-normalization, and the VR objective are unchanged.
IMPORTANT: each directional loss calls torch.randperm() separately, just like B0/B1.
Within each batch, only that batch's original positive and negative nodes occur.
The final partial batch is computed separately, with no padding or fake negatives.
"""
from __future__ import annotations

import torch

from model_b1 import MyModelB1


class MyModelB2(MyModelB1):
    def __init__(self, *args, groups_per_call: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        if groups_per_call not in (2, 4, 8):
            raise ValueError("groups_per_call must be 2, 4, or 8")
        self.groups_per_call = groups_per_call

    def _cal_loss_pre_normalized(self, z1, z2, batch_compute=False, batch_size=256):
        if not batch_compute:
            # B1 already implements the exact full-negative (non-batch) behavior.
            return super()._cal_loss_pre_normalized(z1, z2, batch_compute, batch_size)

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        num_nodes = z1.size(0)
        if num_nodes == 0:
            raise ValueError("empty z1 not supported by original MyModel.loss")
        if z2.size(0) != num_nodes:
            raise ValueError("z1 and z2 must contain the same number of nodes")

        # Exactly one randperm per direction; MyModelB1.loss calls this twice.
        # Do not replace this with a shared permutation.
        indices = torch.randperm(num_nodes).to(z1.device)
        full_batches, tail_nodes = divmod(num_nodes, batch_size)
        losses = []

        # One bmm handles g original independent BxB similarity matrices.
        # This does NOT compare nodes across different original batches.
        for first in range(0, full_batches, self.groups_per_call):
            g = min(self.groups_per_call, full_batches - first)
            idx = indices[first * batch_size:(first + g) * batch_size].reshape(g, batch_size)
            a = z1[idx]  # [g, batch_size, dimensions]
            b = z2[idx]
            refl_sim = torch.exp(torch.bmm(a, a.transpose(1, 2)) / self.t)
            between_sim = torch.exp(torch.bmm(a, b.transpose(1, 2)) / self.t)

            refl_diag = refl_sim.diagonal(dim1=1, dim2=2)
            positive = between_sim.diagonal(dim1=1, dim2=2)
            denominator = refl_sim.sum(dim=2) + between_sim.sum(dim=2) - refl_diag
            losses.append((-torch.log(positive / denominator)).reshape(-1))

        if tail_nodes:
            # No padded candidates: preserve original final-batch length exactly.
            tail = indices[full_batches * batch_size:]
            a = z1[tail]
            b = z2[tail]
            refl_sim = torch.exp(torch.mm(a, a.t()) / self.t)
            between_sim = torch.exp(torch.mm(a, b.t()) / self.t)
            positive = between_sim.diag()
            refl_diag = refl_sim.diag()
            denominator = refl_sim.sum(dim=1) + between_sim.sum(dim=1) - refl_diag
            losses.append(-torch.log(positive / denominator))

        return torch.cat(losses)
