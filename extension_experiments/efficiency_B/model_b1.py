"""Independent B1 loss optimization. Do not edit the original EPAGCL Model.py.

The only algorithmic change is to normalize z1 and z2 once in loss() before
entering the original per-batch loss loops. Both directional randperm calls,
positive/negative samples, denominator, temperature and reduction are retained.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from Model import MyModel


class MyModelB1(MyModel):
    """Drop-in replacement for MyModel, changing only the loss implementation."""

    def loss(self, z1, z2, batch_compute=False, batch_size=256):
        # F.normalize uses the same defaults as the original MyModel.cal_sim:
        # p=2, dim=1, eps=1e-12. Do not detach; gradients must flow through it.
        n1 = F.normalize(z1)
        n2 = F.normalize(z2)
        l1 = self._cal_loss_pre_normalized(n1, n2, batch_compute, batch_size)
        l2 = self._cal_loss_pre_normalized(n2, n1, batch_compute, batch_size)
        return ((l1 + l2) * 0.5).mean()

    def _cal_loss_pre_normalized(self, z1, z2, batch_compute=False, batch_size=256):
        # Keep the index creation in precisely the same places as Model.py.
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        if not batch_compute:
            indices = torch.arange(0, num_nodes).to(device)
        else:
            # Each direction calls this method separately: TWO distinct draws.
            indices = torch.randperm(num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            if not batch_compute:
                refl_sim = torch.exp(torch.mm(z1[mask], z1.t()) / self.t)
                between_sim = torch.exp(torch.mm(z1[mask], z2.t()) / self.t)
                losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                         / (refl_sim.sum(1) + between_sim.sum(1)
                                            - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
            else:
                refl_sim = torch.exp(torch.mm(z1[mask], z1[mask].t()) / self.t)
                between_sim = torch.exp(torch.mm(z1[mask], z2[mask].t()) / self.t)
                losses.append(-torch.log(between_sim.diag()
                                         / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())))
        return torch.cat(losses)
