from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMA:
    """
    Exponential Moving Average teacher.
    teacher_params = decay * teacher_params + (1-decay) * student_params
    """
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        msd = model.state_dict()
        esd = self.ema.state_dict()
        for k, v in esd.items():
            if k in msd:
                model_v = msd[k].detach()
                if torch.is_floating_point(v):
                    v.mul_(d).add_(model_v, alpha=1.0 - d)
                else:
                    v.copy_(model_v)

    def to(self, device: str | torch.device) -> EMA:
        self.ema.to(device)
        return self
