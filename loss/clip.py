import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, pred, target, reduction="mean"):
        """compute the clip loss within a batch

        Args:
            pred (Tensor): (b, l, d)
            target (Tensor): (b, l, d)

        Returns:
            Tensor: clip loss
        """
        b, l, d = pred.shape
        device = pred.device

        pred = F.normalize(pred, p=2, dim=-1)
        target = F.normalize(target, p=2, dim=-1).transpose(1, 2)
        logits = pred @ target * self.logit_scale
        labels = torch.arange(l)[None, ...].repeat(b, 1).to(device)
        loss1 = F.cross_entropy(logits, labels, reduction=reduction)
        loss2 = F.cross_entropy(logits.transpose(1, 2), labels, reduction=reduction)
        loss = (loss1 + loss2) / 2

        return loss
