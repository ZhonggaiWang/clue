"""SASA PTC and per-pixel one-hot BCE segmentation losses."""
import torch
from torch.nn import functional as F

def get_masked_ptc_loss(inputs, mask):
    b, c, h, w = inputs.shape

    inputs = inputs.reshape(b, c, h*w)

    def cos_sim(x):
        x = F.normalize(x, p=2, dim=1, eps=1e-8)
        cos_sim = torch.matmul(x.transpose(1,2), x)
        return torch.abs(cos_sim)

    inputs_cos = cos_sim(inputs)

    pos_mask = mask == 1
    neg_mask = mask == 0
    loss = 0.5*(1 - torch.sum(pos_mask * inputs_cos) / (pos_mask.sum()+1)) + 0.5 * torch.sum(neg_mask * inputs_cos) / (neg_mask.sum()+1)
    return loss


def get_seg_loss(pred, label, ignore_index=255):
    ignore_mask = (label != ignore_index).float().unsqueeze(1)  # [B, 1, H, W]

    # one-hot label
    num_classes = pred.size(1)
    one_hot_label = F.one_hot(torch.clamp(label, max=num_classes - 1), num_classes=num_classes).permute(0, 3, 1,
                                                                                                         2).float()
    loss = F.binary_cross_entropy_with_logits(pred, one_hot_label, reduction='none')
    loss = (loss * ignore_mask).sum() / ignore_mask.sum().clamp_min(1)

    return loss
