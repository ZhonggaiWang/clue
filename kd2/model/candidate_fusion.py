"""Detached paired classifier-cosine arbitration on old/new foreground conflicts."""
import torch
from torch.nn import functional as F

@torch.no_grad()
def candidate_evidence(features, weights, old_pixels, new_pixels, hard, valid, tags, old_count, margin):
    if not 0 <= margin <= 2:
        raise ValueError("Cosine margin must be between 0 and 2")
    old_pixels = old_pixels.long()
    new_pixels = new_pixels.long()
    scores = F.conv2d(F.normalize(features.detach(), dim=1),
                      F.normalize(weights.detach(), dim=1))
    scores = F.interpolate(scores, size=hard.shape[-2:], mode='bilinear', align_corners=False)
    channels = scores.shape[1]
    old_score = scores.gather(1, (old_pixels - 1).clamp(0, channels - 1)[:, None])[:, 0]
    new_score = scores.gather(1, (new_pixels - 1).clamp(0, channels - 1)[:, None])[:, 0]
    licensed = tags.gather(1, (new_pixels - 1).clamp(0, channels - 1).flatten(1)).reshape_as(hard) > 0
    conflict = (valid & (hard != 255) & (old_pixels > 0) & (old_pixels <= old_count)
                & (new_pixels > old_count) & (new_pixels <= channels)
                & (hard == new_pixels) & licensed)
    gap = old_score - new_score
    to_old = conflict & (gap >= margin) & (old_score >= 0)
    new_strong = conflict & (-gap >= margin) & (new_score >= 0)
    return dict(conflict=conflict, to_old=to_old, new_strong=new_strong,
                gap=gap, old_score=old_score, new_score=new_score, old_pixels=old_pixels)

def candidate_correction(logits, hard, valid, evidence):
    # BCE(old)-BCE(new): the softplus terms cancel exactly; denominator is unchanged.
    count = (valid & (hard != 255)).sum().clamp_min(1)
    channels = logits.shape[1]
    old_logit = logits.gather(1, evidence['old_pixels'].clamp(0, channels-1)[:, None])[:, 0]
    new_logit = logits.gather(1, hard.clamp(0, channels-1)[:, None])[:, 0]
    return ((new_logit - old_logit) * evidence['to_old']).sum() / count
