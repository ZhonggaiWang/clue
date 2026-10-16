"""Independent experimental background repair; no training integration yet.

Labels are proposed only inside expanded existing new-class seed boxes.
The module accepts no dense ground truth and changes no PTC calculation.
"""
import torch
from torch.nn import functional as F
from utils.camutils import multi_scale_cam2


@torch.no_grad()
def propose_background(model, images, tags, hard, cams, valid, old_count,
                       cam_scales, threshold=.75):
    height, width = images.shape[-2:]
    candidate = torch.zeros_like(hard)
    local_score = torch.full_like(hard, -1., dtype=images.dtype)
    agree = torch.zeros_like(valid)
    was_training = model.training
    model.eval()
    try:
        for b in range(images.shape[0]):
            for c in range(old_count + 1, tags.shape[1] + 1):
                if tags[b, c - 1] <= 0:
                    continue
                region = (hard[b] == c) & valid[b]
                if not region.any():
                    continue
                ys, xs = region.nonzero(as_tuple=True)
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                dy = max(1, int((y1 - y0) * .25))
                dx = max(1, int((x1 - x0) * .25))
                y0, y1 = max(0, y0-dy), min(height, y1+dy)
                x0, x1 = max(0, x0-dx), min(width, x1+dx)
                view = F.interpolate(images[b:b+1, :, y0:y1, x0:x1],
                                     (height, width), mode='bilinear', align_corners=False)
                local, _ = multi_scale_cam2(model, view, cam_scales)
                local = F.interpolate(local, (y1-y0, x1-x0), mode='bilinear',
                                      align_corners=False) * tags[b:b+1, :, None, None]
                score = local[0, c-1]
                eligible = (valid[b, y0:y1, x0:x1]
                            & (hard[b, y0:y1, x0:x1] == 0)
                            & (score > local_score[b, y0:y1, x0:x1]))
                local_score[b, y0:y1, x0:x1][eligible] = score[eligible]
                candidate[b, y0:y1, x0:x1][eligible] = c
                agree[b, y0:y1, x0:x1][eligible] = (local.argmax(1)[0][eligible] + 1 == c)
    finally:
        model.train(was_training)
    eligible = valid & (hard == 0) & (candidate > old_count)
    accepted = eligible & agree & (local_score >= threshold)
    original_score = cams.gather(1, (candidate.clamp(1, cams.shape[1])-1)[:, None])[:, 0]
    return dict(candidate=candidate, eligible=eligible, accepted=accepted,
                local_score=local_score, original_score=original_score, agree=agree)


@torch.no_grad()
def select_recovery(proposals, mode='local'):
    if mode == 'local':
        return proposals['accepted'].clone()
    if mode != 'confidence':
        raise ValueError(mode)
    selected = torch.zeros_like(proposals['accepted'])
    for b in range(selected.shape[0]):
        eligible = proposals['eligible'][b]
        labels = proposals['candidate'][b]
        for c in labels[eligible].unique().tolist():
            mask = eligible & (labels == c)
            index = mask.flatten().nonzero().flatten()
            k = int((mask & proposals['accepted'][b]).sum())
            score = proposals['original_score'][b].flatten()[index]
            order = torch.argsort(score, descending=True, stable=True)
            selected[b].view(-1)[index[order[:k]]] = True
    return selected


def recovery_correction(logits, hard, valid, proposals, selected):
    """Exact one-hot BCE target replacement, retaining original denominator.

    softplus terms cancel: BCE(new)-BCE(background) = z_background-z_new.
    Existing foreground, ignore, and padding pixels are protected explicitly.
    """
    candidate = proposals['candidate']
    changed = selected & proposals['eligible'] & valid & (hard == 0)
    target_logits = logits.gather(1, candidate.clamp(0, logits.shape[1]-1)[:, None])[:, 0]
    count = (valid & (hard != 255)).sum().clamp_min(1)
    return ((logits[:, 0] - target_logits) * changed).sum() / count
