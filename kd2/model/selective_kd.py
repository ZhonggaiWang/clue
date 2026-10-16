"""Old-foreground Bernoulli KD, with detached evidence masks and no extra encode."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def kd_evidence(old_logits, old_pixels, target, valid, paired, recovered, old_count,
                teacher_threshold=0.75):
    """Partition eligible pixels into protected old, supported new and unresolved."""
    if not 0 < teacher_threshold < 1:
        raise ValueError('Teacher threshold must lie in (0,1)')
    eligible = valid & (target != 255)
    old_fg = (old_pixels > 0) & (old_pixels <= old_count)
    teacher_score = old_logits.sigmoid().gather(1, old_pixels.long()[:,None])[:,0]
    confident = eligible & old_fg & (teacher_score >= teacher_threshold)
    new_candidate = eligible & (target > old_count) & (target < 255)
    conflict = paired['conflict'] & eligible
    to_old = paired['to_old'] & eligible
    old_supported = confident & (paired['old_score'] >= 0) & (
        (old_fg & ~new_candidate & ~conflict) | to_old)
    new_supported = eligible & (recovered | (new_candidate & ~conflict) | paired['new_strong'])
    ambiguous = conflict & ~to_old & ~paired['new_strong']
    # C's accepted new pixels override eligibility for distillation.
    protected = old_supported & ~new_supported & ~ambiguous
    unresolved = eligible & ~protected & ~new_supported
    return dict(eligible=eligible, selected=protected, new_supported=new_supported,
                unresolved=unresolved, ambiguous=ambiguous, conflict=conflict,
                old_conflict=protected & conflict, teacher_confident=confident)


def old_foreground_kd(student_logits, teacher_logits, selected, eligible, old_count,
                      temperature=2.0):
    """Sum Bernoulli KL over OLD FOREGROUND channels, divided by all valid pixels.

    Neither background nor new output channels receive a direct KD gradient.
    Same denominator gives the same per-pixel weight for plain/selective variants.
    """
    if old_count < 1 or temperature <= 0:
        raise ValueError('Invalid old count or temperature')
    student = student_logits[:,1:old_count+1] / temperature
    teacher = teacher_logits.detach()[:,1:old_count+1] / temperature
    probability = teacher.sigmoid()
    kl = (probability * (F.logsigmoid(teacher)-F.logsigmoid(student))
          + (1-probability)*(F.logsigmoid(-teacher)-F.logsigmoid(-student)))
    per_pixel = kl.sum(1) * temperature**2
    selected = selected.detach() & eligible.detach()
    count = eligible.sum().clamp_min(1)
    total = (per_pixel*selected).sum()/count
    return total, per_pixel.detach()
