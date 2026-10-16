"""Channel-level old-foreground KD. Masks use predictions/evidence only, never GT."""
import torch
from torch.nn import functional as F

@torch.no_grad()
def relation_selection(teacher_logits, old_pixels, masks, old_count, mode, threshold=.75):
    if mode not in ['counter','relations'] or old_count<1 or not .5<threshold<1:
        raise ValueError('Invalid channel KD selection')
    probability=teacher_logits.detach()[:,1:old_count+1].sigmoid()
    eligible=masks['eligible'].detach()[:,None].expand_as(probability)
    if mode=='counter':
        # Keep ordinary soft knowledge except positive old-class relations contradicted by supported new evidence.
        rejected=masks['new_supported'].detach()[:,None] & (probability>.5)
        selected=eligible & ~rejected
    else:
        classes=torch.arange(1,old_count+1,device=probability.device)[None,:,None,None]
        positive=masks['selected'].detach()[:,None] & (old_pixels.detach()[:,None]==classes) & (probability>=threshold)
        negative=probability<=1-threshold
        selected=eligible & (positive|negative)
    return selected, probability

def channel_foreground_kd(student_logits, teacher_logits, old_pixels, masks, old_count,
                          mode, temperature=2., threshold=.75):
    if temperature<=0: raise ValueError('Temperature must be positive')
    selected,raw_probability=relation_selection(teacher_logits,old_pixels,masks,old_count,mode,threshold)
    eligible=masks['eligible'].detach()
    student=student_logits[:,1:old_count+1]/temperature
    teacher=teacher_logits.detach()[:,1:old_count+1]/temperature
    q=teacher.sigmoid()
    kl=(q*(F.logsigmoid(teacher)-F.logsigmoid(student))+
        (1-q)*(F.logsigmoid(-teacher)-F.logsigmoid(-student)))*temperature**2
    pixel_count=eligible.sum().clamp_min(1)
    selected_count=selected.sum()
    eligible_channels=eligible.sum()*old_count
    # Equal summed channel weights when nonempty; this is not equality of KL or gradient mass.
    scale=(eligible_channels.to(kl.dtype)/selected_count.clamp_min(1)) if mode=='relations' else kl.new_ones(())
    weighted_sum=(kl*selected).sum()*scale
    total=weighted_sum/pixel_count
    full_per_pixel=kl.detach().sum(1)
    raw_sum=(kl.detach()*selected).sum()
    full_sum=(full_per_pixel*eligible).sum()
    positive=raw_probability>.5
    confident_negative=raw_probability<=1-threshold
    uncertain=(raw_probability>1-threshold)&(raw_probability<threshold)
    stats={
      'kd_pixels':selected.any(1).sum().to(kl.dtype),
      'kd_selected_channel_sum':selected_count.to(kl.dtype),
      'kd_eligible_channel_sum':eligible_channels.to(kl.dtype),
      'kd_channel_weight_sum':selected_count.to(kl.dtype)*scale,
      'kd_positive_channel_sum':(selected&positive).sum().to(kl.dtype),
      'kd_confident_negative_channel_sum':(selected&confident_negative).sum().to(kl.dtype),
      'kd_uncertain_channel_sum':(selected&uncertain).sum().to(kl.dtype),
      'kd_rejected_positive_channel_sum':(eligible[:,None]&positive&~selected).sum().to(kl.dtype),
      'kd_raw_selected_kl_sum':raw_sum,
      'kd_kl_sum':weighted_sum.detach(),
      'kd_plain_kl_sum':full_sum,
      'kd_weight_scale':scale.detach(),
    }
    return total,full_per_pixel,stats
