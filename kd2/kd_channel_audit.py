"""Offline Bernoulli old-channel quality diagnostics; never used by training."""
import torch
COLUMNS=['channel_elements','gt_old_positive_channels','confident_positive_channels',
         'correct_positive_channels','wrong_positive_channels','confident_negative_channels',
         'correct_negative_channels','wrong_negative_channels','new_gt_pixels',
         'new_gt_pixels_all_old_confident_negative']
@torch.no_grad()
def binary_quality_counts(probability,gt,pixel_mask,old_count,total,threshold=.75):
    if probability.shape[1]!=old_count or probability.shape[0]!=gt.shape[0] or probability.shape[-2:]!=gt.shape[-2:]:
        raise ValueError('Probability and label dimensions do not match')
    if not .5<threshold<1:
        raise ValueError('Confidence threshold must exceed .5')
    ids=torch.arange(1,old_count+1,device=gt.device)[None,:,None,None]
    truth=gt[:,None]==ids
    positive=probability>=threshold
    negative=probability<=1-threshold
    selected=pixel_mask[:,None]
    new_gt=(gt>old_count)&(gt<=total)
    vals=[pixel_mask.sum()*old_count,
          (selected&truth).sum(),
          (selected&positive).sum(),
          (selected&positive&truth).sum(),
          (selected&positive&~truth).sum(),
          (selected&negative).sum(),
          (selected&negative&~truth).sum(),
          (selected&negative&truth).sum(),
          (pixel_mask&new_gt).sum(),
          (pixel_mask&new_gt&negative.all(1)).sum()]
    return torch.stack(vals).to(torch.int64)
