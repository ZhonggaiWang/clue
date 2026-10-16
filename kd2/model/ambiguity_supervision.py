"""A revision: retract incompatible binary constraints without promoting a winner.

Evidence comes from the unchanged v7 teacher. This module changes only what
conflict evidence authorizes; baseline PTC/CLS/SEG and C stay intact.
"""
import torch
from torch.nn import functional as F

def ambiguity_correction(logits,hard,valid,evidence,mix=1.,mode='pair_abstain'):
    if not 0<=mix<=1:raise ValueError('Mixture must be in [0,1]')
    if mode not in ['pair_abstain','pixel_abstain']:raise ValueError(mode)
    candidate=evidence['candidate'].detach()
    domain=valid & (hard>0) & (hard<logits.shape[1]) & (candidate>=0) & (candidate<logits.shape[1]) & (candidate!=hard)
    weight=evidence['weight'].detach()*mix*domain
    source_logit=logits.gather(1,hard.clamp(0,logits.shape[1]-1)[:,None])[:,0]
    candidate_logit=logits.gather(1,candidate.clamp(0,logits.shape[1]-1)[:,None])[:,0]
    if mode=='pair_abstain':
        # Remove only source-positive and alternative-negative constraints.
        # Remaining class-wise BCE terms are unchanged; no new positive target.
        withdrawn=F.softplus(-source_logit)+F.softplus(candidate_logit)
    else:
        # Matched control: discard the entire pixel's original segmentation BCE.
        withdrawn=F.softplus(logits).sum(1)-source_logit
    count=(valid & (hard!=255)).sum().clamp_min(1)
    return -(weight*withdrawn).sum()/count
