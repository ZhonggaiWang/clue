"""A: full-class temporal teacher evidence stable under two image views.

The all-class teacher can propose old, new or background targets. It uses no
GT and adds no trainable module/loss to PTC. View consistency is evidence,
not independence or a calibrated correctness probability.
"""
import copy
import math
import torch
from torch.nn import functional as F


def make_evidence_teacher(student):
    teacher=copy.deepcopy(student).eval().requires_grad_(False)
    return teacher


@torch.no_grad()
def update_evidence_teacher(teacher,student,decay):
    if not 0<=decay<1:raise ValueError('EMA decay must be in [0,1)')
    for target,current in zip(teacher.parameters(),student.parameters()):
        target.mul_(decay).add_(current.detach(),alpha=1-decay)
    for target,current in zip(teacher.buffers(),student.buffers()):
        target.copy_(current)
    teacher.eval()


@torch.no_grad()
def view_probabilities(teacher,images,tags,old_count,mode='crossview',scale=1.25,support_sink=None):
    if teacher.training:raise ValueError('Evidence teacher must be in eval mode')
    if mode not in ['crossview','single']:raise ValueError(mode)
    h,w=images.shape[-2:]
    views=[(images,False)]
    if mode=='crossview':
        size=(max(16,round(h*scale/16)*16),max(16,round(w*scale/16)*16))
        views.append((F.interpolate(images.flip(-1),size,mode='bilinear',align_corners=False),True))
    probabilities=[]
    for image,flipped in views:
        _,logits,features,_=teacher(image)
        if support_sink is not None:
            cam=F.relu(F.conv2d(features,teacher.classifier.get_weight()))
            cam=cam+F.adaptive_max_pool2d(-cam,(1,1))
            cam=cam/(F.adaptive_max_pool2d(cam,(1,1))+1e-5)
            cam=F.interpolate(cam,(h,w),mode='bilinear',align_corners=False)
            support_sink.append(cam.flip(-1) if flipped else cam)
        logits=F.interpolate(logits,(h,w),mode='bilinear',align_corners=False)
        if flipped:logits=logits.flip(-1)
        # Human current-class image tags license new outputs; old classes are
        # never suppressed using unobserved old image labels or old CLS guesses.
        if old_count+1<logits.shape[1]:
            absent=tags[:,old_count:]<=0
            logits[:,old_count+1:]=logits[:,old_count+1:].masked_fill(absent[:,:,None,None],-1e4)
        probabilities.append(logits.softmax(1))
    return torch.stack(probabilities)


@torch.no_grad()
def consensus_evidence(probabilities,hard,valid):
    # Shape V,B,K,H,W. A source must rank the same class first in each view.
    if probabilities.ndim!=5 or probabilities.shape[2]<2:raise ValueError('Expected V,B,K,H,W probabilities')
    values,labels=probabilities.topk(2,dim=2)
    candidate=labels[0,:,0]
    agreement=(labels[:,:,0]==candidate[None]).all(0)
    margin=(values[:,:,0]-values[:,:,1]).amin(0).clamp(0,1)
    domain=valid & (hard>0) & (hard<probabilities.shape[2])
    selected=domain & agreement & (candidate!=hard)
    weight=margin*selected
    return dict(candidate=candidate,weight=weight,selected=weight>0,
                agreement=agreement,margin=margin,domain=domain)


@torch.no_grad()
def qualify_new_source(evidence,view_cams,hard,old_count):
    # Surviving positive evidence for an original current-class label resists
    # its removal, even when the all-class teacher agrees on another class.
    # CAM magnitude is a source-support heuristic, not a calibrated probability.
    support=view_cams.amax(0).gather(1,(hard-1).clamp(0,view_cams.shape[2]-1)[:,None])[:,0].clamp(0,1)
    new=(hard>old_count)&(hard<=view_cams.shape[2])
    factor=torch.where(new,1-support,torch.ones_like(support))
    weight=evidence['weight']*factor
    return dict(evidence,weight=weight,selected=weight>0,original_source_support=support,source_factor=factor)


@torch.no_grad()
def temporal_evidence(teacher,images,tags,hard,valid,old_count,args):
    mode='single' if args.temporal_mode=='single' else 'crossview'
    support=[] if args.temporal_protection=='cam' else None
    p=view_probabilities(teacher,images,tags,old_count,mode,args.temporal_view_scale,support)
    e=consensus_evidence(p,hard,valid)
    return qualify_new_source(e,torch.stack(support),hard,old_count) if support is not None else e


def temporal_correction(logits,hard,valid,evidence,mix=1.):
    if not 0<=mix<=1:raise ValueError('Target mixture must be in [0,1]')
    c=evidence['candidate'];w=evidence['weight'].detach()*mix
    domain=valid & (hard>0) & (hard<logits.shape[1]) & (c>=0) & (c<logits.shape[1])
    w=w*domain
    current=logits.gather(1,hard.clamp(0,logits.shape[1]-1)[:,None])[:,0]
    proposed=logits.gather(1,c.clamp(0,logits.shape[1]-1)[:,None])[:,0]
    count=(valid & (hard!=255)).sum().clamp_min(1)
    # BCE terms cancel exactly for a normalized two-class soft target mixture.
    return ((current-proposed)*w).sum()/count
