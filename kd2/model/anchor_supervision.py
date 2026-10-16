"""A: foreground supervision from seed evidence stable across disjoint subsets.

No image input, model, decoder prediction, crop or encoder call is accepted.
Scores and weights are detached evidence, not calibrated probabilities.
"""
import torch
from torch.nn import functional as F

@torch.no_grad()
def trusted_seeds(old_logits, cams, cams_aux, tags, hard, valid, old_count,
                  seed_threshold=.9):
    size=hard.shape[-2:]
    prob=F.interpolate(old_logits,size,mode='bilinear',align_corners=False).softmax(1)
    score,label=prob.max(1)
    seeds=torch.zeros_like(hard)
    old=valid & (hard>0) & (hard<=old_count) & (label==hard) & (score>=seed_threshold)
    seeds[old]=hard[old]
    new_cams=cams[:,old_count:]*tags[:,old_count:,None,None]
    new_aux=cams_aux[:,old_count:]*tags[:,old_count:,None,None]
    if new_cams.shape[1]:
        ns,ni=new_cams.max(1);auxs,auxi=new_aux.max(1)
        new=valid & (hard>old_count) & (hard!=255) & (hard==ni+old_count+1)
        new &= (ni==auxi) & (ns>=seed_threshold) & (auxs>=seed_threshold)
        new &= (tags[:,old_count:].sum(1)>0)[:,None,None]
        seeds[new]=hard[new]
    return seeds

@torch.no_grad()
def seed_evidence(features, seeds, hard, valid, class_count, mode='stable',
                  support=.7, min_seeds=3, topk=5, radius=2, old_count=None, scope='all_seen'):
    if mode not in ['stable','pooled','hard']:raise ValueError(mode)
    b,_,h,w=features.shape;n=h*w
    feat=F.normalize(features.detach().float().flatten(2).transpose(1,2),dim=-1,eps=1e-8)
    vg=F.adaptive_avg_pool2d(valid[:,None].float(),(h,w))[:,0]>=.999
    coords=torch.stack(torch.meshgrid(torch.arange(h,device=features.device),torch.arange(w,device=features.device),indexing='ij'),-1).reshape(n,2)
    far=(coords[:,None]-coords[None]).abs().amax(-1)>radius
    scores=torch.full((b,2,class_count,h,w),-2.,device=features.device)
    for bi in range(b):
        sim=feat[bi]@feat[bi].T
        for c in range(1,class_count):
            occupied=F.adaptive_avg_pool2d((seeds[bi:bi+1,None]==c).float(),(h,w))[0,0]>=.5
            idx=(occupied & vg[bi]).flatten().nonzero().flatten()
            if idx.numel()<2*min_seeds:continue
            # Own generator: no changes to training RNG, crops, dropout or shuffling.
            generator=torch.Generator(device=idx.device).manual_seed(104729*c+17)
            order=idx[torch.randperm(idx.numel(),generator=generator,device=idx.device)]
            groups=[order[::2],order[1::2]] if mode!='pooled' else [order,order]
            for group,anchors in enumerate(groups):
                enough=far[:,anchors].sum(1)>=min_seeds
                values=sim[:,anchors].masked_fill(~far[:,anchors],-2.)
                best=values.topk(min(topk,anchors.numel()),dim=1).values
                available=best>-1.5
                mean=best.masked_fill(~available,0).sum(1)/available.sum(1).clamp_min(1)
                scores[bi,group,c]=torch.where(enough & vg[bi].flatten(),mean,-2.).reshape(h,w)
    comparable=(scores>-1.5).all(1)
    # Both seed subsets must be available for a class to enter the competition.
    evidence=((scores-support)/(1-support)).clamp(0,1)
    evidence=evidence.masked_fill(~comparable[:,None],-2.)
    candidate=evidence.mean(1).argmax(1)
    chosen=evidence.gather(2,candidate[:,None,None].expand(-1,2,1,-1,-1)).squeeze(2)
    alternatives=evidence.scatter(2,candidate[:,None,None].expand(-1,2,1,-1,-1),-2.).amax(2)
    # Worst-subset advantage is small for weak, tied or unstable evidence.
    weight=(chosen-alternatives).amin(1).clamp(0,1)
    weight*=comparable.sum(1)>=2
    size=hard.shape[-2:]
    candidate=F.interpolate(candidate[:,None].float(),size,mode='nearest')[:,0].long()
    weight=F.interpolate(weight[:,None],size,mode='nearest')[:,0]
    original_available=F.interpolate(comparable.float(),size,mode='nearest').gather(1,hard.clamp(0,class_count-1)[:,None])[:,0].bool()
    identifiable=valid & (hard>0)&(hard<class_count)&original_available
    eligible=identifiable & (candidate>0)&(candidate!=hard)&(seeds==0)
    weight*=eligible
    if scope not in ['licensed','all_seen']:raise ValueError(scope)
    if scope=='licensed':
        if old_count is None:raise ValueError('Source competence requires known old class count')
        # Closed-set teacher evidence cannot establish that a novel label is old.
        licensed=(candidate>old_count)|((hard<=old_count)&(candidate<=old_count))
        weight*=licensed
    if mode=='hard':weight=(weight>0).float()
    return dict(candidate=candidate,weight=weight,selected=weight>0,seeds=seeds,
                identifiable=identifiable,scores=scores)

@torch.no_grad()
def anchor_evidence(features, old_seg, cams, cams_aux, tags, hard, valid, old_count,args):
    seeds=trusted_seeds(old_seg,cams,cams_aux,tags,hard,valid,old_count,args.anchor_seed_threshold)
    return seed_evidence(features,seeds,hard,valid,tags.shape[1]+1,args.anchor_mode,
        args.anchor_support,args.anchor_min_seeds,args.anchor_topk,args.anchor_radius,old_count,args.anchor_scope)

def anchor_correction(logits,hard,valid,evidence,mix=1.):
    if not 0<=mix<=1:raise ValueError(mix)
    candidate=evidence['candidate'];weight=evidence['weight']*mix
    weight=weight*valid*(hard>0)*(hard<logits.shape[1])*(candidate>0)*(candidate<logits.shape[1])
    current=logits.gather(1,hard.clamp(0,logits.shape[1]-1)[:,None])[:,0]
    proposed=logits.gather(1,candidate.clamp(0,logits.shape[1]-1)[:,None])[:,0]
    denominator=(valid & (hard!=255)).sum().clamp_min(1)
    # Exact BCE difference for (1-weight)*onehot(original)+weight*onehot(candidate).
    return ((current-proposed)*weight).sum()/denominator
