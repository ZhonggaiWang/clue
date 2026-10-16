"""Region-reference conflict coordination. No GT, image, model or encoder input."""
import torch
from torch.nn import functional as F


def _components(mask):
    """Four-connected components on a small feature grid; deterministic."""
    h, w = mask.shape
    remaining = set(mask.flatten().nonzero().flatten().cpu().tolist())
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, group = [start], []
        while stack:
            i = stack.pop()
            group.append(i)
            y, x = divmod(i, w)
            for yy, xx in [(y-1,x),(y+1,x),(y,x-1),(y,x+1)]:
                j = yy*w+xx
                if 0 <= yy < h and 0 <= xx < w and j in remaining:
                    remaining.remove(j)
                    stack.append(j)
        groups.append(sorted(group))
    return groups


@torch.no_grad()
def region_evidence(features, old_logits, cams, cams_aux, tags, hard, valid,
                    old_count, recovery=None, seed_threshold=.9,
                    support=.7, margin=.05, min_region=2, max_regions=4,
                    reference_mode='regions', exclude_new_from_old=True, batch_fallback=True):
    """Use detached already-encoded batch features; local references take precedence.

    Foreground conflicts only; baseline-background C targets remain unchanged.
    Missing local classes may use other images in the same per-GPU batch.
    Old references exclude any low-threshold new-CAM support in the main mode.
    Main/aux strong CAM seeds and strong already-computed C proposals supply new
    references. A region reference never validates one of its own seed pixels.
    Similarity and source scores are heuristic evidence, not probabilities.
    """
    if not 0 <= support < 1 or not 0 < margin < 2:
        raise ValueError('Invalid similarity criteria')
    if reference_mode not in ['regions', 'pooled']:
        raise ValueError(reference_mode)
    if min_region < 1 or max_regions < 1:
        raise ValueError('Invalid reference size')
    b, channels, fh, fw = features.shape
    size = hard.shape[-2:]
    classes = tags.shape[1] + 1
    if not 0 < old_count < classes-1:
        raise ValueError('Expected old and newly introduced classes')
    old_prob = F.interpolate(old_logits.detach().float(), size, mode='bilinear',
                             align_corners=False).softmax(1)
    old_score, old_label = old_prob.max(1)
    main = F.interpolate(cams.detach().float(), size, mode='bilinear', align_corners=False)
    auxiliary = F.interpolate(cams_aux.detach().float(), size, mode='bilinear',
                               align_corners=False)
    nc = main[:,old_count:] * tags[:,old_count:,None,None]
    na = auxiliary[:,old_count:] * tags[:,old_count:,None,None]
    ns, ni = nc.max(1)
    aus, aui = na.max(1)
    new_label = ni + old_count + 1
    licensed = tags[:,old_count:].sum(1)[:,None,None] > 0
    new_present = licensed & (ns >= .25)
    seeds = torch.zeros_like(hard)
    old_seed = valid & (hard > 0) & (hard <= old_count)
    old_seed &= (old_label == hard) & (old_score >= seed_threshold)
    if exclude_new_from_old:
        old_seed &= ~new_present
    seeds[old_seed] = hard[old_seed]
    new_seed = valid & licensed & (hard > old_count) & (hard < classes)
    new_seed &= (hard == new_label) & (ni == aui)
    new_seed &= (ns >= seed_threshold) & (aus >= seed_threshold)
    seeds[new_seed] = hard[new_seed]
    c_seeds = torch.zeros_like(valid)
    if recovery is not None:
        proposal, chosen = recovery['proposals'], recovery['selected']
        c_seeds = valid & (hard == 0) & chosen & proposal['accepted']
        c_seeds &= proposal['local_score'] >= seed_threshold
        seeds[c_seeds] = proposal['candidate'][c_seeds]
    conflict = valid & (hard > 0) & (hard < classes) & (old_label > 0)
    conflict &= new_present & (old_label != new_label) & (seeds == 0)
    feat = F.normalize(features.detach().float().flatten(2).transpose(1,2),
                       dim=-1, eps=1e-8)
    grid_valid = F.adaptive_avg_pool2d(valid[:,None].float(), (fh,fw))[:,0] >= .999
    grid_scores = features.new_full((b,classes,fh,fw), -2., dtype=torch.float32)
    reference_records = []
    bank = {}
    for bi in range(b):
        image_records = []
        for c in range(1,classes):
            occupancy = F.adaptive_avg_pool2d((seeds[bi:bi+1,None] == c).float(),
                                              (fh,fw))[0,0]
            mask = (occupancy >= .5) & grid_valid[bi]
            components = [g for g in _components(mask) if len(g) >= min_region]
            if reference_mode == 'pooled' and components:
                components = [sorted(i for g in components for i in g)]
            # Split large connected regions into spatial 4x4-token pieces, so
            # different appearances are not necessarily merged into one mean.
            regions = []
            for group in components:
                if reference_mode == 'pooled':
                    pieces = [group]
                else:
                    cells = {}
                    for i in group:
                        y,x = divmod(i,fw)
                        cells.setdefault((y//4,x//4),[]).append(i)
                    pieces = [v for _,v in sorted(cells.items()) if len(v) >= min_region]
                regions.extend(pieces)
            regions = sorted(regions,key=lambda x:(-len(x),x[0]))[:max_regions]
            if not regions:
                continue
            prototypes = []
            for group in regions:
                idx = torch.tensor(group,device=features.device,dtype=torch.long)
                prototypes.append(F.normalize(feat[bi,idx].mean(0),dim=0,eps=1e-8))
                image_records.append(dict(class_id=c,tokens=len(group),indices=group))
            prototypes = torch.stack(prototypes)
            bank[bi,c] = prototypes
        reference_records.append(image_records)
    borrowed = torch.zeros((b,classes),dtype=torch.bool,device=features.device)
    for bi in range(b):
        for c in range(1,classes):
            refs = bank.get((bi,c))
            if refs is None and batch_fallback:
                alternatives = [v for (image,label),v in bank.items() if image != bi and label == c]
                if alternatives:
                    refs = torch.cat(alternatives)
                    borrowed[bi,c] = True
            if refs is not None:
                grid_scores[bi,c] = (feat[bi] @ refs.T).amax(1).reshape(fh,fw)
    # Scores are upsampled, then resolved at each original pixel; a large
    # conflict component is never assigned one class by majority vote.
    scores = F.interpolate(grid_scores,size,mode='bilinear',align_corners=False)
    # Only the two disputed class hypotheses compete; unrelated available
    # references cannot win merely because a candidate reference is absent.
    old_similarity = scores.gather(1,old_label[:,None])[:,0]
    new_similarity = scores.gather(1,new_label[:,None])[:,0]
    new_wins = new_similarity > old_similarity
    candidate = torch.where(new_wins,new_label,old_label)
    similarity = torch.maximum(old_similarity,new_similarity)
    advantage = (new_similarity-old_similarity).abs()
    available = grid_scores.flatten(2).amax(2)>-1.5
    comparable = (old_similarity > -1.5) & (new_similarity > -1.5)
    eligible = conflict & comparable
    accepted = eligible & (similarity >= support) & (advantage >= margin)
    rejected = eligible & ~accepted
    # If no old/new reference pair exists, do not enter coordination at all.
    # Such unresolved pixels retain the baseline target and are logged separately.
    changed = accepted & (candidate != hard)
    confidence = ((similarity-support)/(1-support)).clamp(0,1)
    confidence *= (advantage/(margin*2)).clamp(0,1)
    return dict(candidate=candidate,conflict=conflict,eligible=eligible,
                accepted=accepted,rejected=rejected,changed=changed,
                seeds=seeds,c_seeds=c_seeds,scores=scores,
                similarity=similarity,advantage=advantage,confidence=confidence,
                references=reference_records,available=available,borrowed=borrowed,
                no_reference=conflict & ~comparable)


def region_correction(logits, hard, valid, evidence, ramp=1., mode='resolve'):
    """Exact native BCE delta: accepted target replacement + rejected-pixel abstention.

    Baseline normalization is unchanged. PTC and CLS are not inputs or modified.
    """
    if not 0 <= ramp <= 1 or mode not in ['resolve','reject_only']:
        raise ValueError((ramp,mode))
    classes = logits.shape[1]
    domain = valid & (hard>0) & (hard<classes)
    accepted = evidence['accepted'] & domain
    rejected = evidence['rejected'] & domain
    if (accepted & rejected).any():
        raise ValueError('Accepted/rejected overlap')
    candidate = evidence['candidate'].clamp(0,classes-1)
    original = logits.gather(1,hard.clamp(0,classes-1)[:,None])[:,0]
    proposed = logits.gather(1,candidate[:,None])[:,0]
    replace = (original-proposed)*accepted if mode=='resolve' else torch.zeros_like(original)
    bce = F.softplus(logits).sum(1)-original
    n = (valid & (hard != 255)).sum().clamp_min(1)
    return ramp*(replace-bce*rejected).sum()/n
