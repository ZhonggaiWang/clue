"""Retained C background recovery; original PTC/CLS/SEG remain unchanged."""
from model.background_recovery import propose_background, select_recovery, recovery_correction

def apply_supervision(losses, net, teacher, images, tags, target, old_seg,
                      cams, cam_labels, valid, old_classes, seg, args, evidence_sink=None):
    if teacher is not None and args.recovery_mode != 'off' and args.iteration >= args.recovery_start:
        proposals = propose_background(net, images, tags, target, cams, valid,
            old_classes, args.cam_scales, threshold=args.high_thre)
        selected = select_recovery(proposals, args.recovery_mode)
        if evidence_sink is not None:
            evidence_sink.update(proposals=proposals, selected=selected)
        correction = recovery_correction(seg, target, valid, proposals, selected)
        losses['total'] = losses['total'] + args.w_seg * correction
        losses['recovery_correction'] = correction
        count = (valid & (target != 255)).sum().clamp_min(1)
        losses['recovered_fraction'] = selected.float().sum() / count
    return losses
