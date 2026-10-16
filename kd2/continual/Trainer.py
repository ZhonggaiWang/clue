"""Three-loss baseline and explicit stage-to-stage transfer."""
import torch
from torch.nn import functional as F
from model.losses import get_masked_ptc_loss, get_seg_loss
from model.composition import apply_supervision
from model.candidate_fusion import candidate_evidence, candidate_correction
from model.selective_kd import kd_evidence, old_foreground_kd
from model.channel_kd import channel_foreground_kd
from model.anchor_supervision import anchor_evidence, anchor_correction
from model.temporal_supervision import temporal_evidence, temporal_correction
from model.ambiguity_supervision import ambiguity_correction
from model.region_supervision import region_evidence, region_correction
from utils.camutils import (multi_scale_cam2, cam_to_label, label_to_aff_mask,
                            refine_cams_with_bkg_v2, get_mixed_label)


def unwrap(model):
    return model.module if hasattr(model, 'module') else model


def load_previous(student, teacher, checkpoint, task, step):
    blob = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if blob.get('task') != task or blob.get('step') != step-1:
        raise ValueError('Previous checkpoint must match task and immediately preceding step')
    if blob.get('baseline') != 'sasa-cls-ptc-seg-v1':
        raise ValueError('Use a checkpoint from this minimal baseline, not a SASA prototype model')
    teacher.load_state_dict(blob['model_state'], strict=True)
    result = student.load_state_dict(blob['model_state'], strict=False)
    expected = {f'classifier.{step}.weight', f'aux_classifier.{step}.weight', f'decoder.conv8.{step}.weight'}
    if set(result.missing_keys) != expected or result.unexpected_keys:
        raise RuntimeError(f'Unexpected stage transfer: {result}')
    teacher.requires_grad_(False)
    teacher.eval()
    return result


def initialize_backbone(model, path):
    blob = torch.load(path, map_location='cpu', weights_only=False)
    state = blob.get('model', blob.get('state_dict', blob))
    state = {k.removeprefix('module.').removeprefix('encoder.'): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if not k.startswith('head.')}
    model.encoder.load_state_dict(state, strict=True)


def loss_terms(model, teacher, batch, par, args, evidence_teacher=None):
    device = next(model.parameters()).device
    images = batch['image'].to(device)
    valid = batch['valid'].to(device)
    tags = batch['tags'].to(device)
    net = unwrap(model)
    total_classes = net.num_classes-1
    new_classes = net.classes_list[-1]
    old_classes = total_classes-new_classes if teacher is not None else 0
    with torch.no_grad():
        old_pixels = None
        if teacher is not None:
            old_cls, old_seg, _, _ = teacher(images)
            tags = torch.cat([(old_cls > args.old_cls_threshold).float(), tags[:, old_classes:]], dim=1)
            old_pixels = F.interpolate(old_seg, size=images.shape[-2:], mode='bilinear', align_corners=False).argmax(1)
            cams, cams_aux = multi_scale_cam2(net, images, args.cam_scales)
    cls, seg, features, cls_aux = model(images)
    # Both CAM heads need image-level training; together these form basic CLS.
    cls_loss = F.multilabel_soft_margin_loss(cls, tags) + F.multilabel_soft_margin_loss(cls_aux, tags)
    # The supervised base stage uses CLS + SEG only. Original PTC runs in increments.
    ptc_loss = images.new_zeros(())
    if teacher is not None:
        fh, fw = features.shape[-2:]
        aux = F.interpolate(cams_aux, size=(fh, fw), mode='bilinear', align_corners=False)
        boxes = [(0, fh, 0, fw)] * images.shape[0]
        _, small_labels = cam_to_label(aux.detach(), tags, img_box=boxes, ignore_mid=True,
                                       bkg_thre=args.bkg_thre, high_thre=args.high_thre,
                                       low_thre=args.low_thre, ignore_index=255)
        small_valid = F.interpolate(valid[:, None].float(), size=(fh, fw), mode='nearest')[:, 0].bool()
        small_labels[~small_valid] = 255
        ptc_loss = get_masked_ptc_loss(features, label_to_aff_mask(small_labels))
    with torch.no_grad():
        if teacher is None:
            target = batch['mask'].to(device).clone()
        else:
            mean = images.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
            std = images.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
            rgb = images * std + mean
            boxes = []
            for mask in valid:
                rows, cols = mask.nonzero(as_tuple=True)
                boxes.append((int(rows.min()), int(rows.max())+1, int(cols.min()), int(cols.max())+1))
            cam_labels = refine_cams_with_bkg_v2(par, rgb, cams * tags[:, :, None, None], tags,
                                                args.high_thre, args.low_thre, 255, boxes)
            # Basic new-class CAM override; old segmentation supplies all remaining pixels.
            target = get_mixed_label(cam_labels, old_pixels, total_classes, new_classes)
        target[~valid] = 255
    seg = F.interpolate(seg, size=target.shape[-2:], mode='bilinear', align_corners=False)
    seg_loss = get_seg_loss(seg, target)
    total = cls_loss + args.w_ptc * ptc_loss + args.w_seg * seg_loss
    losses = {'total': total, 'cls': cls_loss, 'ptc': ptc_loss, 'seg': seg_loss}
    if teacher is not None:
        region_active = getattr(args, 'region_mode', 'off') != 'off' and args.iteration >= args.region_start
        kd_active = getattr(args, 'kd_mode', 'off') != 'off' and args.iteration >= args.kd_start
        recovery_evidence = {} if (region_active or kd_active) else None
        paired_evidence = None
        losses = apply_supervision(losses, net, teacher, images, tags, target, old_seg,
            cams, cam_labels, valid, old_classes, seg, args, evidence_sink=recovery_evidence)
        if getattr(args, 'fusion_mode', 'off') != 'off' and args.iteration >= args.fusion_start:
            evidence = candidate_evidence(features, net.classifier.get_weight(), old_pixels,
                cam_labels, target, valid, tags, old_classes, args.fusion_margin)
            paired_evidence = evidence
            correction = candidate_correction(seg, target, valid, evidence)
            losses['total'] = losses['total'] + args.w_seg * correction
            losses['fusion_correction'] = correction
            count = (valid & (target != 255)).sum().clamp_min(1)
            for name in ['conflict', 'to_old', 'new_strong']:
                losses['fusion_'+name+'_fraction'] = evidence[name].float().sum()/count
            conflicts = evidence['conflict'].sum().clamp_min(1)
            losses['fusion_to_old_given_conflict'] = evidence['to_old'].float().sum()/conflicts
        if kd_active:
            if paired_evidence is None:
                raise ValueError('KD requires active cosine fusion for matched evidence accounting')
            teacher_logits = F.interpolate(old_seg.detach(), size=target.shape[-2:],
                                           mode='bilinear', align_corners=False)
            recovered = recovery_evidence.get('selected', torch.zeros_like(valid))
            kd_masks = kd_evidence(teacher_logits, old_pixels, target, valid,
                paired_evidence, recovered, old_classes, args.kd_teacher_threshold)
            selected = kd_masks['eligible'] if args.kd_mode == 'plain' else kd_masks['selected']
            channel_stats = None
            if args.kd_mode in ['counter','relations']:
                kd_loss, per_pixel, channel_stats = channel_foreground_kd(seg, teacher_logits,
                    old_pixels, kd_masks, old_classes, args.kd_mode, args.kd_temperature,
                    args.kd_teacher_threshold)
            else:
                kd_loss, per_pixel = old_foreground_kd(seg, teacher_logits, selected,
                    kd_masks['eligible'], old_classes, args.kd_temperature)
            losses['total'] = losses['total'] + args.w_kd * kd_loss
            losses['kd'] = kd_loss
            losses['kd_pixels'] = selected.sum().float()
            for label, key in [('eligible','eligible'),('old_evidence','selected'),
                               ('new_evidence','new_supported'),('unresolved','unresolved'),
                               ('ambiguous_conflict','ambiguous'),('conflict','conflict'),
                               ('old_conflict','old_conflict'),('teacher_confident','teacher_confident')]:
                losses['kd_'+label+'_pixels'] = kd_masks[key].sum().float()
            losses['kd_kl_sum'] = (per_pixel*selected).sum()
            losses['kd_plain_kl_sum'] = (per_pixel*kd_masks['eligible']).sum()
            if channel_stats is not None: losses.update(channel_stats)
        if region_active:
            evidence = region_evidence(features, old_seg, cams, cams_aux, tags, target,
                valid, old_classes, recovery_evidence if recovery_evidence else None,
                args.region_seed_threshold, args.region_support, args.region_margin,
                args.region_min_tokens, args.region_max_references,
                args.region_reference_mode, not args.region_unfiltered_old)
            ramp = min(1., max(0., (args.iteration-args.region_start)/max(1,args.region_ramp)))
            correction = region_correction(seg, target, valid, evidence, ramp, args.region_mode)
            losses['total'] = losses['total'] + args.w_seg*correction
            losses['region_correction'] = correction
            count = (valid & (target != 255)).sum().clamp_min(1)
            for name in ['conflict','eligible','accepted','rejected','changed','no_reference']:
                losses['region_'+name+'_fraction'] = evidence[name].float().sum()/count
            losses['region_seed_fraction'] = (evidence['seeds']>0).float().sum()/count
            losses['region_C_seed_fraction'] = evidence['c_seeds'].float().sum()/count
            losses['region_ramp'] = images.new_tensor(ramp)
        if args.anchor_mode != 'off' and args.iteration >= args.anchor_start:
            evidence = anchor_evidence(features, old_seg, cams, cams_aux, tags,
                target, valid, old_classes, args)
            ramp = min(1., max(0., (args.iteration-args.anchor_start)/max(1,args.anchor_ramp)))
            correction = anchor_correction(seg, target, valid, evidence, ramp)
            losses['total'] = losses['total'] + args.w_seg*correction
            losses['anchor_correction'] = correction
            denom = (valid & (target != 255)).sum().clamp_min(1)
            selected = evidence['selected'];weight = evidence['weight']*ramp
            losses['anchor_selected_fraction'] = selected.float().sum()/denom
            losses['anchor_weight_fraction'] = weight.sum()/denom
            losses['anchor_to_old_weight_fraction'] = (weight*(evidence['candidate'] <= old_classes)).sum()/denom
            losses['anchor_to_new_weight_fraction'] = (weight*(evidence['candidate'] > old_classes)).sum()/denom
            losses['anchor_seed_fraction'] = (evidence['seeds'] > 0).float().sum()/denom
            losses['anchor_identifiable_fraction'] = evidence['identifiable'].float().sum()/denom
            losses['anchor_ramp'] = images.new_tensor(ramp)
    if evidence_teacher is not None and args.temporal_mode != 'off' and args.iteration >= args.temporal_start:
        evidence = temporal_evidence(evidence_teacher, images, tags, target, valid, old_classes, args)
        ramp = min(1., max(0., (args.iteration-args.temporal_start)/max(1,args.temporal_ramp)))
        objective = getattr(args, 'temporal_objective', 'replace')
        correction = (temporal_correction(seg, target, valid, evidence, args.temporal_mix*ramp)
            if objective == 'replace' else ambiguity_correction(seg, target, valid, evidence,
                args.temporal_mix*ramp, objective))
        losses['total'] = losses['total'] + args.w_seg*correction
        losses['temporal_correction'] = correction
        count = (valid & (target != 255)).sum().clamp_min(1)
        weight=evidence['weight']*args.temporal_mix*ramp
        losses['temporal_selected_fraction'] = evidence['selected'].float().sum()/count
        losses['temporal_weight_fraction'] = weight.sum()/count
        losses['temporal_to_old_weight_fraction'] = (weight*(evidence['candidate']>0)*(evidence['candidate']<=old_classes)).sum()/count
        losses['temporal_to_new_weight_fraction'] = (weight*(evidence['candidate']>old_classes)).sum()/count
        losses['temporal_to_background_weight_fraction'] = (weight*(evidence['candidate']==0)).sum()/count
        losses['temporal_ramp'] = images.new_tensor(ramp)
        losses['temporal_extra_views_per_image'] = images.new_tensor(1 if args.temporal_mode=='single' else 2)
    return losses, target


@torch.no_grad()
def evaluate(model, dataset, device, rank=0, world=1):
    net = unwrap(model)
    was_training = net.training
    net.eval()
    n = net.num_classes
    hist = torch.zeros((n, n), dtype=torch.float64, device=device)
    for index in range(rank, len(dataset), world):
        batch = dataset[index]
        image = batch['image'][None].to(device)
        # Preserve aspect ratio; pad only to the Transformer patch multiple.
        h, w = image.shape[-2:]
        image = F.pad(image, (0, (-w) % 16, 0, (-h) % 16))
        _, seg, _, _ = net(image)
        pred = F.interpolate(seg, size=image.shape[-2:], mode='bilinear', align_corners=False)[0, :, :h, :w].argmax(0)
        gt = batch['mask'].to(device)
        valid = (gt >= 0) & (gt < n)
        hist += torch.bincount(n * gt[valid] + pred[valid], minlength=n*n).reshape(n, n)
    if world > 1:
        torch.distributed.all_reduce(hist)
    denom = hist.sum(0) + hist.sum(1) - hist.diag()
    iou = torch.where(denom > 0, hist.diag()/denom, torch.nan)
    net.train(was_training)
    return {'miou': float(iou.nanmean()*100), 'iou': [float(x*100) if torch.isfinite(x) else None for x in iou],
            'images': len(dataset), 'class_ids': dataset.seen}
