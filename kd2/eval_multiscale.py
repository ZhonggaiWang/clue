"""Standalone VOC evaluation: average resized segmentation logits over scales/views.

Example (single GPU):
  python -B eval_multiscale.py --checkpoint runs/CF10_4gpu/10-5/step2/model_final.pth \
    --data_folder /path/to/VOC2012 --scales 0.5 1.0 1.5 --output reports/ms.json
Use torchrun --nproc_per_node=4 for sharded evaluation; --flip adds horizontal views.
--scales 1 reproduces the existing single-scale evaluate() without changing training.
"""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
for key, sub in {'TORCH_HOME':'torch','XDG_CACHE_HOME':'cache','HF_HOME':'hf',
                 'CUDA_CACHE_PATH':'cuda','TRITON_CACHE_DIR':'triton','TMPDIR':'tmp'}.items():
    path = ROOT / '.t' if key == 'TMPDIR' else ROOT / '.runtime' / sub
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError(f'Cache escapes project: {path}')
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import datetime
import hashlib
import json
import math
import time
import torch
from torch.nn import functional as F
import tasks
from datasets.voc import VOCDataset
from model.model_seg_neg import network


@torch.inference_mode()
def multiscale_logits(model, image, scales=(0.5, 1.0, 1.5), flip=False):
    """Resize -> optional flip -> patch pad -> logits -> unpad -> unflip -> resize.

    Image is one normalized RGB tensor [1,3,H,W]. No class tags, CAM, CRF,
    recovery or fusion training module is used. Every view has equal weight.
    """
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError('Expected one image [1,3,H,W]')
    if not scales or any(not math.isfinite(s) or s <= 0 for s in scales):
        raise ValueError('Scales must be finite and positive')
    h, w = image.shape[-2:]
    accumulated = None
    views = 0
    for scale in scales:
        sh, sw = max(16, round(h * scale)), max(16, round(w * scale))
        resized = image if (sh, sw) == (h, w) else F.interpolate(
            image, size=(sh, sw), mode='bilinear', align_corners=False)
        for flipped in ((False, True) if flip else (False,)):
            view = resized.flip(-1) if flipped else resized
            padded = F.pad(view, (0, (-sw) % 16, 0, (-sh) % 16))
            _, seg, _, _ = model(padded)
            seg = F.interpolate(seg, size=padded.shape[-2:], mode='bilinear',
                                align_corners=False)[..., :sh, :sw]
            if flipped:
                seg = seg.flip(-1)
            seg = F.interpolate(seg.float(), size=(h, w), mode='bilinear',
                                align_corners=False)
            accumulated = seg if accumulated is None else accumulated + seg
            views += 1
    return accumulated / views


@torch.inference_mode()
def evaluate_multiscale(model, dataset, device, scales, flip, rank=0, world=1):
    model.eval()
    n = model.num_classes
    hist = torch.zeros((n, n), dtype=torch.float64, device=device)
    processed = torch.zeros((), dtype=torch.int64, device=device)
    local_total = len(range(rank, len(dataset), world))
    for local_index, index in enumerate(range(rank, len(dataset), world), 1):
        batch = dataset[index]
        image = batch['image'][None].to(device)
        pred = multiscale_logits(model, image, scales, flip)[0].argmax(0)
        gt = batch['mask'].to(device)
        valid = (gt >= 0) & (gt < n)
        hist += torch.bincount(n * gt[valid] + pred[valid], minlength=n*n).reshape(n, n)
        processed += 1
        if rank == 0 and (local_index % 50 == 0 or local_index == local_total):
            print(json.dumps({'rank0_images':local_index,'rank0_total':local_total}), flush=True)
    if world > 1:
        torch.distributed.all_reduce(hist)
        torch.distributed.all_reduce(processed)
    if processed.item() != len(dataset):
        raise RuntimeError('Not all validation images were evaluated')
    denom = hist.sum(0) + hist.sum(1) - hist.diag()
    iou = torch.where(denom > 0, hist.diag() / denom, torch.nan) * 100
    class_to_index = {c:i for i,c in enumerate(dataset.seen)}
    def mean_for(ids):
        indices = [class_to_index[c] for c in ids if c in class_to_index]
        values = iou[indices]
        return float(values.nanmean()) if len(indices) and torch.isfinite(values).any() else None
    return {'miou':float(iou.nanmean()),
            'iou':[float(v) if torch.isfinite(v) else None for v in iou],
            'class_ids':dataset.seen, 'images':int(processed.item()),
            'old_class_ids':dataset.old, 'new_class_ids':[c for c in dataset.current if c != 0],
            'old_including_background_miou':mean_for(dataset.old),
            'new_miou':mean_for([c for c in dataset.current if c != 0]),
            'background_iou':mean_for([0]), 'confusion_matrix':hist.cpu().tolist()}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_folder', required=True)
    parser.add_argument('--list_folder', default=str(ROOT/'datasets/voc'))
    parser.add_argument('--scales', type=float, nargs='+', default=[0.5,1.0,1.5])
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--val_limit', type=int, default=0, help='Smoke check only; 0 uses full validation')
    parser.add_argument('--output', required=True, help='New JSON file inside this project')
    args = parser.parse_args()
    if any(not math.isfinite(s) or s <= 0 for s in args.scales) or args.val_limit < 0:
        parser.error('Scales must be finite/positive and val_limit nonnegative')
    output = Path(args.output).resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        parser.error('Output must be a new file inside this project')
    checkpoint = Path(args.checkpoint).resolve()
    blob = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if blob.get('baseline') != 'sasa-cls-ptc-seg-v1':
        raise ValueError('Checkpoint is not from the CLUE minimal baseline')
    task, step = blob['task'], blob['step']
    counts = tasks.get_per_task_classes('voc', task, step)
    model = network('deit_base_patch16_224', sum(counts), counts, pretrained=False,
                    aux_layer=blob.get('args', {}).get('aux_layer', -3), step=step+1)
    model.load_state_dict(blob['model_state'], strict=True)
    local_rank = int(os.environ.get('LOCAL_RANK',0))
    rank, world = int(os.environ.get('RANK',0)), int(os.environ.get('WORLD_SIZE',1))
    torch.set_num_threads(1)
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        torch.distributed.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    try:
        model.to(device).eval()
        dataset = VOCDataset(args.data_folder,args.list_folder,task,step,train=False,limit=args.val_limit)
        start = time.monotonic()
        metrics = evaluate_multiscale(model,dataset,device,args.scales,args.flip,rank,world)
        elapsed = time.monotonic()-start
        if rank == 0:
            metrics.update(task=task,step=step,checkpoint=str(checkpoint),
                checkpoint_sha256=sha256(checkpoint), checkpoint_iteration=blob.get('iteration'),
                scales=args.scales,flip=args.flip,aggregation='equal mean of segmentation logits',
                forwards_per_image=len(args.scales)*(2 if args.flip else 1),
                world_size=world,elapsed_seconds=elapsed,full_validation=args.val_limit==0,
                data_folder=str(Path(args.data_folder).resolve()),
                list_folder=str(Path(args.list_folder).resolve()),
                evaluator_sha256=sha256(Path(__file__)),
                finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            output.parent.mkdir(parents=True,exist_ok=True)
            with output.open('x') as stream:
                json.dump(metrics,stream,indent=2,allow_nan=False)
            print(json.dumps({k:v for k,v in metrics.items() if k not in ['confusion_matrix','iou']},allow_nan=False),flush=True)
    finally:
        if world > 1:
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
