"""Train/evaluate the minimal SASA baseline on VOC incrementally."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
# Keep all writable caches and outputs inside this project.
for key, sub in {'TORCH_HOME':'torch','XDG_CACHE_HOME':'cache','HF_HOME':'hf',
                 'CUDA_CACHE_PATH':'cuda','TRITON_CACHE_DIR':'triton','TMPDIR':'tmp'}.items():
    path = ROOT / '.t' if key == 'TMPDIR' else ROOT / '.runtime' / sub
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError(f'Cache escapes project: {path}')
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import json
import hashlib
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import tasks
from datasets.voc import VOCDataset
from model.model_seg_neg import network
from model.PAR import PAR
from model.temporal_supervision import make_evidence_teacher, update_evidence_teacher
from utils.optimizer import PolyWarmupAdamW
from continual.Trainer import load_previous, initialize_backbone, loss_terms, evaluate


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['10-5', '10-2', '10-10', '15-5', '20-0'], default='10-5')
    p.add_argument('--step', type=int, default=0)
    p.add_argument('--data_folder', required=True)
    p.add_argument('--seg_label_dir', help='Optional augmented dense labels for base-stage training only')
    p.add_argument('--list_folder', default=str(ROOT/'datasets/voc'))
    p.add_argument('--work_dir', default=str(ROOT/'runs/base'))
    p.add_argument('--prev_checkpoint')
    p.add_argument('--pretrained_weights')
    p.add_argument('--from_scratch', action='store_true', help='Explicit random initialization for tests')
    p.add_argument('--eval_checkpoint')
    p.add_argument('--max_iters', type=int, default=8000)
    p.add_argument('--eval_iters', type=int, default=2000)
    p.add_argument('--log_iters', type=int, default=50)
    p.add_argument('--spg', type=int, default=1)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--crop_size', type=int, default=448)
    p.add_argument('--scales', type=float, nargs=2, default=[0.5, 2.0])
    p.add_argument('--cam_scales', type=float, nargs='+', default=[1.0, 0.5, 1.5])
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--warmup_iters', type=int, default=0)
    p.add_argument('--warmup_ratio', type=float, default=1e-6)
    p.add_argument('--wt_decay', type=float, default=0.01)
    p.add_argument('--w_ptc', type=float, default=0.2)
    p.add_argument('--w_seg', type=float, default=0.1)
    p.add_argument('--bkg_thre', type=float, default=0.5)
    p.add_argument('--high_thre', type=float, default=0.75)
    p.add_argument('--low_thre', type=float, default=0.25)
    p.add_argument('--old_cls_threshold', type=float, default=2.0)
    p.add_argument('--aux_layer', type=int, default=-3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--train_limit', type=int, default=0)
    p.add_argument('--val_limit', type=int, default=0)
    p.add_argument('--recovery_mode', choices=['off','local','confidence'], default='off')
    p.add_argument('--recovery_start', type=int, default=2000)
    p.add_argument('--anchor_mode', choices=['off','stable','pooled','hard'], default='off')
    p.add_argument('--anchor_scope', choices=['licensed','all_seen'], default='licensed')
    p.add_argument('--anchor_start', type=int, default=2000)
    p.add_argument('--anchor_ramp', type=int, default=2000)
    p.add_argument('--anchor_seed_threshold', type=float, default=.9)
    p.add_argument('--anchor_support', type=float, default=.7)
    p.add_argument('--anchor_radius', type=int, default=2)
    p.add_argument('--anchor_min_seeds', type=int, default=3)
    p.add_argument('--anchor_topk', type=int, default=5)
    p.add_argument('--temporal_mode', choices=['off','crossview','single','current'], default='off')
    p.add_argument('--temporal_start', type=int, default=2000)
    p.add_argument('--temporal_ramp', type=int, default=2000)
    p.add_argument('--temporal_decay', type=float, default=.99)
    p.add_argument('--temporal_view_scale', type=float, default=1.25)
    p.add_argument('--temporal_mix', type=float, default=1.)
    p.add_argument('--temporal_protection', choices=['off','cam'], default='cam')
    p.add_argument('--temporal_objective', choices=['replace','pair_abstain','pixel_abstain'], default='replace')
    p.add_argument('--region_mode', choices=['off','resolve','reject_only'], default='off')
    p.add_argument('--region_start', type=int, default=2000)
    p.add_argument('--region_ramp', type=int, default=2000)
    p.add_argument('--region_seed_threshold', type=float, default=.9)
    p.add_argument('--region_support', type=float, default=.7)
    p.add_argument('--region_margin', type=float, default=.05)
    p.add_argument('--region_min_tokens', type=int, default=2)
    p.add_argument('--region_max_references', type=int, default=4)
    p.add_argument('--region_reference_mode', choices=['regions','pooled'], default='regions')
    p.add_argument('--region_unfiltered_old', action='store_true')
    p.add_argument('--fusion_mode', choices=['off','cosine'], default='off')
    p.add_argument('--fusion_margin', type=float, default=.1)
    p.add_argument('--fusion_start', type=int, default=2000)
    p.add_argument('--kd_mode', choices=['off','plain','evidence','counter','relations'], default='off')
    p.add_argument('--w_kd', type=float, default=0.1)
    p.add_argument('--kd_start', type=int, default=2000)
    p.add_argument('--kd_temperature', type=float, default=2.0)
    p.add_argument('--kd_teacher_threshold', type=float, default=0.75)
    return p


def safe_output(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Write outputs inside the baseline project')
    return path


def build(step, args):
    counts = tasks.get_per_task_classes('voc', args.task, step)
    return network('deit_base_patch16_224', sum(counts), counts, pretrained=False,
                   aux_layer=args.aux_layer, step=step+1)


def main():
    args = parser().parse_args()
    if args.crop_size < 32 or args.crop_size % 16:
        raise ValueError('crop_size must be at least 32 and divisible by 16')
    if args.max_iters < 1 or args.spg < 1 or min(args.scales) <= 0 or min(args.cam_scales) <= 0:
        raise ValueError('Invalid iteration, batch size or scales')
    if args.kd_mode != 'off':
        if args.step == 0 or args.w_kd < 0 or args.kd_temperature <= 0:
            raise ValueError('KD is incremental-only with nonnegative weight and positive temperature')
        if args.fusion_mode != 'cosine' or args.recovery_mode != 'local':
            raise ValueError('Registered KD comparison requires current C+cosine fusion')
        if args.kd_start < max(args.fusion_start, args.recovery_start):
            raise ValueError('KD cannot precede evidence activation')
    if args.step == 1 and not args.eval_checkpoint:
        policy = json.loads((ROOT.parent/'STEP0_DEFAULT.json').read_text())
        if args.task != policy['task']:
            raise ValueError('A task-matched future-background0 step0 must be registered first')
        args.prev_checkpoint = args.prev_checkpoint or policy['checkpoint']
        if hashlib.sha256(Path(args.prev_checkpoint).read_bytes()).hexdigest() != policy['sha256']:
            raise ValueError('Step1 must start from the user-approved future-background0 step0')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        torch.distributed.init_process_group('nccl' if device.type == 'cuda' else 'gloo',
                                             **({'device_id': device} if device.type == 'cuda' else {}))
    random.seed(args.seed+rank); np.random.seed(args.seed+rank); torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    student = build(args.step, args)
    teacher = None
    if args.eval_checkpoint:
        blob = torch.load(args.eval_checkpoint, map_location='cpu', weights_only=False)
        if blob['task'] != args.task or blob['step'] != args.step:
            raise ValueError('Evaluation checkpoint/task/step mismatch')
        student.load_state_dict(blob['model_state'], strict=True)
    elif args.step > 0:
        teacher = build(args.step-1, args)
        previous = args.prev_checkpoint or str(Path(args.work_dir)/args.task/f'step{args.step-1}'/'model_final.pth')
        load_previous(student, teacher, previous, args.task, args.step)
        teacher.to(device)
    elif args.pretrained_weights:
        initialize_backbone(student, args.pretrained_weights)
    elif not args.from_scratch:
        raise ValueError('step 0 requires --pretrained_weights (or explicit --from_scratch)')
    student.to(device)
    val = VOCDataset(args.data_folder, args.list_folder, args.task, args.step, train=False, limit=args.val_limit)
    if args.eval_checkpoint:
        metrics = evaluate(student, val, device, rank, world)
        if rank == 0: print(json.dumps(metrics), flush=True)
        if world > 1: torch.distributed.destroy_process_group()
        return
    out = safe_output(Path(args.work_dir)/args.task/f'step{args.step}')
    setup_error = [None]
    if rank == 0:
        try:
            out.mkdir(parents=True, exist_ok=True)
            with (out/'config.json').open('x') as f:
                json.dump(vars(args), f, indent=2)
        except OSError as error:
            setup_error[0] = str(error)
    if world > 1:
        torch.distributed.broadcast_object_list(setup_error, src=0)
    if setup_error[0]:
        if world > 1: torch.distributed.destroy_process_group()
        raise RuntimeError('Output already exists or cannot be created: '+setup_error[0])
    groups = student.get_param_groups()
    optim = PolyWarmupAdamW([{'params':[p for p in group if p.requires_grad], 'lr':args.lr*(10 if i>=2 else 1)} for i,group in enumerate(groups)],
                           lr=args.lr, weight_decay=args.wt_decay, betas=(0.9, 0.999), warmup_iter=args.warmup_iters,
                           max_iter=args.max_iters, warmup_ratio=args.warmup_ratio, power=0.9)
    model = DistributedDataParallel(student, device_ids=[local_rank] if device.type == 'cuda' else None) if world>1 else student
    if args.temporal_mode != 'off' and (args.step == 0 or args.anchor_mode != 'off'):
        raise ValueError('Temporal A requires an incremental stage and v6 A disabled')
    if not 0<=args.temporal_mix<=1 or not 0<=args.temporal_decay<1 or args.temporal_view_scale<=0:
        raise ValueError('Invalid temporal evidence settings')
    if args.region_mode != 'off' and (args.step == 0 or args.anchor_mode != 'off' or args.temporal_mode != 'off'):
        raise ValueError('Region D is incremental-only and excludes retired anchor/temporal A')
    evidence_teacher = make_evidence_teacher(student) if args.temporal_mode != 'off' else None
    train = VOCDataset(args.data_folder, args.list_folder, args.task, args.step, crop_size=args.crop_size, scales=args.scales, limit=args.train_limit, seg_label_dir=args.seg_label_dir)
    sampler = DistributedSampler(train, num_replicas=world, rank=rank, seed=args.seed) if world>1 else None
    loader = DataLoader(train, batch_size=args.spg, shuffle=sampler is None, sampler=sampler,
                        num_workers=args.num_workers, drop_last=False, pin_memory=device.type=='cuda')
    if not len(loader): raise ValueError('Empty training split')
    iterator = iter(loader); epoch = 0
    par = PAR(num_iter=10, dilations=[1,2,4,8,12,24]).to(device).eval()
    student.train()
    kd_audit = None
    kd_audit_keys = None
    kd_audit_batches = 0
    for iteration in range(args.max_iters):
        try: batch = next(iterator)
        except StopIteration:
            epoch += 1
            if sampler: sampler.set_epoch(epoch)
            iterator = iter(loader); batch = next(iterator)
        optim.zero_grad(set_to_none=True)
        args.iteration = iteration
        losses, _ = loss_terms(model, teacher, batch, par, args, evidence_teacher)
        if 'kd_pixels' in losses:
            keys = [k for k in losses if k.startswith('kd_') and (k.endswith('_pixels') or k.endswith('_sum'))]
            values = torch.stack([losses[k].detach().double() for k in keys])
            if kd_audit is None:
                kd_audit_keys = keys
                kd_audit = torch.zeros_like(values)
            if keys != kd_audit_keys: raise RuntimeError('KD audit keys changed')
            kd_audit += values
            kd_audit_batches += 1
        if not torch.isfinite(losses['total']): raise FloatingPointError('Non-finite loss')
        losses['total'].backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in student.parameters()):
            raise FloatingPointError('Non-finite gradients')
        optim.step()
        if evidence_teacher is not None:
            update_evidence_teacher(evidence_teacher, student, 0. if args.temporal_mode=='current' else args.temporal_decay)
        if iteration == 0 or (iteration+1) % args.log_iters == 0 or iteration+1 == args.max_iters:
            values = torch.stack([v.detach() for v in losses.values()])
            if world>1: torch.distributed.all_reduce(values); values /= world
            if rank == 0:
                record = dict(iteration=iteration+1, **dict(zip(losses, values.tolist())))
                print(json.dumps(record), flush=True)
                with (out/'losses.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        if (args.eval_iters>0 and (iteration+1)%args.eval_iters==0) or iteration+1==args.max_iters:
            metrics = evaluate(student, val, device, rank, world)
            if rank==0:
                metrics['iteration']=iteration+1
                print(json.dumps(metrics), flush=True)
                with (out/'metrics.jsonl').open('a') as f: f.write(json.dumps(metrics)+'\n')
    if kd_audit is not None:
        if world > 1: torch.distributed.all_reduce(kd_audit)
        if rank == 0:
            audit = dict(zip(kd_audit_keys, kd_audit.cpu().tolist()))
            audit.update(batches_per_rank=kd_audit_batches, world_size=world,
                         scope='All active KD training batches, summed across ranks; repeated augmented pixels',
                         kd_mode=args.kd_mode, temperature=args.kd_temperature, weight=args.w_kd)
            with (out/'kd_audit.json').open('x') as stream: json.dump(audit,stream,indent=2)
    if rank==0:
        checkpoint = dict(baseline='sasa-cls-ptc-seg-v1', task=args.task, step=args.step,
                          model_state=student.state_dict(), args=vars(args), iteration=args.max_iters)
        if evidence_teacher is not None:
            checkpoint['evidence_state'] = evidence_teacher.state_dict()
        temp = out/'model_final.pth.tmp'
        torch.save(checkpoint, temp); temp.replace(out/'model_final.pth')
    if world>1: torch.distributed.barrier(); torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
