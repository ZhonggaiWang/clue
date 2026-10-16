"""Held-out KD-mask quality audit; dense GT is read only for final counting."""
from eval_multiscale import ROOT, sha256
assert sha256(ROOT/'kd_channel_audit.py')=='fd9690cca9227125440e170bfad8fa4cd1c7fa25de14823b6ab07a574314afa0'
from kd_channel_audit import binary_quality_counts,COLUMNS as CHANNEL_COLUMNS
from pathlib import Path
import argparse,json,os
import numpy as np
import torch
from torch.nn import functional as F
import tasks
from datasets.voc import VOCDataset
from model.model_seg_neg import network
from model.PAR import PAR
from model.candidate_fusion import candidate_evidence
from model.background_recovery import propose_background, select_recovery
from model.selective_kd import kd_evidence
from model.channel_kd import relation_selection,channel_foreground_kd
from utils.camutils import multi_scale_cam2,refine_cams_with_bkg_v2,get_mixed_label

p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
p.add_argument('--sample_size',type=int,default=128);a=p.parse_args()
assert a.sample_size>0
cp=Path(a.checkpoint).resolve();out=Path(a.output).resolve()
assert out.is_relative_to(ROOT) and not out.exists()
blob=torch.load(cp,map_location='cpu',weights_only=False);step=blob['step'];args=blob['args']
assert step in [1,2] and blob['task']=='10-5'
previous=Path(args['prev_checkpoint']).resolve()
oldblob=torch.load(previous,map_location='cpu',weights_only=False)
assert oldblob['step']==step-1 and oldblob['task']==blob['task']
rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
local=int(os.environ.get('LOCAL_RANK',0));torch.set_num_threads(1);torch.cuda.set_device(local)
device=torch.device('cuda',local)
if world>1:torch.distributed.init_process_group('nccl')
def build(blob):
 counts=tasks.get_per_task_classes('voc','10-5',blob['step'])
 net=network('deit_base_patch16_224',sum(counts),counts,pretrained=False,
  aux_layer=blob['args'].get('aux_layer',-3),step=blob['step']+1)
 net.load_state_dict(blob['model_state'],strict=True);return net.to(device).eval()
student=build(blob);teacher=build(oldblob);teacher.requires_grad_(False)
old_count=teacher.num_classes-1;total=student.num_classes-1;new_count=total-old_count
dataset=VOCDataset('/data/zhonggai/coco/PascalVOC12',str(ROOT/'datasets/voc'),'10-5',step,train=False)
tags_file=np.load(ROOT/'datasets/voc/cls_labels_onehot.npy',allow_pickle=True).item()
indices=np.linspace(0,len(dataset)-1,min(a.sample_size,len(dataset)),dtype=int).tolist()
names=['eligible','teacher_foreground','teacher_background','selected','new_supported','unresolved','conflict',
       'ambiguous','old_conflict','new_strong','C_recovered']
columns=['pixels','teacher_correct','fused_correct','gt_old_foreground','gt_new','gt_background','teacher_coarse_correct']
stats=torch.zeros(len(names),len(columns),dtype=torch.int64,device=device)
channel_stats=torch.zeros(len(names),len(CHANNEL_COLUMNS),dtype=torch.int64,device=device)
par=PAR(num_iter=10,dilations=[1,2,4,8,12,24]).to(device).eval()
MODE_NAMES=['counter','relations']
MODE_COLUMNS=['eligible_GT_channels','selected_GT_channels','selected_positive_channels','correct_positive_channels','selected_negative_channels','correct_negative_channels','wrong_negative_channels','withheld_correct_positive_channels','withheld_wrong_positive_channels','selected_uncertain_channels','eligible_pixels','eligible_channels','selected_channels','channel_weight_sum','raw_selected_KL_sum','weighted_KL_sum','ordinary_KL_sum']
mode_stats=torch.zeros((len(MODE_NAMES),len(MODE_COLUMNS)),dtype=torch.float64,device=device)
with torch.inference_mode():
 for position,idx in enumerate(indices[rank::world]):
  item=dataset[idx];image=item['image'][None].to(device);h,w=image.shape[-2:]
  image=F.pad(image,(0,(-w)%16,0,(-h)%16));hp,wp=image.shape[-2:]
  valid=torch.zeros((1,hp,wp),dtype=torch.bool,device=device);valid[:,:h,:w]=True
  old_cls,old_seg,_,_=teacher(image)
  old_logits=F.interpolate(old_seg,size=(hp,wp),mode='bilinear',align_corners=False)
  old_pixels=old_logits.argmax(1)
  # Image tags omit background: semantic class c occupies index c-1.
  semantic_ids=np.asarray(dataset.seen[1:],dtype=np.int64)
  weak=torch.as_tensor(np.asarray(tags_file[item['name']])[semantic_ids-1],device=device,dtype=image.dtype)[None]
  assert weak.shape[1]==total and old_cls.shape[1]==old_count
  tags=torch.cat([(old_cls>args['old_cls_threshold']).float(),weak[:,old_count:]],dim=1)
  cams,_=multi_scale_cam2(student,image,args['cam_scales'])
  _,student_seg,features,_=student(image)
  student_logits=F.interpolate(student_seg,size=(hp,wp),mode='bilinear',align_corners=False)
  mean=image.new_tensor([.485,.456,.406])[None,:,None,None]
  std=image.new_tensor([.229,.224,.225])[None,:,None,None]
  cam_labels=refine_cams_with_bkg_v2(par,image*std+mean,cams*tags[:,:,None,None],tags,
   args['high_thre'],args['low_thre'],255,[(0,h,0,w)])
  hard=get_mixed_label(cam_labels,old_pixels,total,new_count);hard[~valid]=255
  paired=candidate_evidence(features,student.classifier.get_weight(),old_pixels,
   cam_labels,hard,valid,tags,old_count,args['fusion_margin'])
  proposals=propose_background(student,image,tags,hard,cams,valid,old_count,
   args['cam_scales'],threshold=args['high_thre'])
  recovered=select_recovery(proposals,'local')
  masks=kd_evidence(old_logits,old_pixels,hard,valid,paired,recovered,old_count,args['kd_teacher_threshold'])
  masks.update(teacher_foreground=masks['eligible']&(old_pixels>0),
               teacher_background=masks['eligible']&(old_pixels==0),
               new_strong=paired['new_strong'],C_recovered=recovered)
  fused=hard.clone();fused[recovered]=proposals['candidate'][recovered]
  fused[paired['to_old']]=old_pixels[paired['to_old']].to(fused.dtype)
  candidate_modes={}
  for mode in MODE_NAMES:
   chosen,prob=relation_selection(old_logits,old_pixels,masks,old_count,mode,args['kd_teacher_threshold'])
   _,_,audit=channel_foreground_kd(student_logits,old_logits,old_pixels,masks,old_count,mode,args['kd_temperature'],args['kd_teacher_threshold'])
   candidate_modes[mode]=(chosen,prob,audit)
  # Pixel GT enters only below, after every prediction and mask is fixed.
  gt=F.pad(item['mask'][None].to(device),(0,wp-w,0,hp-h),value=255)
  gt_valid=valid&(gt!=255)
  channel_ids=torch.arange(1,old_count+1,device=device)[None,:,None,None]
  truth=gt[:,None]==channel_ids
  counted=(gt_valid&masks['eligible'])[:,None].expand_as(truth)
  for i,mode in enumerate(MODE_NAMES):
   chosen,prob,audit=candidate_modes[mode]
   chosen_gt=chosen&counted
   positive=prob>.5;negative=~positive
   uncertain=(prob>1-args['kd_teacher_threshold'])&(prob<args['kd_teacher_threshold'])
   values_mode=[counted.sum(),chosen_gt.sum(),(chosen_gt&positive).sum(),(chosen_gt&positive&truth).sum(),(chosen_gt&negative).sum(),(chosen_gt&negative&~truth).sum(),(chosen_gt&negative&truth).sum(),(counted&~chosen&positive&truth).sum(),(counted&~chosen&positive&~truth).sum(),(chosen_gt&uncertain).sum(),masks['eligible'].sum(),audit['kd_eligible_channel_sum'],audit['kd_selected_channel_sum'],audit['kd_channel_weight_sum'],audit['kd_raw_selected_kl_sum'],audit['kd_kl_sum'],audit['kd_plain_kl_sum']]
   mode_stats[i]+=torch.stack([v.double() for v in values_mode])
  if position%16==0:print(json.dumps({'rank':rank,'processed':position+1,'local_images':len(indices[rank::world])}),flush=True)
  old_probability=old_logits[:,1:old_count+1].sigmoid()
  values=[torch.ones_like(valid),old_pixels==gt,fused==gt,(gt>0)&(gt<=old_count),
          (gt>old_count)&(gt<=total),gt==0,
          (old_pixels==gt)|((old_pixels==0)&(gt>old_count)&(gt<=total))]
  for i,name in enumerate(names):
   selected=masks[name]&gt_valid
   for j,value in enumerate(values):stats[i,j]+=(selected&value).sum()
   channel_stats[i]+=binary_quality_counts(old_probability,gt,selected,old_count,total,args['kd_teacher_threshold'])
if world>1:
 torch.distributed.all_reduce(stats)
 torch.distributed.all_reduce(channel_stats)
 torch.distributed.all_reduce(mode_stats)
if rank==0:
 result={'checkpoint_sha256':sha256(cp),'teacher_sha256':sha256(previous),'step':step,
  'images':len(indices),'image_names':[dataset.names[i] for i in indices],
  'scope':'Frozen final models on deterministic held-out subset, original-aspect images; not training-time accuracy',
  'input_tags':'Existing image-level labels; no pixel GT input to candidate selection',
  'groups':{name:dict(zip(columns,stats[i].cpu().tolist())) for i,name in enumerate(names)}}
 for row in result['groups'].values():
  row['teacher_accuracy_percent']=100*row['teacher_correct']/row['pixels'] if row['pixels'] else None
  row['teacher_coarse_accuracy_percent']=100*row['teacher_coarse_correct']/row['pixels'] if row['pixels'] else None
  row['fused_accuracy_percent']=100*row['fused_correct']/row['pixels'] if row['pixels'] else None
 result['coarse_teacher_semantics']='For the old teacher, current newly introduced GT classes map to its background category. Coarse accuracy and full expanded-label accuracy describe different label spaces; channel diagnostics separately check binary relations.'
 result['actual_kd_mask_category']='channel_'+args['kd_mode'] if args['kd_mode'] in MODE_NAMES else ('eligible' if args['kd_mode']=='plain' else 'selected')
 result['selected_group_definition']='Legacy protected-old pixel mask for reference. Actual counter/relations KD uses per-channel selection described by candidate_channel_modes.'
 result['old_channel_diagnostic']={'probability':'Sigmoid of raw teacher old-foreground logits, matching confidence gating; training KD temperature is separate','threshold':args['kd_teacher_threshold'],'interpretation':'Teacher category BG on new GT can still provide correct old-class absence signals; GT is used only in this offline diagnostic.','groups':{name:dict(zip(CHANNEL_COLUMNS,channel_stats[i].cpu().tolist())) for i,name in enumerate(names)}}
 for row in result['old_channel_diagnostic']['groups'].values():
  row['positive_precision_percent']=100*row['correct_positive_channels']/row['confident_positive_channels'] if row['confident_positive_channels'] else None
  row['negative_precision_percent']=100*row['correct_negative_channels']/row['confident_negative_channels'] if row['confident_negative_channels'] else None
  row['old_gt_false_negative_percent']=100*row['wrong_negative_channels']/row['gt_old_positive_channels'] if row['gt_old_positive_channels'] else None
  row['new_gt_all_old_confident_negative_percent']=100*row['new_gt_pixels_all_old_confident_negative']/row['new_gt_pixels'] if row['new_gt_pixels'] else None
 result['candidate_channel_modes']={name:dict(zip(MODE_COLUMNS,mode_stats[i].cpu().tolist())) for i,name in enumerate(MODE_NAMES)}
 for row in result['candidate_channel_modes'].values():
  def ratio(n,d):return 100*row[n]/row[d] if row[d] else None
  row['channel_coverage_percent']=ratio('selected_channels','eligible_channels')
  row['positive_precision_percent']=ratio('correct_positive_channels','selected_positive_channels')
  row['negative_precision_percent']=ratio('correct_negative_channels','selected_negative_channels')
  row['withheld_positive_error_percent']=100*row['withheld_wrong_positive_channels']/(row['withheld_wrong_positive_channels']+row['withheld_correct_positive_channels']) if row['withheld_wrong_positive_channels']+row['withheld_correct_positive_channels'] else None
  row['weight_budget_percent']=ratio('channel_weight_sum','eligible_channels')
  row['weighted_KL_relative_to_ordinary_percent']=ratio('weighted_KL_sum','ordinary_KL_sum')
 result['candidate_diagnostic_scope']='Counterfactual channel masks on identical frozen ordinaryKD model/teacher and128 heldoutimages. All masks fixedbeforepixelGTcounting. Confidence usesrawsigmoid; KL usesT2. Channelweight-budget normalizationdoesnotguaranteeequalKL/gradients. Not training-time accuracy or performance.'
 result['channel_module_sha256']=sha256(ROOT/'model/channel_kd.py')
 out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps({'step':step,'images':len(indices),'groups':result['groups']}),flush=True)
if world>1:torch.distributed.destroy_process_group()
