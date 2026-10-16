"""Save per-image confusion on a common 1449-image validation set for paired analysis."""
from eval_multiscale import ROOT, multiscale_logits, sha256
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
import tasks
from datasets.voc import VOCDataset
from model.model_seg_neg import network

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
out=Path(a.output).resolve()
assert out.is_relative_to(ROOT) and not out.exists()
cp=Path(a.checkpoint).resolve()
blob=torch.load(cp,map_location='cpu',weights_only=False)
assert blob['task']=='10-5' and blob['step'] in (1,2)
rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
local=int(os.environ.get('LOCAL_RANK',0))
torch.set_num_threads(1);torch.cuda.set_device(local);device=torch.device('cuda',local)
if world>1:torch.distributed.init_process_group('nccl')
counts=tasks.get_per_task_classes('voc','10-5',blob['step'])
model=network('deit_base_patch16_224',sum(counts),counts,pretrained=False,
              aux_layer=blob['args'].get('aux_layer',-3),step=blob['step']+1)
model.load_state_dict(blob['model_state'],strict=True);model.to(device).eval()
dataset=VOCDataset('/data/zhonggai/coco/PascalVOC12',str(ROOT/'datasets/voc'),'10-5',2,train=False)
assert len(dataset)==1449 and dataset.seen==list(range(21))
hist=torch.zeros((len(dataset),21,21),dtype=torch.int64,device=device)
done=torch.zeros(len(dataset),dtype=torch.int64,device=device)
with torch.inference_mode():
 for idx in range(rank,len(dataset),world):
  item=dataset[idx]
  pred=multiscale_logits(model,item['image'][None].to(device),[1.],False)[0].argmax(0)
  gt=item['mask'].to(device);valid=(gt>=0)&(gt<21)
  hist[idx]=torch.bincount(21*gt[valid]+pred[valid],minlength=441).reshape(21,21)
  done[idx]=1
if world>1:
 torch.distributed.all_reduce(hist);torch.distributed.all_reduce(done)
assert (done==1).all()
if rank==0:
 out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('xb') as stream:
  np.savez_compressed(stream,confusion=hist.cpu().numpy(),names=np.array(dataset.names),
   checkpoint_sha256=np.array(sha256(cp)),step=np.array(blob['step']))
 print(json.dumps({'images':len(dataset),'step':blob['step'],'output':str(out)}),flush=True)
if world>1:torch.distributed.destroy_process_group()
