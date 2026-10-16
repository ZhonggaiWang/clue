"""Standalone CLUE evaluation with optional multiscale, flip and DenseCRF."""
import sys,os,argparse,json,math,time,datetime
from pathlib import Path
from eval_multiscale import ROOT,multiscale_logits,sha256
import numpy as np
import torch
from PIL import Image
import tasks
from datasets.voc import VOCDataset
from model.model_seg_neg import network

def densecrf_probabilities(rgb,probability,iterations=10,gaussian_sxy=3,
                           gaussian_compat=3,bilateral_sxy=80,
                           bilateral_srgb=13,bilateral_compat=10):
    sys.path.insert(0,str(ROOT/'.crf_deps'))
    import pydensecrf.densecrf as dcrf
    if probability.ndim!=3 or rgb.shape!=(*probability.shape[1:],3):
        raise ValueError('Expected probabilities C,H,W and matching RGB H,W,3')
    if rgb.dtype!=np.uint8 or not np.isfinite(probability).all():
        raise ValueError('Expected uint8 RGB and finite probabilities')
    c,h,w=probability.shape
    if iterations<1 or c<2:
        raise ValueError('Invalid iterations or classes')
    unary=-np.log(np.clip(probability,1e-6,1.0))
    crf=dcrf.DenseCRF2D(w,h,c)
    crf.setUnaryEnergy(np.ascontiguousarray(unary.reshape(c,-1),dtype=np.float32))
    crf.addPairwiseGaussian(sxy=gaussian_sxy,compat=gaussian_compat,
        kernel=dcrf.DIAG_KERNEL,normalization=dcrf.NORMALIZE_SYMMETRIC)
    crf.addPairwiseBilateral(sxy=bilateral_sxy,srgb=bilateral_srgb,
        rgbim=np.ascontiguousarray(rgb),compat=bilateral_compat,
        kernel=dcrf.DIAG_KERNEL,normalization=dcrf.NORMALIZE_SYMMETRIC)
    return np.asarray(crf.inference(iterations),dtype=np.float32).reshape(c,h,w)

def metric(hist,dataset):
    cm=hist.cpu().numpy()
    den=cm.sum(0)+cm.sum(1)-np.diag(cm)
    iou=np.divide(100*np.diag(cm),den,out=np.full(len(den),np.nan),where=den>0)
    mapping={c:i for i,c in enumerate(dataset.seen)}
    def mean(ids):
        values=iou[[mapping[c] for c in ids if c in mapping]]
        return float(np.nanmean(values)) if np.isfinite(values).any() else None
    return {'miou':float(np.nanmean(iou)),
        'iou':[float(x) if np.isfinite(x) else None for x in iou],
        'old_including_background_miou':mean(dataset.old),
        'new_miou':mean([c for c in dataset.current if c!=0]),
        'old_foreground_miou':mean([c for c in dataset.old if c!=0]),
        'background_iou':mean([0]),'confusion_matrix':cm.tolist()}

@torch.inference_mode()
def evaluate(model,dataset,device,scales,flip,crf,params,rank,world):
    n=model.num_classes
    raw=torch.zeros(n,n,dtype=torch.float64,device=device)
    refined=raw.clone()
    count=torch.zeros((),dtype=torch.int64,device=device)
    inference_seconds=0.;crf_seconds=0.
    total=len(range(rank,len(dataset),world))
    for local_index,index in enumerate(range(rank,len(dataset),world),1):
        item=dataset[index];image=item['image'][None].to(device)
        start=time.monotonic()
        logits=multiscale_logits(model,image,scales,flip)[0]
        prediction=logits.argmax(0)
        if device.type=='cuda':torch.cuda.synchronize(device)
        inference_seconds+=time.monotonic()-start
        if crf:
            probability=logits.softmax(0).cpu().numpy()
            with Image.open(dataset.root/'JPEGImages'/(item['name']+'.jpg')) as handle:
                rgb=np.asarray(handle.convert('RGB'),dtype=np.uint8).copy()
            start=time.monotonic()
            q=densecrf_probabilities(rgb,probability,**params)
            refined_prediction=torch.from_numpy(q.argmax(0)).to(device)
            crf_seconds+=time.monotonic()-start
        else:
            refined_prediction=prediction
        # Ground truth is used only to accumulate held-out metrics.
        gt=item['mask'].to(device);valid=(gt>=0)&(gt<n)
        raw+=torch.bincount(n*gt[valid]+prediction[valid],minlength=n*n).reshape(n,n)
        refined+=torch.bincount(n*gt[valid]+refined_prediction[valid],minlength=n*n).reshape(n,n)
        count+=1
        if rank==0 and (local_index%50==0 or local_index==total):
            print(json.dumps({'rank0_images':local_index,'rank0_total':total}),flush=True)
    timing=torch.tensor([inference_seconds,crf_seconds],dtype=torch.float64,device=device)
    if world>1:
        for item in [raw,refined,count,timing]:torch.distributed.all_reduce(item)
    assert count.item()==len(dataset),'Incomplete validation'
    before,after=metric(raw,dataset),metric(refined,dataset)
    delta={key:after[key]-before[key] for key in
           ['miou','old_including_background_miou','old_foreground_miou','new_miou']
           if before[key] is not None and after[key] is not None}
    return {'without_crf':before,'with_crf':after if crf else None,
            'delta_crf_minus_raw':delta if crf else None,'images':int(count.item()),
            'summed_inference_seconds':float(timing[0]),
            'summed_crf_seconds':float(timing[1])}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--data_folder',required=True)
    parser.add_argument('--list_folder',default=str(ROOT/'datasets/voc'))
    parser.add_argument('--scales',nargs='+',type=float,default=[1.0])
    parser.add_argument('--flip',action='store_true')
    parser.add_argument('--crf',action='store_true')
    parser.add_argument('--crf_iterations',type=int,default=10)
    parser.add_argument('--gaussian_sxy',type=float,default=3)
    parser.add_argument('--gaussian_compat',type=float,default=3)
    parser.add_argument('--bilateral_sxy',type=float,default=80)
    parser.add_argument('--bilateral_srgb',type=float,default=13)
    parser.add_argument('--bilateral_compat',type=float,default=10)
    parser.add_argument('--val_limit',type=int,default=0)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    params={key:getattr(args,key) for key in
            ['gaussian_sxy','gaussian_compat','bilateral_sxy','bilateral_srgb','bilateral_compat']}
    params['iterations']=args.crf_iterations
    if not args.scales or any(not math.isfinite(x) or x<=0 for x in args.scales):
        parser.error('Scales must be finite and positive')
    if args.val_limit<0 or args.crf_iterations<1 or any(not math.isfinite(v) or
        v<0 or (v==0 and not k.endswith('compat')) for k,v in params.items()):
        parser.error('Invalid CRF parameters or validation limit')
    out=Path(args.output).resolve()
    if not out.is_relative_to(ROOT) or out.exists():parser.error('New output must be inside project')
    cp=Path(args.checkpoint).resolve()
    blob=torch.load(cp,map_location='cpu',weights_only=False)
    if blob.get('baseline')!='sasa-cls-ptc-seg-v1':raise ValueError('Unexpected checkpoint')
    task,step=blob['task'],blob['step']
    counts=tasks.get_per_task_classes('voc',task,step)
    model=network('deit_base_patch16_224',sum(counts),counts,pretrained=False,
                  aux_layer=blob['args'].get('aux_layer',-3),step=step+1)
    model.load_state_dict(blob['model_state'],strict=True)
    local=int(os.environ.get('LOCAL_RANK',0));rank=int(os.environ.get('RANK',0))
    world=int(os.environ.get('WORLD_SIZE',1));torch.set_num_threads(1)
    device=torch.device('cuda',local) if torch.cuda.is_available() else torch.device('cpu')
    if device.type=='cuda':torch.cuda.set_device(device)
    if world>1:torch.distributed.init_process_group('nccl' if device.type=='cuda' else 'gloo')
    try:
        model.to(device).eval()
        dataset=VOCDataset(args.data_folder,args.list_folder,task,step,train=False,limit=args.val_limit)
        start=time.monotonic()
        result=evaluate(model,dataset,device,args.scales,args.flip,args.crf,params,rank,world)
        if rank==0:
            result.update(task=task,step=step,class_ids=dataset.seen,checkpoint=str(cp),
                checkpoint_sha256=sha256(cp),checkpoint_iteration=blob.get('iteration'),
                scales=args.scales,flip=args.flip,densecrf=args.crf,
                crf_parameters=params if args.crf else None,unary='negative log softmax of mean segmentation logits',
                aggregation='equal mean of logits over scales and flips',
                full_validation=args.val_limit==0,world_size=world,
                forwards_per_image=len(args.scales)*(2 if args.flip else 1),
                elapsed_seconds=time.monotonic()-start,evaluator_sha256=sha256(Path(__file__)),
                finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            out.parent.mkdir(parents=True,exist_ok=True)
            with out.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
            print(json.dumps({'raw_miou':result['without_crf']['miou'],
                'crf_miou':result['with_crf']['miou'] if args.crf else None,
                'delta':result['delta_crf_minus_raw'],'images':result['images']}),flush=True)
    finally:
        if world>1:torch.distributed.destroy_process_group()
if __name__=='__main__':main()
