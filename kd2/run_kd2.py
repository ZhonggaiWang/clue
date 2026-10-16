"""Two independent full CLUE chains, two GPUs each, global batch8."""
from pathlib import Path
import datetime,hashlib,json,os,socket,subprocess,time,sys
ROOT=Path(__file__).resolve().parent
PYTHON=str(ROOT.parent/'fuse0/.e/bin/python')
DATA='/data/zhonggai/coco/PascalVOC12'
protocol=json.loads((ROOT/'experiment.json').read_text())
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def verify_source():
 for name,h in json.loads((ROOT/'source_manifest.json').read_text()).items():
  assert digest(ROOT/name)==h,'Source changed: '+name
 for name,h in protocol['unchanged_components'].items():assert digest(ROOT/name)==h
 assert digest(Path(protocol['base']['checkpoint']))==protocol['base']['sha256']
def idle(gpus):
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
 records={int(x.split(',')[0]):(x.split(',')[1].strip(),int(x.split(',')[2])) for x in rows.splitlines()}
 active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
 return all(records[g][1]<250 and records[g][0] not in active for g in gpus)
def worker(mode,gpus):
 assert mode in ['counter','relations'] and gpus==protocol['gpu_pairs'][mode]
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),OMP_NUM_THREADS='1',USE_LIBUV='0',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
 state={'status':'running','pid':os.getpid(),'gpus':gpus,'mode':mode,'world_size':2,'per_gpu_batch':4,'global_batch':8,'started_utc':now(),'jobs':[]}
 def save():
  state['updated_utc']=now();p=ROOT/('run_state_'+mode+'.json');tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(p)
 def waitidle():
  state['phase']='waiting_for_idle_GPUs';save()
  while not idle(gpus):time.sleep(20)
 def distributed(entry,args):
  with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
  return [PYTHON,'-B','-m','torch.distributed.run','--master_addr=127.0.0.1','--master_port='+str(port),'--nproc_per_node=2',entry]+args
 def execute(label,cmd):
  verify_source();waitidle()
  log=ROOT/(label+'.log');job={'name':label,'command':cmd,'log':str(log),'start_utc':now(),'status':'running'}
  state['phase']=label;state['jobs'].append(job);save();started=time.monotonic()
  with log.open('x') as stream:
   proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
   job['pid']=proc.pid;save();code=proc.wait()
  job.update(exit_code=code,elapsed_seconds=time.monotonic()-started,end_utc=now(),status='complete' if code==0 else 'failed');save()
  if code:raise RuntimeError(label+' failed; evidence retained in '+str(log))
  return job
 def traincommand(step,previous,smoke):
  work=ROOT/'runs'/('smoke_'+mode if smoke else mode)
  args=['--task','10-5','--step',str(step),'--data_folder',DATA,'--list_folder',str(ROOT/'datasets/voc'),'--work_dir',str(work),'--prev_checkpoint',str(previous),
  '--max_iters','3' if smoke else '8000','--eval_iters','0' if smoke else '2000','--log_iters','1' if smoke else '50','--spg','4','--num_workers','2' if smoke else '4',
  '--crop_size','448','--scales','.5','2','--cam_scales','1','.5','1.5','--lr','2e-5','--warmup_iters','0' if smoke else '2000','--warmup_ratio','1e-6','--wt_decay','.01',
  '--w_ptc','.2','--w_seg','.1','--bkg_thre','.5','--high_thre','.75','--low_thre','.25','--old_cls_threshold','2','--aux_layer','-3','--seed','0',
  '--recovery_mode','local','--recovery_start','0' if smoke else '2000','--fusion_mode','cosine','--fusion_margin','.1','--fusion_start','0' if smoke else '2000',
  '--kd_mode',mode,'--w_kd','.1','--kd_temperature','2','--kd_teacher_threshold','.75','--kd_start','0' if smoke else '2000']
  if smoke:args+=['--train_limit','32','--val_limit','4']
  return distributed('train.py',args),work/'10-5'/('step'+str(step))
 try:
  verify_source();save()
  for smoke in [True,False]:
   previous=Path(protocol['base']['checkpoint'])
   for step in [1,2]:
    cmd,out=traincommand(step,previous,smoke)
    job=execute(('smoke_' if smoke else 'full_')+mode+'_step'+str(step),cmd)
    previous=out/'model_final.pth'
    metric=json.loads((out/'metrics.jsonl').read_text().splitlines()[-1]);assert metric['iteration']==(3 if smoke else 8000)
    audit=json.loads((out/'kd_audit.json').read_text());assert audit['kd_pixels']>0 and audit['world_size']==2 and audit['batches_per_rank']==(3 if smoke else 6000)
    job.update(final_metric=metric,checkpoint=str(previous),checkpoint_sha256=digest(previous),kd_audit=audit);save()
   if smoke:state['smoke_status']='pass';save()
  report=ROOT/'reports'/mode;report.mkdir(parents=True,exist_ok=True)
  for step in [1,2]:
   cp=ROOT/'runs'/mode/'10-5'/('step'+str(step))/'model_final.pth'
   execute('eval_'+mode+'_step'+str(step),distributed('eval_multiscale.py',['--checkpoint',str(cp),'--data_folder',DATA,'--scales','.5','.75','1','1.25','1.5','--flip','--output',str(report/('step'+str(step)+'_ms_flip.json'))]))
   execute('paired_'+mode+'_step'+str(step),distributed('collect_paired_predictions.py',['--checkpoint',str(cp),'--output',str(report/('step'+str(step)+'_paired.npz'))]))
  cp=ROOT/'runs'/mode/'10-5/step2/model_final.pth'
  execute('quality_'+mode,distributed('audit_relation.py',['--checkpoint',str(cp),'--sample_size','128','--output',str(report/'final_relation_quality.json')]))
  state.update(status='complete',phase='complete',end_utc=now());save()
 except Exception as error:
  state.update(status='failed',error=str(error),end_utc=now());save();raise
def main():
 verify_source()
 assert not (ROOT/'run_state.json').exists(),'Do not duplicate an existing orchestration'
 state={'status':'running','pid':os.getpid(),'started_utc':now(),'gpu_pairs':protocol['gpu_pairs'],'workers':{}}
 def save():(ROOT/'run_state.json').write_text(json.dumps(state,indent=2))
 save();children=[]
 for mode,gpus in protocol['gpu_pairs'].items():
  with (ROOT/('worker_'+mode+'.log')).open('x') as f:
   p=subprocess.Popen([PYTHON,'-B',str(ROOT/'run_kd2.py'),mode]+list(map(str,gpus)),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  children.append((mode,p));state['workers'][mode]={'pid':p.pid,'gpus':gpus};save()
 for mode,p in children:
  state['workers'][mode]['exit_code']=p.wait();save()
 if any(x['exit_code'] for x in state['workers'].values()):
  state.update(status='failed',end_utc=now());save();raise RuntimeError('Worker failed; preserve logs and successful stages')
 code=subprocess.call([PYTHON,'-B','analyze_kd2.py'],cwd=ROOT)
 state.update(status='complete' if code==0 else 'analysis_failed',analysis_exit_code=code,end_utc=now());save()
if __name__=='__main__':
 if len(sys.argv)>1:worker(sys.argv[1],list(map(int,sys.argv[2:])))
 else:main()

