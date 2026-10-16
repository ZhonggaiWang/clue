"""Run two registered KD chains sequentially on GPUs2,4; final checkpoints only."""
from pathlib import Path
import datetime,hashlib,json,os,socket,subprocess,time
ROOT=Path(__file__).resolve().parent
PYTHON=str(ROOT.parent/'fuse0/.e/bin/python')
DATA='/data/zhonggai/coco/PascalVOC12'
protocol=json.loads((ROOT/'experiment.json').read_text())
GPUS=protocol['gpus']
ENV=os.environ.copy()
ENV.update(CUDA_VISIBLE_DEVICES=','.join(map(str,GPUS)),OMP_NUM_THREADS='1',
           USE_LIBUV='0',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',TMPDIR=str(ROOT/'.t'))
(ROOT/'.t').mkdir(exist_ok=True)
state={'status':'running','pid':os.getpid(),'gpus':GPUS,'jobs':[]}
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save():
 state['updated_utc']=now()
 p=ROOT/'run_state.json';t=ROOT/'run_state.tmp';t.write_text(json.dumps(state,indent=2));t.replace(p)
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def verify():
 for name,h in json.loads((ROOT/'source_manifest.json').read_text()).items():
  assert digest(ROOT/name)==h,'Source changed: '+name
 for attempt in range(6):
  rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
  usage={int(x.split(',')[0]):int(x.split(',')[1]) for x in rows.splitlines()}
  if all(usage[g]<250 for g in GPUS):return
  time.sleep(2)
 raise RuntimeError('Reserved GPU occupied: '+str(usage))
def distributed(entry,arguments):
 with socket.socket() as sock:
  sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
 return [PYTHON,'-B','-m','torch.distributed.run','--master_addr=127.0.0.1',
         '--master_port='+str(port),'--nproc_per_node=2',entry]+arguments
def execute(label,cmd):
 verify()
 log=ROOT/(label+'.log')
 job={'name':label,'status':'running','command':cmd,'log':str(log),'start_utc':now()}
 state['jobs'].append(job);save();started=time.monotonic()
 with log.open('x') as stream:
  proc=subprocess.Popen(cmd,cwd=ROOT,env=ENV,stdout=stream,stderr=subprocess.STDOUT)
  job['pid']=proc.pid;save();code=proc.wait()
 job.update(exit_code=code,elapsed_seconds=time.monotonic()-started,end_utc=now(),status='complete' if code==0 else 'failed');save()
 if code:raise RuntimeError(label+' failed; see '+str(log))
 return job
def train_command(mode,step,previous,smoke):
 work=ROOT/'runs'/('smoke_'+mode if smoke else mode)
 args=['--task','10-5','--step',str(step),'--data_folder',DATA,
  '--list_folder',str(ROOT/'datasets/voc'),'--work_dir',str(work),'--prev_checkpoint',str(previous),
  '--max_iters','3' if smoke else '8000','--eval_iters','0' if smoke else '2000',
  '--log_iters','1' if smoke else '50','--spg','4','--num_workers','2' if smoke else '4',
  '--crop_size','448','--scales','.5','2','--cam_scales','1','.5','1.5',
  '--lr','2e-5','--warmup_iters','0' if smoke else '2000','--warmup_ratio','1e-6',
  '--wt_decay','.01','--w_ptc','.2','--w_seg','.1','--bkg_thre','.5','--high_thre','.75',
  '--low_thre','.25','--old_cls_threshold','2','--aux_layer','-3','--seed','0',
  '--recovery_mode','local','--recovery_start','0' if smoke else '2000',
  '--fusion_mode','cosine','--fusion_margin','.1','--fusion_start','0' if smoke else '2000',
  '--kd_mode',mode,'--w_kd','.1','--kd_temperature','2','--kd_teacher_threshold','.75',
  '--kd_start','0' if smoke else '2000']
 if smoke:args+=['--train_limit','32','--val_limit','4']
 return distributed('train.py',args),work/'10-5'/('step'+str(step))
state['started_utc']=now();save()
try:
 base=Path(protocol['base']['checkpoint'])
 assert digest(base)==protocol['base']['sha256'],'Step0 hash changed'
 for smoke in [True,False]:
  for mode in ['plain','evidence']:
   previous=base
   for step in [1,2]:
    cmd,out=train_command(mode,step,previous,smoke)
    label=('smoke_' if smoke else 'full_')+mode+'_step'+str(step)
    job=execute(label,cmd)
    previous=out/'model_final.pth'
    metric=json.loads((out/'metrics.jsonl').read_text().splitlines()[-1])
    assert metric['iteration']==(3 if smoke else 8000)
    audit=json.loads((out/'kd_audit.json').read_text())
    assert audit['kd_eligible_pixels']>0
    assert audit['kd_pixels']>0,'KD never selected a pixel'
    if mode=='evidence':assert audit['kd_pixels']<audit['kd_eligible_pixels'],'Selective KD collapsed to global KD'
    job.update(final_metric=metric,checkpoint=str(previous),checkpoint_sha256=digest(previous),kd_audit=audit);save()
    if not smoke:
     report=ROOT/'reports'/mode;report.mkdir(parents=True,exist_ok=True)
     evalargs=['--checkpoint',str(previous),'--data_folder',DATA,'--scales','.5','.75','1','1.25','1.5',
               '--flip','--output',str(report/('step'+str(step)+'_ms_flip.json'))]
     execute('eval_'+mode+'_step'+str(step),distributed('eval_multiscale.py',evalargs))
  if smoke:
   state['smoke_status']='pass';save()
 state.update(status='training_complete',end_utc=now(),analysis_status='pending');save()
except Exception as error:
 state.update(status='failed',error=str(error),end_utc=now());save();raise
