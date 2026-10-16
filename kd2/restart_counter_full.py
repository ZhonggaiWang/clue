"""User-requested complete counter-KD restart from canonical Step0; GPUs4,5."""
from pathlib import Path
import json,os,socket,subprocess,time,hashlib
import run_kd2 as frozen
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/counter_restart_20261015'
OUT=ROOT/'reports/counter_restart_20261015'
GPUS=[4,5]
state={'status':'running','pid':os.getpid(),'gpus':GPUS,'global_batch':8,'world_size':2,'per_gpu_batch':4,'seg_start':0,'kd_start':2000,'jobs':[],'started_utc':frozen.now()}
def save():
 state['updated_utc']=frozen.now();p=OUT/'run_state.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2));t.replace(p)
def dist(entry,args):
 with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
 return [frozen.PYTHON,'-B','-m','torch.distributed.run','--master_addr=127.0.0.1','--master_port='+str(port),'--nproc_per_node=2',entry]+args
def execute(label,cmd):
 frozen.verify_source()
 state['phase']='waiting_for_GPU4_5_and_shm';save()
 while not frozen.idle(GPUS) or os.statvfs('/dev/shm').f_bavail*os.statvfs('/dev/shm').f_frsize<1024**3:time.sleep(20)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='4,5',OMP_NUM_THREADS='1',USE_LIBUV='0',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
 state['phase']=label;j={'name':label,'command':cmd,'start_utc':frozen.now(),'status':'running'};state['jobs'].append(j);save()
 with (OUT/(label+'.log')).open('x') as f:
  p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT);j['pid']=p.pid;save();code=p.wait()
 j.update(status='complete' if code==0 else 'failed',exit_code=code,end_utc=frozen.now());save()
 assert code==0,(label,code)
 return j
def report():
 result={'variants':{},'limitations':['Single seed; physical GPU pairs differ.','Old interrupted run excluded from final comparisons.']}
 for mode,run,out in [('plain',ROOT.parent/'kd1/runs/plain',ROOT.parent/'kd1/reports/plain'),('counter_restart',RUN,OUT)]:
  stages={}
  for step in [1,2]:
   r=run/'10-5'/('step'+str(step));m=json.loads((r/'metrics.jsonl').read_text().splitlines()[-1]);ms=json.loads((out/('step'+str(step)+'_ms_flip.json')).read_text());n=11 if step==1 else 16
   stages[str(step)]={'SS_all':m['miou'],'SS_old_including_background':sum(m['iou'][:n])/n,'SS_new':sum(m['iou'][n:])/len(m['iou'][n:]),'MS5flip_all':ms['miou'],'MS5flip_old_including_background':ms['old_including_background_miou'],'MS5flip_new':ms['new_miou']}
  result['variants'][mode]=stages
 result['counter_minus_plain_pp']={str(s):{k:v-result['variants']['plain'][str(s)][k] for k,v in result['variants']['counter_restart'][str(s)].items()} for s in [1,2]}
 (OUT/'results_comparison.json').write_text(json.dumps(result,indent=2));return result
try:
 OUT.mkdir(parents=True,exist_ok=False);save();frozen.verify_source()
 original=json.loads((ROOT/'run_state_counter.json').read_text())
 previous=Path(frozen.protocol['base']['checkpoint'])
 for step in [1,2]:
  template=[j for j in original['jobs'] if j['name']=='full_counter_step'+str(step)][0]
  cmd=template['command'][:]
  for flag,value in [('--work_dir',str(RUN)),('--prev_checkpoint',str(previous))]:cmd[cmd.index(flag)+1]=value
  for i,x in enumerate(cmd):
   if x.startswith('--master_port='):
    with socket.socket() as s:s.bind(('127.0.0.1',0));cmd[i]='--master_port='+str(s.getsockname()[1])
  j=execute('full_step'+str(step),cmd)
  out=RUN/'10-5'/('step'+str(step));previous=out/'model_final.pth'
  m=json.loads((out/'metrics.jsonl').read_text().splitlines()[-1]);a=json.loads((out/'kd_audit.json').read_text())
  assert m['iteration']==8000 and m['images']==(1240 if step==1 else 1449)
  assert a['batches_per_rank']==6000 and a['world_size']==2 and a['kd_mode']=='counter'
  first=json.loads((out/'losses.jsonl').read_text().splitlines()[0]) if (out/'losses.jsonl').exists() else None
  j.update(final_metric=m,checkpoint=str(previous),checkpoint_sha256=frozen.digest(previous),kd_audit=a,first_loss=first);save()
 for step in [1,2]:
  cp=RUN/'10-5'/('step'+str(step))/'model_final.pth'
  execute('MS5flip_step'+str(step),dist('eval_multiscale.py',['--checkpoint',str(cp),'--data_folder',frozen.DATA,'--scales','.5','.75','1','1.25','1.5','--flip','--output',str(OUT/('step'+str(step)+'_ms_flip.json'))]))
  execute('paired_step'+str(step),dist('collect_paired_predictions.py',['--checkpoint',str(cp),'--output',str(OUT/('step'+str(step)+'_paired.npz'))]))
 execute('quality_step2',dist('audit_relation.py',['--checkpoint',str(previous),'--sample_size','128','--output',str(OUT/'final_relation_quality.json')]))
 report();state.update(status='complete',phase='complete');save()
except Exception as e:
 state.update(status='failed',error=repr(e))
 if OUT.exists():save()
 raise

