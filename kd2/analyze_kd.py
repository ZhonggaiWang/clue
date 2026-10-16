"""Final registered comparison; bootstrap represents image sampling, not seed variance."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reports'
GROUPS={'all':list(range(21)),'old_with_background':list(range(16)),
        'old_foreground':list(range(1,16)),'new':list(range(16,21)),
        'initial_foreground':list(range(1,11)),'first_added':list(range(11,16))}
def ious(cm):
 den=cm.sum(0)+cm.sum(1)-np.diag(cm)
 return np.divide(np.diag(cm)*100,den,out=np.full(len(den),np.nan),where=den>0)
def grouped(cm):
 v=ious(cm)
 return {k:float(np.nanmean(v[ids])) for k,ids in GROUPS.items()}
def errors(cm):
 old=cm[1:16].sum();new=cm[16:21].sum()
 return {'old_to_new_percent':float(cm[1:16,16:21].sum()/max(old,1)*100),
         'old_to_background_percent':float(cm[1:16,0].sum()/max(old,1)*100),
         'new_to_old_percent':float(cm[16:21,1:16].sum()/max(new,1)*100),
         'new_to_background_percent':float(cm[16:21,0].sum()/max(new,1)*100)}
def old_region_retention(before,after):
 # Identical full-val images and identical GT0..15 support; new GT rows excluded.
 before=before.copy();after=after.copy()
 before[16:]=0;after[16:]=0
 bv,av=ious(before)[:16],ious(after)[:16]
 return {'scope':'Same 1449 images, restrict GT to0..15 for both models; future/new GT excluded',
         'before_miou':float(np.nanmean(bv)),'after_miou':float(np.nanmean(av)),
         'forgetting_pp':float(np.nanmean(bv-av)),
         'before_foreground_miou':float(np.nanmean(bv[1:])),
         'after_foreground_miou':float(np.nanmean(av[1:])),
         'per_class_forgetting_pp':(bv-av).tolist()}
def analyze():
 result={'task':'10-5','seed':0,'variants':{},'limitations':[
  'Single training seed; bootstrap covers evaluation-image sampling, not training variability.',
  'Selective KD reduces total KD mass as well as changing locations; no matched-mass control.',
  'No same-runtime no-KD control; historical A800 results cannot isolate the net KD effect.']}
 arrays={};names=None
 for mode in ['plain','evidence']:
  stages={};pair={}
  for step in [1,2]:
   run=ROOT/'runs'/mode/'10-5'/('step'+str(step))
   met=json.loads((run/'metrics.jsonl').read_text().splitlines()[-1])
   assert met['iteration']==8000
   ms=json.loads((OUT/mode/('step'+str(step)+'_ms_flip.json')).read_text())
   audit=json.loads((run/'kd_audit.json').read_text());eligible=audit['kd_eligible_pixels']
   coverage={k:100*v/eligible for k,v in audit.items() if k.endswith('_pixels')}
   data=np.load(OUT/mode/('step'+str(step)+'_paired.npz'))
   if names is None:names=data['names']
   assert np.array_equal(names,data['names'])
   assert str(data['checkpoint_sha256'])==ms['checkpoint_sha256']
   pair[step]=data['confusion'].copy()
   stages[str(step)]={'single_scale':met,'five_scale_flip':{k:ms[k] for k in ['miou','iou','old_including_background_miou','new_miou','checkpoint_sha256']},
    'kd_coverage_percent':coverage,'kd_audit':audit,
    'mean_kd_over_all_valid_pixels':audit['kd_kl_sum']/eligible,
    'mean_kd_over_selected_pixels':audit['kd_kl_sum']/max(audit['kd_pixels'],1)}
  arrays[mode]=pair[2]
  cm=pair[2].sum(0)
  result['variants'][mode]={'stages':stages,'final_groups':grouped(cm),'final_errors':errors(cm),
    'old_region_retention':old_region_retention(pair[1].sum(0),cm)}
  assert abs(grouped(cm)['all']-stages['2']['single_scale']['miou'])<1e-6
 result['delta_evidence_minus_plain']={k:result['variants']['evidence']['final_groups'][k]-result['variants']['plain']['final_groups'][k] for k in GROUPS}
 rng=np.random.default_rng(20261013);n=len(names);draws=[]
 for _ in range(2000):
  idx=rng.integers(0,n,size=n)
  a=grouped(arrays['plain'][idx].sum(0));b=grouped(arrays['evidence'][idx].sum(0))
  draws.append([b[k]-a[k] for k in GROUPS])
 ci=np.quantile(np.array(draws),[.025,.975],axis=0)
 result['paired_image_bootstrap']={'replicates':2000,'seed':20261013,'images':n,
  'delta95CI':{k:ci[:,i].tolist() for i,k in enumerate(GROUPS)}}
 for kind,key in [('single_scale','single_scale'),('five_scale_flip','five_scale_flip')]:
  scores={m:result['variants'][m]['stages']['2'][key]['miou'] for m in ['plain','evidence']}
  result[kind+'_winner']=max(scores,key=scores.get)
 with (OUT/'results_analysis.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
 print(json.dumps({'delta':result['delta_evidence_minus_plain'],'CI':result['paired_image_bootstrap'],
                   'single_scale_winner':result['single_scale_winner']}))
if __name__=='__main__':analyze()
