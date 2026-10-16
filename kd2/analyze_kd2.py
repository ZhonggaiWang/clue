"""Registered comparison against same world2/per-GPU4 historical plain KD."""
from pathlib import Path
import json,hashlib
import numpy as np
from analyze_kd import GROUPS,grouped,errors,old_region_retention
ROOT=Path(__file__).resolve().parent
REF=ROOT.parent/'kd1'
OUT=ROOT/'reports'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def analyze():
 protocol=json.loads((ROOT/'experiment.json').read_text())
 result={'task':'10-5','seed':0,'protocol':protocol,'variants':{},'limitations':['Single training seed; image bootstrap represents evaluation sampling, not seed variability.','Historical plain shares runtime/world2/spg4/global8 but uses a different physical GPU pair.','Channel-weight matching does not match measured KL or gradient norm.','Offline quality uses fixed 128 heldout images after masks are fixed; GT does not enter training.']}
 arrays={};names=None
 for mode in ['plain','counter','relations']:
  root=REF if mode=='plain' else ROOT
  stages={};pair={}
  for step in [1,2]:
   run=root/'runs'/mode/'10-5'/('step'+str(step))
   met=json.loads((run/'metrics.jsonl').read_text().splitlines()[-1]);assert met['iteration']==8000
   ms=json.loads((root/'reports'/mode/('step'+str(step)+'_ms_flip.json')).read_text())
   audit=json.loads((run/'kd_audit.json').read_text());assert audit['world_size']==2 and audit['batches_per_rank']==6000
   cp=run/'model_final.pth';assert sha(cp)==ms['checkpoint_sha256']
   data=np.load(root/'reports'/mode/('step'+str(step)+'_paired.npz'))
   if names is None:names=data['names']
   assert np.array_equal(names,data['names']) and len(names)==1449
   assert str(data['checkpoint_sha256'])==ms['checkpoint_sha256']
   pair[step]=data['confusion'].copy();eligible=audit['kd_eligible_pixels']
   stages[str(step)]={'single_scale':met,'five_scale_flip':ms,'kd_audit':audit,'KD_mean_all_valid_pixels':audit['kd_kl_sum']/eligible}
   if mode!='plain':
    assert audit['kd_selected_channel_sum']<=audit['kd_eligible_channel_sum']
    stages[str(step)]['channel_coverage_percent']=100*audit['kd_selected_channel_sum']/audit['kd_eligible_channel_sum']
    stages[str(step)]['channel_weight_budget_percent']=100*audit['kd_channel_weight_sum']/audit['kd_eligible_channel_sum']
    stages[str(step)]['KL_relative_to_ordinary_same_outputs_percent']=100*audit['kd_kl_sum']/max(audit['kd_plain_kl_sum'],1e-12)
  arrays[mode]=pair[2];cm=pair[2].sum(0)
  result['variants'][mode]={'stages':stages,'final_groups':grouped(cm),'final_errors':errors(cm),'old_region_retention':old_region_retention(pair[1].sum(0),cm)}
  assert abs(grouped(cm)['all']-stages['2']['single_scale']['miou'])<1e-6
 result['comparisons']={}
 for mode in ['counter','relations']:
  rng=np.random.default_rng(20261014);draws=[];n=len(names)
  for _ in range(2000):
   idx=rng.integers(0,n,size=n);a=grouped(arrays['plain'][idx].sum(0));b=grouped(arrays[mode][idx].sum(0));draws.append([b[k]-a[k] for k in GROUPS])
  ci=np.quantile(np.array(draws),[.025,.975],axis=0)
  result['comparisons'][mode]={'SS_delta_pp':{k:result['variants'][mode]['final_groups'][k]-result['variants']['plain']['final_groups'][k] for k in GROUPS},'SS_paired_image_bootstrap95CI':{k:ci[:,i].tolist() for i,k in enumerate(GROUPS)}}
  result['comparisons'][mode]['MS_flip_delta_pp']=result['variants'][mode]['stages']['2']['five_scale_flip']['miou']-result['variants']['plain']['stages']['2']['five_scale_flip']['miou']
 result['pretraining_quality']=json.loads((OUT/'pretraining_relation_quality.json').read_text())
 result['source_hash_audit']={n:sha(ROOT/n)==h for n,h in json.loads((ROOT/'source_manifest.json').read_text()).items()};assert all(result['source_hash_audit'].values())
 with (OUT/'results_analysis.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
 lines=['CLUE证据KD优化结果；两个新策略，均为两卡、每卡batch4，总batch8。','原始PTC/C/F不变，SEG从0，C/F/KD从2000，固定future-background0/noPTC step0。']
 for mode,v in result['variants'].items():
  st=v['stages']['2'];lines.append(f"{mode}: 最终SS {st['single_scale']['miou']:.4f}，五尺度flip {st['five_scale_flip']['miou']:.4f}，SS旧类含背景 {v['final_groups']['old_with_background']:.4f}，新类 {v['final_groups']['new']:.4f}。")
 lines+=['单seed；普通KD使用既有同配置对照，物理GPU对不同；不能把图像bootstrap当训练重复实验。','通道权重匹配不意味着KL或梯度强度相同。GT只用于验证和固定掩码后的离线统计。']
 (OUT/'KD2_REPORT_ZH.txt').write_text('\n'.join(lines)+'\n')
 print(json.dumps(result['comparisons']),flush=True)
if __name__=='__main__':analyze()

