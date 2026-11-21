"""Fixed pilot choices, paired whole-seed inference; never tune on validation."""
import csv
import numpy as np
from common import load,save,sha,R

def analyze(F,O,c,t):
 O.mkdir(parents=True,exist_ok=True);freeze=load(F/'FROZEN_CHOICES.json');runs=load(F/'validation_summary.json');seeds=t['validation_seeds'];B=c['bootstrap']['draws'];lookup={};allrows=[];checks=0
 for g in freeze['groups']:
  rr=sorted([r for r in runs if (r['method'],r['batch'],r['h'])==(g['method'],g['batch'],g['h'])],key=lambda r:r['seed'])
  eligible=[r['seed'] for r in rr]==seeds and all(not r['failed_replicas'] and not r['budget_censored'] and r['completed_iterations']==g['max_iterations'] for r in rr)
  for k in g['checkpoints']:
   q=dict(group_id=g['id'],method=g['method'],batch=None if g['method']=='ula' else g['batch'],h=g['h'],K=k,eligible=eligible,failed_replicas=sum(r['failed_replicas'] for r in rr),failed_runs=sum(r['failed_replicas']>0 for r in rr),censored_runs=sum(r['budget_censored'] for r in rr),expected_seeds=len(seeds),recorded_seeds=len(rr))
   if eligible:
    pp=[next(p for p in r['checkpoints'] if p['iteration']==k) for r in rr]
    v=np.array([[[x[z] for z in ['sw2','variance_nrmse']] for x in p['references']] for p in pp])
    times={z:np.array([p[z+'_seconds'] if z=='sampling' else p[z+'_total_seconds'] for p in pp]) for z in ['sampling','warm','cold']}
    for r,p in zip(rr,pp):
     np.testing.assert_allclose(p['warm_total_seconds'],r['anchor_seconds_attributed']+r['setup_seconds']+p['sampling_seconds'],rtol=1e-12,atol=1e-12)
     np.testing.assert_allclose(p['cold_total_seconds'],p['warm_total_seconds']+r['compile_seconds'],rtol=1e-12,atol=1e-12);checks+=2
    q['_values']=v;q['_times']=times
   allrows.append(q);lookup[g['id'],k]=q
 valid=[q for q in allrows if q['eligible']];ix=np.random.default_rng(c['bootstrap']['seed']).integers(0,len(seeds),(B,len(seeds)))
 if valid:
  V=np.stack([q['_values'] for q in valid],axis=1);mean=V.mean(0);bs=np.concatenate([V[b].mean(1) for b in np.array_split(ix,40)]);sd=bs.std(0,ddof=1);crit=float(np.quantile((abs(bs-mean)/np.maximum(sd,1e-12)).max(axis=(1,2,3)),.95))
  upper=(mean+crit*sd).max(1)
  for i,q in enumerate(valid):
   a=mean[i].max(0);q.update(sw2=float(a[0]),variance_nrmse=float(a[1]),sw2_simultaneous_upper=float(upper[i,0]),variance_simultaneous_upper=float(upper[i,1]));q['_boot_accuracy']=bs[:,i].max(1)
   for cost,tt in q['_times'].items():q[cost+'_seconds']=float(tt.mean());q['_'+cost+'_boot']=tt[ix].mean(1)
 else:crit=None
 def ci(v):return np.quantile(v,[.025,.975]).tolist() if len(v) else None
 output=[];chosen={}
 for s in freeze['selections']:
  q=lookup.get((s['group_id'],s['K']));row={k:s[k] for k in ['cost','variance_threshold','method','group_id','K']};row['pilot_selected']=s['pilot_choice'] is not None;eligible=bool(q and q['eligible']);row['eligible']=eligible
  row.update(point_attained=False,simultaneous_supported=False,status='No pilot choice' if q is None else 'Whole configuration excluded',seconds=None,time_ci95=None,bootstrap_not_reached_fraction=1.)
  if eligible:
   cost=s['cost'];eps=s['variance_threshold'];passed=q['sw2']<=.3 and q['variance_nrmse']<=eps;support=q['sw2_simultaneous_upper']<=.3 and q['variance_simultaneous_upper']<=eps;ok=(q['_boot_accuracy'][:,0]<=.3)&(q['_boot_accuracy'][:,1]<=eps)
   row.update(h=q['h'],batch=q['batch'],sw2=q['sw2'],variance_nrmse=q['variance_nrmse'],sw2_simultaneous_upper=q['sw2_simultaneous_upper'],variance_simultaneous_upper=q['variance_simultaneous_upper'],point_attained=bool(passed),simultaneous_supported=bool(support),status='Reached' if passed else 'Not Reached',seconds=q[cost+'_seconds'],time_ci95=ci(q['_'+cost+'_boot']),conditional_attained_time_ci95=ci(q['_'+cost+'_boot'][ok]),bootstrap_not_reached_fraction=float((~ok).mean()))
   chosen[s['cost'],eps,s['method']]=(q,ok)
  output.append(row)
 ratios=[]
 for cost in ['sampling','warm','cold']:
  for eps in c['metrics']['variance_thresholds']:
   for num,den in [('cv1','cv2'),('ula','cv2')]:
    a=next(r for r in output if (r['cost'],r['variance_threshold'],r['method'])==(cost,eps,num));b=next(r for r in output if (r['cost'],r['variance_threshold'],r['method'])==(cost,eps,den));qa=chosen.get((cost,eps,num));qb=chosen.get((cost,eps,den));row=dict(cost=cost,variance_threshold=eps,ratio=num+'_over_'+den,point_ratio=a['seconds']/b['seconds'] if a['point_attained'] and b['point_attained'] else None,supported_ratio=a['seconds']/b['seconds'] if a['simultaneous_supported'] and b['simultaneous_supported'] else None,bootstrap_both_attained_fraction=0.,conditional_ratio_ci95=None)
    if qa and qb:
     both=qa[1]&qb[1];rat=qa[0]['_'+cost+'_boot']/qb[0]['_'+cost+'_boot'];row.update(bootstrap_both_attained_fraction=float(both.mean()),conditional_ratio_ci95=ci(rat[both]),fixed_choice_runtime_ratio_ci95=ci(rat))
    ratios.append(row)
 for q in allrows:
  for key in list(q):
   if key.startswith('_'):del q[key]
 result=dict(target=t['id'],scope=c['inference'],selection='Every h,n,K frozen before independent validation; no validation reoptimization',replicas=512,seeds=seeds,expected_runs=len(freeze['groups'])*len(seeds),completed_runs=len(runs),failed_replicas=sum(r['failed_replicas'] for r in runs),failed_runs=sum(r['failed_replicas']>0 for r in runs),censored_runs=sum(r['budget_censored'] for r in runs),failure_denominator=sum(r['replicas'] for r in runs),failure_rate=sum(r['failed_replicas'] for r in runs)/sum(r['replicas'] for r in runs),simultaneous_critical=crit,configuration_checkpoints=allrows,fixed_choices=output,ratios=ratios,timing_definition=c['cost_definitions'],audit=dict(timing_identities_checked=checks,selection_sha256=sha(F/'FROZEN_CHOICES.json'),protocol_sha256=sha(F.parent/'PROTOCOL.json')))
 save(O/'analysis.json',result)
 for name,rows in [('fixed_choices.csv',output),('ratios.csv',ratios),('configuration_checkpoints.csv',allrows)]:
  keys=list(dict.fromkeys(k for r in rows for k in r))
  with (O/name).open('w') as f:
   w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 md=['# Fixed-choice held-out validation: '+t['id'],'',f"{len(runs)}/{result['expected_runs']} runs; R=512,8 independent paired seeds; {result['failed_replicas']} failed replicas/{result['failure_denominator']} ({result['failure_rate']:.3%}); {result['censored_runs']} censored runs.",'','All choices were selected on the earlier two-seed R64 pilot. No replacement h,n,K is chosen using held-out results. A failed/incomplete seed excludes the whole parameter group at every frozen checkpoint, not just the failed seed. Review-motivated extension, not the original P3B confirmatory analysis.','']
 def fmt(x):return 'Not Reached' if x is None else f'{x:.4f}'
 for cost in ['sampling','warm','cold']:
  md += ['## '+cost,'', '|Variance tolerance; SW2 <= 0.3|CV1 seconds|CV2 seconds|ULA seconds|CV1/CV2|ULA/CV2|','|---|---:|---:|---:|---:|---:|']
  for eps in [.5,.4,.3,.2]:
   ss=[next(q for q in output if (q['cost'],q['variance_threshold'],q['method'])==(cost,eps,m)) for m in ['cv1','cv2','ula']];rr=[next(q for q in ratios if (q['cost'],q['variance_threshold'],q['ratio'])==(cost,eps,z)) for z in ['cv1_over_cv2','ula_over_cv2']];md.append('|'+str(eps)+'|'+'|'.join(fmt(q['seconds']) if q['point_attained'] else 'Not Reached' for q in ss)+'|'+'|'.join(fmt(q['point_ratio']) for q in rr)+'|')
  md += ['','These are point-attainment times for preselected configurations, not minima reselected from held-out data. See fixed_choices.csv for simultaneous-upper-bound-supported attainment and time CIs, and ratios.csv for conditional ratio intervals and bootstrap non-attainment mass.','']
 md+=['## Timing audit','', 'The previous pilot warm fields already included stored mode-fit time plus current setup and sampling. Calling those numbers sampling-only in conversation was incorrect. The formulas agree with original P3B warm accounting, but measurements from different sessions/hardware must not be pooled.','']+[f'- **{k}**: {v}' for k,v in c['cost_definitions'].items()]+['','## Limits','', 'Accuracy is conditional on two fixed reference clouds and finite projections. Variance NRMSE uses per-replica finite-horizon ergodic sums from mode over steps1..K without separate burn-in. Whole-seed bootstrap intervals are approximate with eight seeds; no claim of exact posterior coverage. Historical mode-fit attribution is fixed, not remeasured on this machine. Cold measures shape-specific compilation in a shared/cache-enabled process, not repeated fresh-process starts. No automatic additional dataset, tuning, adaptive-metric study or metric ablation is launched.','']
 (O/'REPORT.md').write_text('\n'.join(md))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
 for ax,cost in zip(axes,['sampling','warm','cold']):
  for m,color in [('cv1','#27659b'),('cv2','#c55a30'),('ula','#3c8872')]:
   qs=[q for q in output if q['cost']==cost and q['method']==m];ax.plot([q['variance_threshold'] for q in qs],[q['seconds'] if q['point_attained'] else np.nan for q in qs],label=m.upper(),color=color)
  ax.set(xlabel='Variance NRMSE tolerance; SW2 <= .3',ylabel=cost+' seconds',yscale='log');ax.invert_xaxis();ax.grid(alpha=.2);ax.legend()
 fig.suptitle(t['id']+' — pilot-frozen choices on independent seeds');fig.savefig(O/'fixed_choice_times.png',dpi=180);plt.close(fig)
 return result
