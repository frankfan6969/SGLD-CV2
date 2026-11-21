"""Three-method frozen-grid inference, retaining failures and censored mass."""
from pathlib import Path
import sys,collections,json,os
import numpy as np
from common import *
def analyze(F,O,c,seeds):
 O.mkdir(parents=True,exist_ok=True);runs=load(F/'validation_summary.json');freeze=load(F/'FROZEN_CANDIDATES.json');groups=collections.defaultdict(list)
 for r in runs:groups[r['method'],r['batch'],r['h']].append(r)
 rows=[];values=[];warm=[];cold=[];methods=['cv1','cv2','ula']
 for key,rr in sorted(groups.items()):
  rr=sorted(rr,key=lambda r:r['seed']);eligible=[r['seed'] for r in rr]==seeds and all(not r['failed_replicas'] and not r['budget_censored'] and r['completed_iterations']==freeze['max_iterations'] for r in rr)
  for k in [k for k in c['validation']['checkpoints'] if k<=freeze['max_iterations']]:
   row=dict(method=key[0],batch=key[1],h=key[2],K=k,eligible=eligible,failed_replicas=sum(r['failed_replicas'] for r in rr),censored_runs=sum(r['budget_censored'] for r in rr))
   if eligible:
    pp=[next(p for p in r['checkpoints'] if p['iteration']==k) for r in rr];row['eligible_index']=len(values);values.append([[[v[t] for t in ['sw2','variance_nrmse']] for v in p['references']] for p in pp]);warm.append([p['warm_total_seconds'] for p in pp]);cold.append([p['cold_total_seconds'] for p in pp])
   rows.append(row)
 if not values:raise RuntimeError('No eligible complete grid; report failure without efficiency ranking')
 V=np.array(values).transpose(1,0,2,3);T=np.array(warm).T;C=np.array(cold).T;valid=[q for q in rows if q['eligible']];ix=np.random.default_rng(c['bootstrap']['seed']).integers(0,len(seeds),(c['bootstrap']['draws'],len(seeds)));bs=np.concatenate([V[b].mean(1) for b in np.array_split(ix,40)]);mean=V.mean(0);sd=bs.std(0,ddof=1);crit=float(np.quantile(np.max(np.abs(bs-mean)/np.maximum(sd,1e-12),axis=(1,2,3)),.95));acc=mean.max(1);upper=(mean+crit*sd).max(1);ba=bs.max(2);bt=T[ix].mean(1);bc=C[ix].mean(1)
 for i,q in enumerate(valid):q.update(sw2=float(acc[i,0]),variance_nrmse=float(acc[i,1]),sw2_simultaneous_upper=float(upper[i,0]),variance_simultaneous_upper=float(upper[i,1]),warm_seconds=float(T[:,i].mean()),cold_seconds=float(C[:,i].mean()))
 sweeps=[]
 for cost,tt in [('warm',bt),('cold',bc)]:
  for eps in c['metrics']['variance_thresholds']:
   row=dict(cost=cost,variance_threshold=eps,methods={},ratios={});bmin={}
   for m in methods:
    jj=np.array([i for i,q in enumerate(valid) if q['method']==m],dtype=int);choices={}
    for name,a in [('point_choice',acc),('simultaneous_supported_choice',upper)]:
     good=jj[(a[jj,0]<=.3)&(a[jj,1]<=eps)];best=None if not len(good) else int(good[np.argmin([valid[i][cost+'_seconds'] for i in good])]);choices[name]=None if best is None else {k:valid[best][k] for k in ['eligible_index','method','batch','h','K','sw2','variance_nrmse',cost+'_seconds']}
    bm=np.where((ba[:,jj,0]<=.3)&(ba[:,jj,1]<=eps),tt[:,jj],np.inf).min(1) if len(jj) else np.full(len(ix),np.inf);finite=np.isfinite(bm);bmin[m]=bm;choices.update(bootstrap_not_reached_fraction=float((~finite).mean()),conditional_time_ci95=np.quantile(bm[finite],[.025,.975]).tolist() if finite.any() else None,no_eligible_complete_configuration=not len(jj));row['methods'][m]=choices
   for numerator,denominator in [('cv1','cv2'),('ula','cv2')]:
    xx=row['methods'][numerator]['point_choice'];yy=row['methods'][denominator]['point_choice'];both=np.isfinite(bmin[numerator])&np.isfinite(bmin[denominator]);xxs=row['methods'][numerator]['simultaneous_supported_choice'];yys=row['methods'][denominator]['simultaneous_supported_choice'];row['ratios'][numerator+'_over_'+denominator]=dict(point_ratio=xx[cost+'_seconds']/yy[cost+'_seconds'] if xx and yy else None,supported_ratio=xxs[cost+'_seconds']/yys[cost+'_seconds'] if xxs and yys else None,bootstrap_both_reached_fraction=float(both.mean()),conditional_ci95=np.quantile(bmin[numerator][both]/bmin[denominator][both],[.025,.975]).tolist() if both.any() else None)
   sweeps.append(row)
 out=dict(scope='New extension; bounded frozen candidates, fresh paired seeds; full-horizon exclusions. Not a continuous optimum. Bootstrap approximate/conditional on fixed refs; point and supported eligibility differ.',validation_runs=len(runs),pilot_runs=len(load(F/'pilot_summary.json')),failed_runs=sum(r['failed_replicas']>0 for r in runs),failed_replicas=sum(r['failed_replicas'] for r in runs),budget_censored_runs=sum(r['budget_censored'] for r in runs),simultaneous_q95=crit,configuration_checkpoints=rows,threshold_sweep=sweeps);save(O/'analysis.json',out)
 md=['# Three-method independent-validation efficiency frontier','','Methods share target,full-H metric,reference,512 replicas and8 fresh seeds; CV1/CV2 candidate settings come from original pilot freeze, ULA steps come from new pilot. The added method and paired comparison are a review-motivated extension, not the original confirmatory experiment.','','|Variance|CV1 warm s|CV2 warm s|ULA warm s|CV1/CV2|ULA/CV2|','|---|---:|---:|---:|---:|---:|']
 def fmt(q):return 'Not Reached' if q is None else f'{q:.4f}'
 for eps in [.5,.4,.3,.2]:
  q=next(q for q in sweeps if q['cost']=='warm' and q['variance_threshold']==eps);times=[q['methods'][m]['point_choice'] for m in methods];md.append('|'+str(eps)+'|'+'|'.join(fmt(z['warm_seconds']) if z else 'Not Reached' for z in times)+'|'+fmt(q['ratios']['cv1_over_cv2']['point_ratio'])+'|'+fmt(q['ratios']['ula_over_cv2']['point_ratio'])+'|')
 md+=['',f"Validation: {len(runs)} runs, {out['failed_replicas']} failed replicas and {out['budget_censored_runs']} censored runs. Critical simultaneous maxstat {crit:.6f}.",'','All41thresholds,warm/cold,supported choices,conditional intervals andNot-Reached mass remain in analysis.json. A censored/incomplete method is not declared slower. Accuracy is fixed-reference finite-cloud and finite-horizon ergodic variance error from steps1..K, without separate burn-in. Cold includes actual cached compilation attribution, not repeated fresh-process startup. No universal superiority, exact-posterior or operation-complexity conclusion follows.'];(O/'REPORT.md').write_text('\n'.join(md)+'\n')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained');colors={'cv1':'#27659b','cv2':'#c55a30','ula':'#3c8872'}
 for ax,metric in zip(axes[0],['sw2','variance_nrmse']):
  for m in methods:
   pts=sorted([q for q in valid if q['method']==m],key=lambda q:(q['warm_seconds'],q[metric]));front=[];best=np.inf
   for q in pts:
    if q[metric]<best:front.append(q);best=q[metric]
   ax.scatter([q['warm_seconds'] for q in pts],[q[metric] for q in pts],s=10,alpha=.25,color=colors[m]);ax.plot([q['warm_seconds'] for q in front],[q[metric] for q in front],color=colors[m],label=m.upper())
  ax.set(xscale='log',xlabel='Setup-inclusive warm CPU seconds',ylabel=metric);ax.legend();ax.grid(alpha=.2)
 for ax,cost in zip(axes[1],['warm','cold']):
  q=[q for q in sweeps if q['cost']==cost]
  for m in methods:ax.plot([r['variance_threshold'] for r in q],[r['methods'][m]['point_choice'][cost+'_seconds'] if r['methods'][m]['point_choice'] else np.nan for r in q],color=colors[m],label=m.upper())
  ax.set(yscale='log',xlabel='Variance tolerance; SW2 <= .3',ylabel=cost+' time to joint accuracy');ax.invert_xaxis();ax.grid(alpha=.2);ax.legend()
 fig.savefig(O/'frontiers.png',dpi=160);plt.close(fig);return out
if __name__=='__main__':
 c=load(H/'PROTOCOL.json');name=sys.argv[1];t=next(t for t in c['targets'] if t['id']==name);analyze(D/name,E/name,c,t['validation_seeds'])
