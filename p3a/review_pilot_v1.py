from pathlib import Path
import json,hashlib,sys,collections
import numpy as np
sys.path.insert(0,str(Path('src').resolve()))
from sgld_cv.transport import AccuracyMetric
ROOT=Path('results/p3a_bank_t4_pilot_v1');OUT=Path('exports/p3a_pilot_review_v1');OUT.mkdir(parents=True,exist_ok=True)
runs=json.loads((ROOT/'summary.json').read_text());assert len(runs)==132
assert all(r['completed_iterations']==2048 and not r['budget_censored'] for r in runs)
groups=collections.defaultdict(list)
for r in runs:groups[r['method'],r['batch'],r['h']].append(r)
rows=[]
for (m,n,h),rs in sorted(groups.items()):
 assert len(rs)==2
 for k in [128,512,2048]:
  ps=[next(p for p in r['checkpoints'] if p['iteration']==k) for r in rs]
  valid=all(r['failed_replicas']==0 and not r['budget_censored'] for r in rs) and all(p['status']=='ok' and p['failed_replicas']==0 for p in ps)
  q=dict(method=m,batch=n,h=h,iteration=k,valid=valid,failed_replicas=sum(p['failed_replicas'] for p in ps),failed_runs=sum(p['failed_replicas']>0 for p in ps),warm_seconds=float(np.mean([p['warm_total_seconds'] for p in ps])),cold_seconds=float(np.mean([p['cold_total_seconds'] for p in ps])),first_failure=min([p['failure_iteration'] for p in ps if p['failure_iteration']] or [0]))
  if valid:
   v=np.array([[[v[t] for t in ['sw2','variance_nrmse']] for v in p['references']] for p in ps])
   q.update(sw2=float(v.mean(0).max(0)[0]),variance_nrmse=float(v.mean(0).max(0)[1]),seed_metrics=v.tolist())
  rows.append(q)
# Pilot-informed shortlisting; freeze before accessing reserved selection seeds.
# Retain both endpoint-distribution and ergodic-variance objectives, with equal treatment of methods.
short=[]
for m in ['cv1','cv2','ula']:
 for n in ([32] if m=='ula' else [1,4,16,32,64,256]):
  eligible=[q for q in rows if q['method']==m and q['batch']==n and q['iteration']==2048 and q['valid']]
  for objective in ['sw2','variance_nrmse']:
   if eligible:
    b=min(eligible,key=lambda q:(q[objective],q['warm_seconds'],q['h']))
    ident=(m,n,b['h']);found=next((s for s in short if (s['method'],s['batch'],s['h'])==ident),None)
    if found:found['pilot_objectives'].append(objective)
    else:short.append(dict(method=m,batch=n,h=b['h'],pilot_objectives=[objective]))
failed=[dict(method=r['method'],batch=r['batch'],h=r['h'],seed=r['seed'],failed_replicas=r['failed_replicas'],failed_at=r['failed_at'],first_failure=min(x for x in r['failed_at'] if x)) for r in runs if r['failed_replicas']]
a=np.load(ROOT/'frozen_bank_assets/target.npz');s=np.load(ROOT/'frozen_bank_assets/scoring.npz');metric=AccuracyMetric('standardized_sliced_w2',a['anchor'],a['W'],s['projections']);rng=np.random.default_rng(202505)
floors={}
for R in [32,128]:
 vals=[metric.distance(s['first'][rng.choice(len(s['first']),R,replace=True)],s['second']) for _ in range(128)]
 floors[str(R)]=dict(mean=float(np.mean(vals)),quantiles=np.quantile(vals,[.025,.5,.975]).tolist(),interpretation='resampling approximation using stored reference cloud, not an exact iid posterior floor; no gate changes')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=dict(stage='P3A independent selection freeze v1',pilot_informed=True,pilot_summary_sha256=sha(ROOT/'summary.json'),pilot_archive_sha256=sha(Path('p3a/P3A_Bank_T4_Pilot_Results.zip')),candidates=short,selection_seeds=[202700,202701,202702,202703],replicas=128,checkpoints=[128,512,2048],runs=4*len(short),stage_seconds=2700,per_run_sampler_seconds=180,shortlist_rule='Per method/batch, union of stable-all-pilot-seeds minimum worst-reference seed-mean SW2 and minimum variance NRMSE at K2048; deterministic warm-time then h tie-break. No new step sizes or batches.',selection_rule='Per method/batch, use configurations with no failed/censored seed through K2048; minimize max(SW2/minimum SW2, variance_NRMSE/minimum variance_NRMSE), with both metrics worst across reference-specific seed means at K2048; warm-time then h tie-break. No new search if none eligible.',heldout_seeds_reserved=list(range(202800,202808)),heldout_consumed=False,initialization_seed=202506,schedule_seed=202507,failure='Unchanged pilot permanent nonfinite/radius rule. All failures retained; no survivor scoring. Any failed/censored seed excludes that configuration from selection.',cost='Same T4 float64 setup-inclusive cold/warm definition as pilot; synchronized sampling; compile attributed once per independent run for cold; actual first-shape compile cache effects reported, not independent cold process timings.',reference='Unchanged frozen Bank assets, both reference clouds; passed original gate',metrics='Same standardized replica SW2 and relative variance NRMSE of per-replica ergodic variance estimates; no thresholds changed',scope='Selection only. Batch-cost frontier and accuracy statements remain preliminary until independent held-out. Weak/unstable batch1 and batch4 retained for diagnosis, not dropped.')
Path('p3a/SELECTION_PROTOCOL_v1.json').write_text(json.dumps(protocol,indent=2)+'\n')
(OUT/'review.json').write_text(json.dumps(dict(configurations=rows,failures=failed,reference_resampling=floors,shortlist=short,gradient_mse=json.loads((ROOT/'gradient_mse.json').read_text()),input_sha256={f:sha(ROOT/f) for f in ['summary.json','validation.json','gradient_mse.json']}),indent=2)+'\n')
lines=['# P3A T4 pilot review','',f'132/132 runs completed; {len(failed)} runs contain failures. Two seeds ×32 mode-start independent replicas; no budget censoring. This is tuning evidence, not held-out inference.','', '## K2048 full grid','', '|Method|Batch|h|Failed replicas /64|SW2|Variance NRMSE|Warm seconds|Cold seconds|','|---|---:|---:|---:|---:|---:|---:|---:|']
for q in rows:
 if q['iteration']!=2048:continue
 lines.append(f"|{q['method']}|{q['batch']}|{q['h']}|{q['failed_replicas']}|{q.get('sw2',float('nan')):.4g}|{q.get('variance_nrmse',float('nan')):.4g}|{q['warm_seconds']:.4g}|{q['cold_seconds']:.4g}|")
lines+=['','## Interpretation','', 'Increasing batch size strongly improves CV1 accuracy and stability, with a large measured T4 time penalty at n256. CV2 substantially reduces gradient MSE, but rare excursions remain possible: at n64, low step sizes give poor variance NRMSE even without triggering the frozen divergence gate. Lack of failures in 64 lanes does not certify stability.','',f'Reference-cloud resampling SW2 floors: R32 mean {floors["32"]["mean"]:.4f}; R128 mean {floors["128"]["mean"]:.4f}. These are empirical finite-cloud estimates. The original SW2<=0.3 gate remains unchanged; R32 endpoint metrics near the floor should not drive superiority claims.','',f'Selection shortlist contains {len(short)} configurations ×4 independent seeds = {4*len(short)} runs, R128, K up to2048. Preserve both SW2-best and variance-best pilot choices per method/batch. Selection does not consume held-out seeds.','', '## Failures','', '|Method|Batch|h|Seed|Failed /32|First iteration|','|---|---:|---:|---:|---:|---:|']
for f in failed:lines.append(f"|{f['method']}|{f['batch']}|{f['h']}|{f['seed']}|{f['failed_replicas']}|{f['first_failure']}|")
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(failed_runs=len(failed),floors=floors,selection_runs=protocol['runs'],shortlist=short),indent=2))
