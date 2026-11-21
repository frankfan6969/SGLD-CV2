"""Frozen P3A heldout analysis; no sampler tuning or changes to raw outputs."""
from pathlib import Path
import json,hashlib,sys,collections
import numpy as np
sys.path.insert(0,str(Path('src').resolve()))
from sgld_cv.transport import AccuracyMetric
from threadpoolctl import threadpool_limits
ROOT=Path('results/p3a_bank_t4_heldout_v1');OUT=Path('exports/p3a_heldout_review_v1');OUT.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
c=json.loads((ROOT/'PROTOCOL.json').read_text());rs=json.loads((ROOT/'summary.json').read_text());manifest=json.loads((ROOT/'manifest.json').read_text())
assert manifest['status']=='complete' and len(rs)==c['runs']==96
assert c==json.loads(Path('p3a/HELDOUT_PROTOCOL_v1.json').read_text())
assert sha(Path('results/p3a_bank_t4_selection_v1/summary.json'))==c['selection_summary_sha256']
assert sha(Path('results/p3a_bank_t4_selection_v1/SELECTED_CONFIGURATIONS.json'))==c['selected_configurations_sha256']
assert sha(Path('results/p3a_bank_t4_pilot_v1/summary.json'))==c['pilot_summary_sha256']
assert set(c['heldout_seeds']).isdisjoint(c['selection_seeds']+c['pilot_seeds'])
assert all(r['completed_iterations']==2048 and r['replicas']==512 and not r['budget_censored'] for r in rs)
g=collections.defaultdict(list)
for r in rs:g[r['method'],r['batch'],r['h']].append(r)
assert len(g)==12 and all(sorted(r['seed'] for r in group)==c['heldout_seeds'] for group in g.values())
a=np.load(ROOT/'frozen_bank_assets/target.npz');ref=np.load(ROOT/'frozen_bank_assets/scoring.npz');metric=AccuracyMetric('standardized_sliced_w2',a['anchor'],a['W'],ref['projections']);refs=[ref['first'],ref['second']]
# Paired bootstrap: identical seed resampling indices for every configuration and metric.
boot=np.random.default_rng(202812).integers(0,8,size=(10000,8))
ci=lambda x:np.quantile(np.asarray(x),[.025,.975]).tolist()
rows=[];lookup={};validated=0;swvalidated=0;failures=[];tails=[]
with threadpool_limits(limits=1):
 for (m,n,h),group in sorted(g.items()):
  group=sorted(group,key=lambda r:r['seed']);valid=all(r['failed_replicas']==0 and not r['budget_censored'] for r in group)
  for r in group:
   if r['failed_replicas']:
    first=min(i for i in r['failed_at'] if i);failures.append(dict(method=m,batch=n,h=h,seed=r['seed'],failed_replicas=r['failed_replicas'],first_failure=first,failed_at=r['failed_at']))
  for k in c['checkpoints']:
   ps=[next(p for p in r['checkpoints'] if p['iteration']==k) for r in group]
   warm=np.array([p['warm_total_seconds'] for p in ps]);cold=np.array([p['cold_total_seconds'] for p in ps])
   q=dict(method=m,batch=n,h=h,iteration=k,valid_full_configuration=valid,failed_runs=sum(r['failed_replicas']>0 for r in group),failed_replicas=sum(r['failed_replicas'] for r in group),total_replicas=4096,first_failure=min([f for r in group for f in r['failed_at'] if f] or [0]),warm_seconds=float(warm.mean()),warm_ci95=ci(warm[boot].mean(1)),cold_seconds=float(cold.mean()),cold_ci95=ci(cold[boot].mean(1)),seed_warm_seconds=warm.tolist(),seed_cold_seconds=cold.tolist())
   # No aggregate accuracy for a formally ineligible configuration, even for earlier good prefixes.
   if valid:
    v=np.array([[[z[t] for t in ['sw2','variance_nrmse']] for z in p['references']] for p in ps]);bs=v[boot].mean(1).max(1);mean=v.mean(0).max(0)
    q.update(sw2=float(mean[0]),variance_nrmse=float(mean[1]),sw2_ci95=ci(bs[:,0]),variance_nrmse_ci95=ci(bs[:,1]),reference_specific_seed_means=v.mean(0).tolist(),seed_accuracy=v.tolist())
    for r,p in zip(group,ps):
     state=np.load(ROOT/f"checkpoint_{m}_n{n}_h{h}_seed{r['seed']}_K{k}.npz");sums=state['functional_sums']/k;d=len(a['anchor']);var=sums[:,d:]-sums[:,:d]**2
     assert np.all(state['failed_at']==0)
     for j,reference in enumerate(refs):
      vref=((reference-a['anchor'])@a['W']).var(0);recomputed=float(np.sqrt(np.mean(((var-vref)/vref)**2,axis=0)).mean());np.testing.assert_allclose(recomputed,p['references'][j]['variance_nrmse'],rtol=1e-11,atol=1e-12)
     if r['seed']==202800 and k==2048:
      for j,reference in enumerate(refs):np.testing.assert_allclose(metric.distance(state['endpoints'],reference),p['references'][j]['sw2'],rtol=1e-11,atol=1e-12)
      swvalidated+=1
     validated+=1
     if k==2048:
      vref=((refs[0]-a['anchor'])@a['W']).var(0);error=(var-vref)/vref;lane_rms=np.sqrt(np.mean(error*error,axis=1));sq=np.sum(error*error,axis=1);top=max(1,int(np.ceil(.01*len(sq))))
      tails.append(dict(method=m,batch=n,seed=r['seed'],median_lane_relative_variance_error=float(np.median(lane_rms)),p99_lane_error=float(np.quantile(lane_rms,.99)),max_lane_error=float(lane_rms.max()),top_1percent_fraction_squared_error=float(np.sort(sq)[-top:].sum()/sq.sum())))
   else:q.update(sw2=None,variance_nrmse=None,reason='Any failed seed excludes complete configuration; no survivor accuracy or prefix frontier')
   rows.append(q);lookup[m,n,h,k]=q
fixed=[]
for spec in c['primary_fixed_choices']:
 q=dict(mode=spec['mode'],threshold=spec['threshold'],method=spec['method'],choice=spec['choice'])
 if not spec['choice']:q.update(status='no_selection_fixed_choice',reached=False)
 else:
  z=spec['choice'];r=lookup[z['method'],z['batch'],z['h'],z['iteration']]
  q.update(warm_seconds=r['warm_seconds'],cold_seconds=r['cold_seconds'],warm_ci95=r['warm_ci95'],cold_ci95=r['cold_ci95'],sw2=r['sw2'],variance_nrmse=r['variance_nrmse'])
  if not r['valid_full_configuration']:q.update(status='failed_configuration',reached=False)
  else:
   v=np.array(r['seed_accuracy']);bs=v[boot].mean(1).max(1);reach=(r['variance_nrmse']<=q['threshold'] and (q['mode']=='variance_only' or r['sw2']<=.3));reachboot=bs[:,1]<=q['threshold']
   if q['mode']=='joint':reachboot &= bs[:,0]<=.3
   q.update(status='reached' if reach else 'not_reached',reached=bool(reach),bootstrap_target_attainment_fraction=float(reachboot.mean()),sw2_ci95=r['sw2_ci95'],variance_nrmse_ci95=r['variance_nrmse_ci95'])
 fixed.append(q)
comparisons=[]
for mode in ['variance_only','joint']:
 for threshold in [.5,.4,.3,.2]:
  choices=[next(q for q in fixed if (q['mode'],q['threshold'],q['method'])==(mode,threshold,m)) for m in ['cv1','cv2']]
  q=dict(mode=mode,threshold=threshold,cv1_status=choices[0]['status'],cv2_status=choices[1]['status'],warm_speedup=None,cold_speedup=None)
  if all(z['reached'] for z in choices):
   rr=[lookup[z['choice']['method'],z['choice']['batch'],z['choice']['h'],z['choice']['iteration']] for z in choices]
   for typ in ['warm','cold']:
    v=[np.array(z['seed_'+typ+'_seconds']) for z in rr];q[typ+'_speedup']=float(v[0].mean()/v[1].mean());q[typ+'_speedup_ci95']=ci(v[0][boot].mean(1)/v[1][boot].mean(1))
  comparisons.append(q)
# Exploratory min across the fixed heldout grid: separately labelled, never substitutes a fixed primary choice.
exploratory=[]
for mode in ['variance_only','joint']:
 for threshold in [.5,.4,.3,.2]:
  for m in ['cv1','cv2','ula']:
   eligible=[r for r in rows if r['method']==m and r['valid_full_configuration'] and r['variance_nrmse']<=threshold and (mode=='variance_only' or r['sw2']<=.3)]
   for typ in ['warm','cold']:
    best=min(eligible,key=lambda r:r[typ+'_seconds']) if eligible else None
    exploratory.append(dict(mode=mode,threshold=threshold,method=m,cost=typ,status='reached' if best else 'not_reached',choice=None if best is None else {k:best[k] for k in ['method','batch','h','iteration','sw2','variance_nrmse',typ+'_seconds']}))
validation=dict(runs=96,seeds_disjoint=True,all_completed=True,budget_censored=0,failed_runs=len(failures),failed_replicas=sum(x['failed_replicas'] for x in failures),nrmse_checkpoint_reconstructions=validated,sw2_final_checkpoint_reconstructions=swvalidated,source_hashes={str(p):sha(p) for p in [ROOT/'summary.json',ROOT/'PROTOCOL.json',Path('p3a/P3A_Bank_T4_Heldout_Results.zip')]},bootstrap_draws=10000,bootstrap_seed=202812,ci='Pointwise percentile 95%; entire seed ensembles resampled in pairs; references fixed. No multiple-testing superiority claim.')
analysis=dict(validation=validation,configuration_checkpoints=rows,failures=failures,tail_diagnostics=tails,primary_fixed_choices=fixed,primary_speedups=comparisons,exploratory_grid_minima=exploratory,selection_failures=c['not_selected'])
(OUT/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n');(OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
lines=['# P3A Bank T4 held-out results','', '96/96 runs completed in 15.0 minutes; 8 new seeds ×512 mode-start independent replicas, K ∈ {128,512,2048}. No budget censoring. All 9 failed runs are CV2, with 11 failed replicas total. The raw results and all recorded failure information remain preserved.','', '## Main conclusion','', 'The frozen tested configurations do not establish a CV2 time-to-accuracy advantage. CV1 batch-size optimization yields stable configurations meeting stricter variance targets; CV2 shows rare instability and large ergodic variance errors. Lower gradient MSE alone does not ensure reliable sampling or faster attainment of the accuracy targets. This conclusion applies to this bounded Bank/T4 search, not every CV2 tuning choice or target.','', '## K2048 configuration summary','', '|Method|Batch|h|Failed runs /8|Failed replicas /4096|SW2 [95% CI]|Variance NRMSE [95% CI]|Warm seconds|Cold seconds|','|---|---:|---:|---:|---:|---|---|---:|---:|']
fmt=lambda v,ci:f'{v:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]'
for r in rows:
 if r['iteration']!=2048:continue
 sw=fmt(r['sw2'],r['sw2_ci95']) if r['valid_full_configuration'] else 'Ineligible'
 va=fmt(r['variance_nrmse'],r['variance_nrmse_ci95']) if r['valid_full_configuration'] else 'Ineligible'
 lines.append(f"|{r['method']}|{r['batch']}|{r['h']}|{r['failed_runs']}|{r['failed_replicas']}|{sw}|{va}|{r['warm_seconds']:.3f}|{r['cold_seconds']:.3f}|")
lines+=['|cv2|32|—|Selection failed|—|Not run|Not run|—|—|','', 'Any failed seed excludes the full configuration at every checkpoint from the formal frontier. Costs of failed configurations remain descriptive. No survivor-only or early-prefix accuracy averages are used. CV2 n32 was excluded because both shortlisted step sizes failed selection; this does not prove that all possible CV2 n32 step sizes fail.','', '## Selection-fixed primary targets','', 'Both reference-specific seed means must pass. Joint mode also requires SW2≤0.3. These are frozen heuristic accuracy targets, not theoretical error guarantees. A missing selection choice is distinct from a heldout failure to attain a chosen target.','', '|Mode|Variance target|Method|Frozen batch,h,K|Held-out status|Warm seconds|Cold seconds|','|---|---:|---|---|---|---:|---:|']
for q in fixed:
 z=q['choice'];desc=f"{z['batch']}, {z['h']}, {z['iteration']}" if z else 'None'
 lines.append(f"|{q['mode']}|{q['threshold']}|{q['method']}|{desc}|{q['status']}|{q.get('warm_seconds',float('nan')):.3f}|{q.get('cold_seconds',float('nan')):.3f}|")
lines+=['', 'No finite CV1/CV2 primary speedup can be reported at any of the four thresholds: CV2 fails its selected 0.5 target, and has no selection-fixed choice for 0.4,0.3,0.2. Do not interpret unavailable ratios as infinite speedups. Time entries for unsuccessful choices describe their measured cost and are not time-to-accuracy estimates.','', '## Diagnostic interpretation','', 'Only CV2 n4 passes the full stability rule among the five heldout CV2 configurations; its K2048 SW2≈0.335 and variance NRMSE≈3.842 miss the 0.3/0.5 joint target. Its variance NRMSE is markedly higher than the pilot/selection value, illustrating sensitivity to rare excursions even without formal divergence. Saved checkpoints provide endpoint clouds and accumulated functionals, not full step-by-step trajectories, so the analysis does not prove a precise failure mechanism.','', 'CV1 n32 (h=.00625) reaches variance target0.5; n256 (h=.025) reaches0.4 at K512 and0.3 at K2048. ULA(h=.1) reaches0.2 at K2048. These are selection-fixed choices with heldout verification; exact costs and uncertainty are in analysis.json.','', '## Limits and cost interpretation','', 'This is one h per batch selected from a bounded pilot/selection grid, not the global optimum over every n,h,T. Frontier optimization on heldout is descriptive and potentially optimistic; it is separated from the primary selection-fixed comparisons. Original pilot/selection failures remain part of the study.','', 'ULA h=.1 is an optimized baseline; the heldout is not a matched-h gradient-noise decomposition. The original pilot includes matched-h ULA controls. SGLD is a diagnostic pilot baseline, not a new heldout arm.','', 'All timing comparisons here use T4, R512, identical replica budgets, setup-inclusive wall-clock. Cold timing adds measured shape-specific compile time, with cache/order effects retained; it is not an independent fresh-process cold-start benchmark. Reference construction, functional evaluation, scoring and backup are outside the primary time axis and recorded separately in raw data. No absolute-time comparison to A100 or R128 selection.','', 'Intervals are pointwise paired-seed bootstrap, conditional on the frozen reference clouds. No multiple-comparison significance or universal dominance claim is made.','', '## Failure audit','', '|Method|Batch|h|Seed|Failed /512|First failure iteration|','|---|---:|---:|---:|---:|---:|']
for f in failures:lines.append(f"|{f['method']}|{f['batch']}|{f['h']}|{f['seed']}|{f['failed_replicas']}|{f['first_failure']}|")
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(validation,indent=2));print('Primary targets',[(q['mode'],q['threshold'],q['method'],q['status']) for q in fixed])
