"""Independent CPU amendment analysis. Preserves frozen choices and raw failures."""
from pathlib import Path
import sys,json,collections,hashlib
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sgld_cv.transport import AccuracyMetric
DATA=ROOT/'results/p3a_bank_jax_cpu_amendment1_validation_v1';ASSETS=ROOT/'results/p3a_bank_t4_heldout_v1/frozen_bank_assets';OUT=ROOT/'exports/p3a_jax_cpu_validation_review_v1';OUT.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
c=json.loads((DATA/'PROTOCOL.json').read_text());runs=json.loads((DATA/'summary.json').read_text());manifest=json.loads((DATA/'manifest.json').read_text())
assert c==json.loads((Path(__file__).parent/'PROTOCOL.json').read_text())
assert all(sha(ROOT/p)==h for p,h in c['input_hashes'].items())
assert manifest['status']=='complete' and len(runs)==c['runs']==296
assert set(c['validation_seeds']).isdisjoint(c['pilot_seeds'])
assert all(r['replicas']==512 and r['completed_iterations']==r['max_iterations'] and not r['budget_censored'] for r in runs)
assert not any(r['timing_interrupted'] for r in runs)
groups=collections.defaultdict(list)
for r in runs:groups[r['method'],r['batch'],r['h']].append(r)
assert len(groups)==37 and all(sorted(r['seed'] for r in rs)==c['validation_seeds'] for rs in groups.values())
a=np.load(ASSETS/'target.npz');s=np.load(ASSETS/'scoring.npz');refs=[s['first'],s['second']];anchor,W=a['anchor'],a['W'];d=len(anchor);refvars=[((r-anchor)@W).var(0) for r in refs];metric=AccuracyMetric('standardized_sliced_w2',anchor,W,s['projections'])
boot=np.random.default_rng(c['bootstrap_seed']).integers(0,8,(c['bootstrap_draws'],8));ci=lambda x:np.quantile(x,[.025,.975]).tolist()
rows=[];lookup={};failures=[];tails=[];prefixes=[];prefix_lookup={};nrmse_checked=sw_checked=0
with threadpool_limits(limits=1):
 for (m,n,h),rs in sorted(groups.items()):
  rs=sorted(rs,key=lambda r:r['seed']);valid=all(r['failed_replicas']==0 for r in rs);horizon=rs[0]['max_iterations'];ks=[k for k in c['checkpoints'] if k<=horizon]
  for r in rs:
   if r['failed_replicas']:failures.append(dict(method=m,batch=n,h=h,seed=r['seed'],failed_replicas=r['failed_replicas'],first_failure=min(x for x in r['failed_at'] if x),failed_at=r['failed_at']))
  for k in ks:
   ps=[next(p for p in r['checkpoints'] if p['iteration']==k) for r in rs];warm=np.array([p['warm_total_seconds'] for p in ps]);cold=np.array([p['cold_total_seconds'] for p in ps]);q=dict(method=m,batch=n,h=h,iteration=k,hK=h*k,horizon=horizon,valid_full_configuration=valid,failed_runs=sum(r['failed_replicas']>0 for r in rs),failed_replicas=sum(r['failed_replicas'] for r in rs),total_replicas=4096,seed_warm_seconds=warm.tolist(),seed_cold_seconds=cold.tolist(),warm_seconds=float(warm.mean()),cold_seconds=float(cold.mean()),warm_ci95=ci(warm[boot].mean(1)),cold_ci95=ci(cold[boot].mean(1)))
   if valid:
    assert all(p['status']=='ok' and p['failed_replicas']==0 for p in ps)
    v=np.array([[[z[t] for t in ['sw2','variance_nrmse']] for z in p['references']] for p in ps]);mean=v.mean(0).max(0);bs=v[boot].mean(1).max(1);q.update(sw2=float(mean[0]),variance_nrmse=float(mean[1]),sw2_ci95=ci(bs[:,0]),variance_nrmse_ci95=ci(bs[:,1]),seed_accuracy=v.tolist(),reference_specific_seed_means=v.mean(0).tolist())
    for r,p in zip(rs,ps):
     z=np.load(DATA/f"checkpoint_{m}_n{n}_h{h}_seed{r['seed']}_K{k}.npz");assert np.all(z['failed_at']==0);avg=z['functional_sums']/k;var=avg[:,d:]-avg[:,:d]**2
     for j,vref in enumerate(refvars):
      recomputed=float(np.sqrt(np.mean(((var-vref)/vref)**2,axis=0)).mean());np.testing.assert_allclose(recomputed,p['references'][j]['variance_nrmse'],rtol=1e-11,atol=1e-12)
     nrmse_checked+=1
     if k==horizon and r['seed']==c['validation_seeds'][0]:
      for j,reference in enumerate(refs):np.testing.assert_allclose(metric.distance(z['endpoints'],reference),p['references'][j]['sw2'],rtol=1e-11,atol=1e-12)
      sw_checked+=1
     if k==horizon:
      err=(var-refvars[0])/refvars[0];sq=(err*err).sum(1);lane=np.sqrt(np.mean(err*err,axis=1));top=max(1,int(np.ceil(.01*len(sq))));tails.append(dict(method=m,batch=n,h=h,seed=r['seed'],iteration=k,median_lane_variance_error=float(np.median(lane)),p99_lane_error=float(np.quantile(lane,.99)),max_lane_error=float(lane.max()),top1percent_squared_error_fraction=float(np.sort(sq)[-top:].sum()/sq.sum())))
   else:q.update(sw2=None,variance_nrmse=None,reason='Any failed validation seed excludes full h configuration at every checkpoint; no survivor scoring or early-prefix formal frontier')
   if not valid and all(p['status']=='ok' and p['failed_replicas']==0 and p.get('references') for p in ps):
    vv=np.array([[[z[t] for t in ['sw2','variance_nrmse']] for z in p['references']] for p in ps]);mm=vv.mean(0).max(0);bb=vv[boot].mean(1).max(1);diagnostic=dict(method=m,batch=n,h=h,iteration=k,all8_seeds_and4096_lanes_valid_at_checkpoint=True,formally_excluded_due_to_later_failure=True,sw2=float(mm[0]),variance_nrmse=float(mm[1]),sw2_ci95=ci(bb[:,0]),variance_nrmse_ci95=ci(bb[:,1]),scope='Diagnostic all-lane prefix, never substitutes frozen full-horizon gate or enters formal frontier');prefixes.append(diagnostic);prefix_lookup[m,n,h,k]=diagnostic

   rows.append(q);lookup[m,n,h,k]=q
fixed=[]
for spec in c['primary_fixed_choices']:
 q={k:spec[k] for k in ['mode','threshold','method','cost','choice']};z=spec['choice']
 if z is None:q.update(status='no_pilot_fixed_choice',reached=False)
 else:
  r=lookup[z['method'],z['batch'],z['h'],z['iteration']];typ=q['cost'];q.update(time_seconds=r[typ+'_seconds'],time_ci95=r[typ+'_ci95'],sw2=r['sw2'],variance_nrmse=r['variance_nrmse'])
  if not r['valid_full_configuration']:
   q.update(status='failed_configuration',reached=False,all_lane_prefix_diagnostic=prefix_lookup.get((z['method'],z['batch'],z['h'],z['iteration'])))
  else:
   v=np.array(r['seed_accuracy']);bs=v[boot].mean(1).max(1);attain=bs[:,1]<=q['threshold'];reach=r['variance_nrmse']<=q['threshold']
   if q['mode']=='joint':attain &= bs[:,0]<=.3;reach=reach and r['sw2']<=.3
   q.update(status='reached' if reach else 'not_reached',reached=bool(reach),bootstrap_attainment_fraction=float(attain.mean()),sw2_ci95=r['sw2_ci95'],variance_nrmse_ci95=r['variance_nrmse_ci95'])
 fixed.append(q)
speedups=[]
for mode in ['variance_only','joint']:
 for threshold in [.5,.4,.3,.2]:
  for cost in ['warm','cold']:
   qq=[next(x for x in fixed if (x['mode'],x['threshold'],x['cost'],x['method'])==(mode,threshold,cost,m)) for m in ['cv1','cv2']];q=dict(mode=mode,threshold=threshold,cost=cost,cv1_status=qq[0]['status'],cv2_status=qq[1]['status'],speedup=None,ci95=None)
   if all(x['reached'] for x in qq):
    vv=[np.array(lookup[x['choice']['method'],x['choice']['batch'],x['choice']['h'],x['choice']['iteration']]['seed_'+cost+'_seconds']) for x in qq];q.update(speedup=float(vv[0].mean()/vv[1].mean()),ci95=ci(vv[0][boot].mean(1)/vv[1][boot].mean(1)))
   speedups.append(q)
# Complete-grid descriptive optima, explicitly separate from frozen primary choices.
exploratory=[]
for mode in ['variance_only','joint']:
 for threshold in [.5,.4,.3,.2]:
  for m in ['cv1','cv2','ula']:
   eligible=[r for r in rows if r['method']==m and r['valid_full_configuration'] and r['variance_nrmse']<=threshold and (mode=='variance_only' or r['sw2']<=.3)]
   for cost in ['warm','cold']:
    r=min(eligible,key=lambda r:(r[cost+'_seconds'],r['batch'],r['h'],r['iteration'])) if eligible else None
    exploratory.append(dict(mode=mode,threshold=threshold,method=m,cost=cost,status='observed_reached' if r else 'not_reached',choice=None if r is None else {k:r[k] for k in ['method','batch','h','iteration','sw2','variance_nrmse',cost+'_seconds']}))
# Paired matched-batch/step comparisons include all common configurations, not selected favorable cases.
matched=[]
for (m,n,h),rs in sorted(groups.items()):
 if m!='cv2' or ('cv1',n,h) not in groups:continue
 for k in c['checkpoints']:
  x,y=lookup['cv1',n,h,k],lookup['cv2',n,h,k];q=dict(batch=n,h=h,iteration=k,status='eligible' if x['valid_full_configuration'] and y['valid_full_configuration'] else 'ineligible_failure')
  if q['status']=='eligible':
   v=[np.array(r['seed_accuracy']) for r in [x,y]];bs=[r[boot].mean(1).max(1) for r in v];q.update(sw2_cv2_minus_cv1=y['sw2']-x['sw2'],sw2_difference_ci95=ci(bs[1][:,0]-bs[0][:,0]),variance_cv2_minus_cv1=y['variance_nrmse']-x['variance_nrmse'],variance_difference_ci95=ci(bs[1][:,1]-bs[0][:,1]))
  matched.append(q)
controls=[]
for r in rows:
 if r['method'] not in ['cv1','cv2'] or ('ula',32,r['h'],r['iteration']) not in lookup:continue
 u=lookup['ula',32,r['h'],r['iteration']];q=dict(method=r['method'],batch=r['batch'],h=r['h'],iteration=r['iteration'],status='eligible' if r['valid_full_configuration'] and u['valid_full_configuration'] else 'ineligible_failure')
 if q['status']=='eligible':
  vv=[np.array(z['seed_accuracy']) for z in [u,r]];bs=[z[boot].mean(1).max(1) for z in vv];q.update(sw2_minus_ula=r['sw2']-u['sw2'],variance_minus_ula=r['variance_nrmse']-u['variance_nrmse'],sw2_difference_ci95=ci(bs[1][:,0]-bs[0][:,0]),variance_difference_ci95=ci(bs[1][:,1]-bs[0][:,1]))
 controls.append(q)
valid_counts=dict(collections.Counter(m for (m,n,h) in groups if lookup[m,n,h,min(r['max_iterations'] for r in groups[m,n,h])]['valid_full_configuration']))
validation=dict(runs=296,failed_runs=len(failures),failed_replicas=sum(r['failed_replicas'] for r in failures),budget_censored=0,seeds_disjoint=True,nrmse_checkpoints_reconstructed=nrmse_checked,sw2_endpoints_reconstructed=sw_checked,stable_configurations=valid_counts,pointwise_bootstrap_draws=c['bootstrap_draws'],bootstrap_seed=c['bootstrap_seed'],ci_scope='Paired full-seed percentile95%; fixed references, no multiplicity correction. Bootstrap attainment fraction is a diagnostic, not probability of future success.',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [DATA/'summary.json',DATA/'PROTOCOL.json',DATA/'P3A_JAX_CPU_Validation_Amendment1.zip']})
analysis=dict(validation=validation,valid_prefix_diagnostics=prefixes,configuration_checkpoints=rows,primary_fixed_choices=fixed,primary_speedups=speedups,exploratory_grid_minima=exploratory,matched_cv_comparisons=matched,matched_ula_gaps=controls,failures=failures,tail_diagnostics=tails)
(OUT/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n');(OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
fmt=lambda v,interval:f'{v:.4f} [{interval[0]:.4f}, {interval[1]:.4f}]'
lines=['# P3A amendment1 — independent JAX CPU validation','', '296/296 runs completed in4h04m.34 CV1/CV2 pilot candidates ×8 fresh seeds,512 replicas, K≤8192;24 ULA control runs through2048. No budget censoring or interrupted timings.20 runs contain28 failed replicas; all are CV2. Original T4/A100 results and all raw failures are preserved.','', '## Conclusion','', 'The expanded small-step search produces additional stable CV2 candidates, but independent CPU validation still does not establish a CV2 time-to-accuracy advantage on this Bank target. All18 CV1 candidates pass the full-horizon stability gate;8 of16 CV2 candidates pass. Stability alone is insufficient: several stable CV2 configurations have large ergodic variance errors.','', 'All four pilot-fixed joint CV2 target comparisons are unavailable:0.5 selects a configuration with validation failures;0.4 and0.3 select stable configurations that miss their target;0.2 has no pilot-fixed choice. No finite primary CV1/CV2 speedup should be reported. This is a result for the bounded amended search, not every possible tuning choice or posterior.','', '## Primary pilot-fixed joint targets — CPU wall-clock seconds','', 'Both reference-specific eight-seed means must satisfy SW2≤0.3 and the variance target. A time for an unsuccessful choice is its measured cost, not time-to-accuracy.','', '|Variance target|Cost|Method|Fixed batch,h,K|Status|Seconds [95%CI]|Variance NRMSE [95%CI]|Bootstrap attainment fraction|','|---|---|---|---|---|---|---|---:|']
for q in fixed:
 if q['mode']!='joint':continue
 z=q['choice'];desc=f"{z['batch']}, {z['h']}, {z['iteration']}" if z else 'None';ti=fmt(q['time_seconds'],q['time_ci95']) if z else '—';va=fmt(q['variance_nrmse'],q['variance_nrmse_ci95']) if q.get('variance_nrmse_ci95') else '—';fraction=f"{q['bootstrap_attainment_fraction']:.3f}" if 'bootstrap_attainment_fraction' in q else '—';lines.append(f"|{q['threshold']}|{q['cost']}|{q['method']}|{desc}|{q['status']}|{ti}|{va}|{fraction}|")
lines+=['', 'For the failed CV2 primary0.5 configuration (n32,h=.0125,K512), all4096 lanes are still valid at K512; formal exclusion reflects later failures through8192. The all-lane prefix diagnostic separately also misses the0.5 variance target. It is reported in analysis.json and is not used to relax the frozen gate or replace the primary comparison.']
lines+=['', '## Descriptive minima within the fixed validation grid','', 'These minima were selected after observing validation and may be optimistic. They do not replace the frozen primary choices or constitute a proof of globally optimal cost.','', '|Joint variance target|Method|Observed warm-time minimum|Batch,h,K|','|---|---|---:|---|']
for q in exploratory:
 if q['mode']!='joint' or q['cost']!='warm':continue
 z=q['choice'];lines.append(f"|{q['threshold']}|{q['method']}|{z['warm_seconds']:.3f}|{z['batch']}, {z['h']}, {z['iteration']}|" if z else f"|{q['threshold']}|{q['method']}|Not reached|—|")
lines+=['', '## Full-horizon candidate outcomes','', '|Method|Batch|h|Horizon|Failed runs/8|Failed replicas/4096|SW2 [95%CI]|Variance NRMSE [95%CI]|Warm seconds|','|---|---:|---:|---:|---:|---:|---|---|---:|']
for r in rows:
 if r['iteration']!=r['horizon']:continue
 sw=fmt(r['sw2'],r['sw2_ci95']) if r['valid_full_configuration'] else 'Ineligible';va=fmt(r['variance_nrmse'],r['variance_nrmse_ci95']) if r['valid_full_configuration'] else 'Ineligible';lines.append(f"|{r['method']}|{r['batch']}|{r['h']}|{r['horizon']}|{r['failed_runs']}|{r['failed_replicas']}|{sw}|{va}|{r['warm_seconds']:.3f}|")
lines+=['', '## Interpretation and scope','', 'The8 surviving CV2 configurations cover batches4,16,32,64,256; none of the three batch1 candidates passes the full-horizon gate. Larger batches can yield accurate stable CV2 configurations, but the amended CPU accuracy–cost comparison does not show an advantage over the tested CV1 candidates. Results on GPU hardware may have different time rankings.','', 'Any failed seed invalidates the complete configuration at every checkpoint, including otherwise good earlier prefixes. Successful replicas are never pooled after deleting failed lanes. The full failure census remains in this report.','', 'Variance NRMSE uses per-replica ergodic variance estimates, relative posterior-variance normalization, coordinate RMSE then coordinate averaging, finally averaging across8 seeds separately for each reference. SW2 uses independent mode-start replica endpoints. The reported summary is the worse of the two reference-specific seed means, not a pooled chain-draw distribution.','', 'The smaller grid and longer horizons were explicitly frozen after inspection of the original heldout. This is an exploratory protocol amendment with genuinely independent new validation seeds. It does not retrospectively change the original registered results.','', 'Paired matched(n,h,K) CV1/CV2 differences and matched-h ULA gaps are in analysis.json, including ineligible cases. ULA controls cover h=.00625,.0125,.1 through2048 only. No full matched-h decomposition is available at8192. SGLD remains a prior pilot diagnostic.','', 'Timing includes the original stored anchor-time attribution, locally measured setup and synchronized compiled sampler (including RNG). Cold adds measured shape-specific compilation with cache/order effects; it is not a fresh-process cold benchmark. Functional/scoring/checkpoint/archive costs are separate. No absolute CPU-versus-T4/A100/V100 efficiency claim is made. CPU-selected cost-optimal settings need GPU-specific verification.','', 'Pointwise paired-seed bootstrap intervals use10000 draws seed203799 and frozen reference clouds; they are not multiplicity-adjusted. The reference finite-sample floor and any unresolved tail events limit claims from small endpoint SW2 differences. Saved states and accumulated functionals permit tail diagnostics but do not prove the precise failure mechanism.','', f'Validation reconstructed {nrmse_checked} checkpoint NRMSE values and {sw_checked} final endpoint SW2 pairs from saved arrays, all agreeing with recorded metrics.','', '## Failure census','', '|Method|Batch|h|Seed|Failed/512|First failure iteration|','|---|---:|---:|---:|---:|---:|']
for f in failures:lines.append(f"|{f['method']}|{f['batch']}|{f['h']}|{f['seed']}|{f['failed_replicas']}|{f['first_failure']}|")
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(validation,indent=2))
print('PRIMARY_WARM_JOINT',[(q['threshold'],q['method'],q['status'],q.get('time_seconds'),q.get('variance_nrmse')) for q in fixed if q['mode']=='joint' and q['cost']=='warm'])
print('EXPLORATORY_WARM_JOINT',[(q['threshold'],q['method'],q['choice']) for q in exploratory if q['mode']=='joint' and q['cost']=='warm'])
