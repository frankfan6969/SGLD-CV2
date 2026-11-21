"""Frozen fixed(n,h) checkpoint sensitivity; no tuning or target changes."""
from pathlib import Path
import sys,json,time,os,hashlib,fcntl
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R/'src'),str(R/'gpu_replication'),str(R/'p2_screening')]
import run as base
import jax
import jax.numpy as jnp
from kernels import sample_kernel
H=Path(__file__).parent;O=R/'exports/p3a_log1p_checkpoint_v1';A=R/'exports/p3a_feature_sensitivity_v1/log1p_previous';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();save=base.save
assert jax.default_backend()=='cpu' and jax.config.x64_enabled;assert json.loads((A/'reference_pair.json').read_text())['reliable']
files=[A/x for x in ['target.npz','scoring.npz','metadata.json','reference_pair.json']]+[R/'p2_screening/kernels.py',R/'gpu_replication/batched_sampler.py',R/'p2_screening/run.py']
c=dict(name='Log1p Bank bounded fixed-configuration checkpoint comparison v1',backend='JAX CPU float64',methods=['cv1','cv2'],batch=16,h=.025,geometry='full',replicas=512,seeds=list(range(204700,204708)),initialization_seed=204699,schedule_seed=204698,checkpoints=[128,256,384,512],max_iterations=512,sw2_threshold=.3,variance_nrmse_threshold=.5,bootstrap_seed=204799,bootstrap_draws=10000,initialization='Every replica independently starts at its own target mode. Original counter RNG paired by seed,replica,iteration,purpose across methods.',metric='Endpoint standardized slicedW2 and relative ergodic variance NRMSE; mean over8 seeds separately for each fixed reference, worse of two reference means. Unchanged definitions.',failure_rule='Any failed replica in any seed through512 invalidates the method/config at every checkpoint. All failures retained; never score survivors.',time='Local stored anchor attribution + actual per-run prepare/state setup + synchronized compiled sampling including RNG. Cold adds compile per shape. Functionals/scoring/I/O excluded equally, separately recorded. Cache/order effects mean cold is not fresh-process cold.',first_attainment='Earliest tested checkpoint with both aggregate metrics passing; no interpolation and no global tuning claim. At128 reached means true first attainment is left-censored <=128; none through512 is Not Reached.',uncertainty='Paired whole-seed bootstrap preserves methods,checkpoints,references. Pointwise percentile95% CIs for metrics/time and paired differences; no multiplicity correction. Recompute earliest crossing per draw and retain Not Reached mass. Speedup CI on jointly attained draws is conditional, explicitly report fraction; no finite ratio when point comparison unavailable.',scope='Exploratory independent follow-up to observed feature sensitivity; thresholds fixed, no new reference or parameters. Four checkpoints only. Stop after these16 runs; no P3B.',budget_seconds=900,input_hashes={str(p.relative_to(R)):sha(p) for p in files})
if (H/'PROTOCOL.json').exists():raise RuntimeError('Protocol already exists; no silent overwrite/rerun')
# Audit prior executed summaries, not future protocol text.
used=set()
for folder in [R/'exports/p3a_feature_sensitivity_v1',R/'results']:
 for f in folder.rglob('*summary.json'):
  try:
   data=json.loads(f.read_text())
   if isinstance(data,list):used.update(int(x['seed']) for x in data if isinstance(x,dict) and 'seed' in x)
  except (ValueError,TypeError):pass
assert set(c['seeds']).isdisjoint(used)
save(H/'PROTOCOL.json',c);save(O/'PROTOCOL.json',c);save(O/'seed_audit.json',dict(passed=True,new_seeds=c['seeds'],prior_executed_seeds=sorted(used)));guard=(O/'.run.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
def arrays(path,**kw):
 tmp=path.with_name(path.name+'.tmp')
 with tmp.open('wb') as f:np.savez_compressed(f,**kw)
 os.replace(tmp,path)
a=np.load(A/'target.npz');X,y,anchor,W=[a[k] for k in ['X','y','anchor','W']];s=np.load(A/'scoring.npz');refs=[s['first'],s['second']];proj=s['projections'];meta=json.loads((A/'metadata.json').read_text());start=perf_counter();records=[];compiled={}
seeds=np.random.default_rng(c['schedule_seed']).permutation(c['seeds']);jobs=[]
for i,seed in enumerate(seeds):
 for m in (['cv1','cv2'] if i%2==0 else ['cv2','cv1']):jobs.append(dict(method=m,seed=int(seed)))
save(O/'schedule.json',jobs)
with threadpool_limits(limits=1):
 for m in c['methods']:
  p,_=base.prepare(X,y,anchor,1.,'logistic',m,'full',W);state=base.ready(base.initial_state(p,c['replicas']));fn=sample_kernel(m,'full',c['batch'],.99);t=perf_counter();exe=fn.lower(p,state,c['initialization_seed'],c['h']).compile();compile_s=perf_counter()-t;t=perf_counter();base.ready(exe(p,state,c['initialization_seed'],c['h']));warmup_s=perf_counter()-t;compiled[m]=(exe,compile_s);save(O/f'compile_{m}.json',dict(compile_seconds=compile_s,warmup128_seconds=warmup_s))
 for i,job in enumerate(jobs):
  assert perf_counter()-start<c['budget_seconds'];m,seed=job['method'],job['seed'];exe,compile_s=compiled[m];t=perf_counter();p,setup=base.prepare(X,y,anchor,1.,'logistic',m,'full',W);state=base.ready(base.initial_state(p,c['replicas']));setup_s=perf_counter()-t;sums=jnp.zeros((c['replicas'],2*len(anchor)));sampling=functionals=scoring=0.;points=[]
  for k in c['checkpoints']:
   t=perf_counter();state,xx=base.ready(exe(p,state,seed,c['h']));sampling+=perf_counter()-t;t=perf_counter();sums=base.ready(sums+base.sum_functionals(xx,p['anchor'],p['transform']));functionals+=perf_counter()-t;failed=np.asarray(state[3]);t=perf_counter();q=base.score(np.asarray(xx[-1]),np.asarray(sums),k,refs,anchor,W,proj);scoring+=perf_counter()-t
   if np.any(failed):q=dict(status='failed_ensemble',references=None)
   q.update(iteration=k,failed_replicas=int(np.sum(failed>0)),warm_total_seconds=meta['anchor_seconds']+setup_s+sampling,cold_total_seconds=meta['anchor_seconds']+setup_s+sampling+compile_s);points.append(q);arrays(O/f'checkpoint_{m}_seed{seed}_K{k}.npz',endpoints=np.asarray(xx[-1]),functional_sums=np.asarray(sums),failed_at=failed)
  r=dict(**job,replicas=c['replicas'],batch=c['batch'],h=c['h'],completed_iterations=512,failed_replicas=int(np.sum(failed>0)),failed_at=failed.tolist(),setup=setup,setup_seconds=setup_s,compile_seconds=compile_s,sampling_seconds=sampling,functional_seconds=functionals,scoring_seconds=scoring,anchor_seconds_attributed=meta['anchor_seconds'],checkpoints=points);save(O/f'result_{m}_seed{seed}.json',r);records.append(r);save(O/'summary.json',records);save(O/'manifest.json',dict(status='running' if i+1<len(jobs) else 'complete',runs_completed=len(records),total_runs=len(jobs),elapsed_seconds=perf_counter()-start,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())));print('CHECKPOINT_TEST',len(records),'/',len(jobs),m,seed,'failed',r['failed_replicas'],flush=True)
