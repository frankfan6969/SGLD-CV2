"""Frozen three-method ULA frontier extension; original results are read-only."""
from pathlib import Path
import sys,os,time,json,platform,subprocess,traceback,fcntl
from time import perf_counter
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R/'gpu_replication'),str(R/'p2_screening')]
import numpy as np,jax,jax.numpy as jnp
from scipy.special import expit
from threadpoolctl import threadpool_limits
from batched_sampler import prepare,initial_state,ready,gradient_parts,advance
from kernels import sample_kernel
from extension_kernels import gradient_kernel,ula_kernel,ula_step,sum_functionals
from common import *
c=load(H/'PROTOCOL.json');started=perf_counter();guard=(D/'.pipeline.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert jax.default_backend()=='cpu' and jax.config.x64_enabled
for rel,digest in c['source_hashes'].items():assert sha(R/rel)==digest,rel
save(D/'environment.json',dict(python=sys.version,jax=jax.__version__,numpy=np.__version__,platform=platform.platform(),devices=[str(d) for d in jax.devices()],x64=jax.config.x64_enabled,pid=os.getpid(),threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','XLA_FLAGS']}))
save(D/'CODE_LOCK.json',{str(p.relative_to(R)):sha(p) for p in list(H.glob('*.py'))+[H/'PROTOCOL.json']})
statuses=[]
def progress(stage,**kwargs):save(D/'STATUS.json',dict(status='running',stage=stage,elapsed_seconds=perf_counter()-started,targets=statuses,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid(),**kwargs))
def run_target(t):
 name=t['id'];F=D/name;O=E/name;F.mkdir(exist_ok=True);O.mkdir(exist_ok=True);target_start=perf_counter();A=R/t['assets'];a=np.load(A/'target.npz');s=np.load(A/'scoring.npz');X,y,anchor,W,Hess=[a[k] for k in ['X','y','anchor','W','H']];refs=[s['first'],s['second']];proj=s['projections'];meta=load(A/'metadata.json')
 for n in ['reference_pair.json','reference_0.json','reference_1.json']:assert load(A/n)['reliable'],(name,n)
 old=load(R/t['parent_candidates']);parent=load(R/t['parent_protocol']);LP=float(parent['LP']);meta['LP']=LP
 save(F/'TARGET.json',dict(target=name,N=len(X),d=len(anchor),LP=LP,assets=t['assets'],reference_gates=True))
 params,_=prepare(X,y,anchor,1.,'logistic','ula','full',W);rng=np.random.default_rng(410010);theta=np.array(refs[0][:3]);normal=rng.normal(size=theta.shape);want=(expit(theta@X.T)-y)@X+theta;blocks=sorted(set(min(len(X),b) for b in [1024,4096,16384] if b*512*8<=64*1024**2));bench=[]
 for b in blocks:
  fn=gradient_kernel(b);pp=jnp.asarray(theta);got=np.asarray(ready(fn(params,pp)));np.testing.assert_allclose(got,want,rtol=1e-9,atol=1e-8)
  st=list(initial_state(params,3));st[0]=pp;_,after=ready(ula_step(b)(params,tuple(st),jnp.asarray(normal),.003125));expected=theta-.003125*(want@np.asarray(params['P']).T)+np.sqrt(.00625)*(normal@np.asarray(params['L']).T);np.testing.assert_allclose(after,expected,rtol=1e-9,atol=1e-9)
  probe=jnp.asarray(np.resize(theta,(512,len(anchor))));ready(fn(params,probe));seconds=[]
  for _ in range(3):
   tt=perf_counter();ready(fn(params,probe));seconds.append(perf_counter()-tt)
  bench.append(dict(block=b,median_gradient_seconds=float(np.median(seconds)),times=seconds,max_gradient_error=float(abs(got-want).max())))
 block=min(bench,key=lambda q:q['median_gradient_seconds'])['block'];save(F/'PREFLIGHT.json',dict(status='passed',gradient_update_checks=True,benchmark=bench,selected_block=block,selection='Fastest median gradient call under64MiB replica-logit workspace; no accuracy outcomes used',scope='Exact full-data gradient; different blocked summation can differ to roundoff'))
 print('PREFLIGHT',name,'block',block,'gradient seconds',min(q['median_gradient_seconds'] for q in bench),flush=True)
 compiled={};records=[]
 def budget():
  if perf_counter()-started>c['budgets']['total_research_seconds'] or perf_counter()-target_start>c['budgets']['target_research_seconds']:raise TimeoutError('Frozen research budget; report incomplete, no favorable imputation')
 def job(phase,m,n,h,seed,Rcount,horizon,checkpoints):
  budget();rid=f'{phase}_{m}_n{n}_h{h:.12g}_seed{seed}';dest=F/(rid+'.json')
  if dest.exists():r=load(dest);records.append(r);return r
  key=(m,n,Rcount)
  if key not in compiled:
   p,_=prepare(X,y,anchor,1.,'logistic',m,'full',W);st=ready(initial_state(p,Rcount));fn=ula_kernel(block) if m=='ula' else sample_kernel(m,'full',n,.99);tt=perf_counter();exe=fn.lower(p,st,410010,h).compile();compile_s=perf_counter()-tt;tt=perf_counter();ready(exe(p,st,410010,h));warmup_s=perf_counter()-tt;compiled[key]=(exe,compile_s);save(F/f'compile_{m}_n{n}_R{Rcount}.json',dict(compile_seconds=compile_s,warmup_seconds=warmup_s,block=block if m=='ula' else None))
  exe,compile_s=compiled[key];tt=perf_counter();p,setup=prepare(X,y,anchor,1.,'logistic',m,'full',W);state=ready(initial_state(p,Rcount));setup_s=perf_counter()-tt;sums=jnp.zeros((Rcount,2*len(anchor)));sampling=functional=evaluation=0.;points=[];last=0;censored=False
  for k in range(128,horizon+1,128):
   budget()
   if sampling>c['budgets']['per_run_sampler_seconds']:censored=True;break
   tt=perf_counter();state,xx=ready(exe(p,state,seed,h));sampling+=perf_counter()-tt;last=k;tt=perf_counter();sums=ready(sums+sum_functionals(xx,p['anchor'],p['transform']));functional+=perf_counter()-tt;failed=np.asarray(state[3])
   if k in checkpoints:
    tt=perf_counter();q=score(np.asarray(xx[-1]),np.asarray(sums),k,anchor,W,proj,refs) if not np.any(failed) else dict(status='failed_ensemble',references=None);evaluation+=perf_counter()-tt;q.update(iteration=k,failed_replicas=int(np.sum(failed>0)),warm_total_seconds=meta['anchor_seconds']+setup_s+sampling,cold_total_seconds=meta['anchor_seconds']+setup_s+sampling+compile_s);points.append(q);arrays(F/f'{rid}_K{k}.npz',endpoints=np.asarray(xx[-1]),functional_sums=np.asarray(sums),failed_at=failed)
   progress('sampling',target=name,phase=phase,method=m,batch=n,h=h,seed=seed,iteration=k,run_horizon=horizon,target_runs_completed=len(records))
  r=dict(id=rid,target=name,phase=phase,method=m,batch=n,h=h,seed=seed,replicas=Rcount,completed_iterations=last,failed_replicas=int(np.sum(np.asarray(state[3])>0)),failed_at=np.asarray(state[3]).tolist(),budget_censored=censored,checkpoints=points,compile_seconds=compile_s,setup_seconds=setup_s,setup=setup,sampling_seconds=sampling,functional_seconds=functional,evaluation_seconds=evaluation,anchor_seconds_attributed=meta['anchor_seconds']);save(dest,r);records.append(r);save(F/'summary.json',records);print('RUN',name,len(records),rid,'failed',r['failed_replicas'],'K',last,'sampler_s',round(sampling,3),flush=True);return r
 pilot=[];jobs=[(h/LP,seed) for h in c['ula']['step_coefficients'] for seed in t['pilot_seeds']];np.random.default_rng(419000).shuffle(jobs)
 for h,seed in jobs:pilot.append(job('pilot','ula',len(X),h,seed,c['ula']['pilot_replicas'],c['ula']['pilot_horizon'],c['ula']['pilot_checkpoints']))
 save(F/'pilot_summary.json',pilot);points=[]
 for h in sorted(set(r['h'] for r in pilot)):
  rr=[r for r in pilot if r['h']==h]
  if len(rr)!=2 or any(r['failed_replicas'] or r['budget_censored'] or r['completed_iterations']!=2048 for r in rr):continue
  for k in c['ula']['pilot_checkpoints']:
   pp=[next(p for p in r['checkpoints'] if p['iteration']==k) for r in rr];v=np.mean([[[z[m] for m in ['sw2','variance_nrmse']] for z in p['references']] for p in pp],axis=0).max(0);points.append(dict(h=h,K=k,warm=np.mean([p['warm_total_seconds'] for p in pp]),cold=np.mean([p['cold_total_seconds'] for p in pp]),sw2=v[0],variance=v[1]))
 if not points:raise RuntimeError('No complete stable ULA pilot candidate; preserve records and stop target without retuning')
 rank=ranks([[q[k] for k in ['warm','cold','sw2','variance']] for q in points]);hs=sorted(set(q['h'] for q in points));scores=[]
 for h in hs:
  rr=[r for q,r in zip(points,rank) if q['h']==h];scores.append((h,(-sum(r==0 for r in rr),min(rr),-sum(1/(1+r) for r in rr),h)))
 selected=[dict(method='ula',batch=len(X),h=h) for h,_ in sorted(scores,key=lambda q:q[1])[:c['ula']['max_frozen_candidates']]];selected+=[{k:r[k] for k in ['method','batch','h']} for r in old['candidates']];freeze=dict(candidates=selected,source_cv_candidates=t['parent_candidates'],validation_seeds=t['validation_seeds'],max_iterations=c['validation']['horizon'],ula_pilot_selection_scores=scores,scope='Freeze before fresh validation, no postvalidation refinement');save(F/'FROZEN_CANDIDATES.json',freeze)
 print('CANDIDATES_FROZEN',name,len(selected),'new validation seeds',t['validation_seeds'],flush=True);jobs=[(q,seed) for q in selected for seed in t['validation_seeds']];np.random.default_rng(419001).shuffle(jobs);validation=[]
 for q,seed in jobs:validation.append(job('validation',q['method'],q['batch'],q['h'],seed,c['validation']['replicas'],c['validation']['horizon'],c['validation']['checkpoints']))
 save(F/'validation_summary.json',validation);progress('analysis',target=name);subprocess.run([sys.executable,str(H/'analyze.py'),name],check=True,timeout=600);return dict(target=name,status='complete',validation_runs=len(validation),pilot_runs=len(pilot),report=str(O/'REPORT.md'))
try:
 with threadpool_limits(limits=1):
  for t in c['targets']:
   progress('target_start',target=t['id'])
   try:status=run_target(t)
   except Exception:
    status=dict(target=t['id'],status='failed_or_budget_censored',error=traceback.format_exc());save(D/t['id']/'INCOMPLETE.json',status);print(status['error'],flush=True)
   statuses.append(status);save(D/t['id']/'STATUS.json',status)
  save(D/'STATUS.json',dict(status='complete' if all(r['status']=='complete' for r in statuses) else 'partial',targets=statuses,elapsed_seconds=perf_counter()-started,pid=os.getpid()));subprocess.run([sys.executable,str(H/'report.py')],check=True);print('P5_ULA_EXTENSION_FINISHED',flush=True)
except Exception:
 save(D/'STATUS.json',dict(status='failed',targets=statuses,error=traceback.format_exc()));raise
