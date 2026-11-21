from pathlib import Path
import sys,json,time,os,traceback,fcntl,subprocess
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R/'src'),str(R/'gpu_replication'),str(R/'p2_screening')]
import run as base
import jax
import jax.numpy as jnp
from kernels import hmc_kernel,sample_kernel
from sgld_cv.diagnostics import chain_diagnostics
from sgld_cv.references import validate_reference_pair
H=Path(__file__).parent;O=R/'exports/p3a_feature_sensitivity_v1';A=R/'results/p3a_bank_t4_heldout_v1/frozen_bank_assets';c=json.loads((H/'STAGE_B_PROTOCOL.json').read_text());save=base.save
assert jax.default_backend()=='cpu' and jax.config.x64_enabled;assert all(base.digest(R/p)==v for p,v in c['input_hashes'].items());guard=(O/'.stage_b.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);start=perf_counter();compiled={};results=[]
def arrays(path,**kw):
 tmp=path.with_name(path.name+'.tmp')
 with tmp.open('wb') as f:np.savez_compressed(f,**kw)
 os.replace(tmp,path)
def progress(**kw):save(O/'stage_b_manifest.json',dict(elapsed_seconds=perf_counter()-start,pid=os.getpid(),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw))
def checktime():
 if perf_counter()-start>c['stage_seconds']:raise TimeoutError('Frozen stageB budget')
def reference(dest,X,y,anchor,B):
 config=c['reference'];p=dict(X=jnp.asarray(X),y=jnp.asarray(y),anchor=jnp.asarray(anchor),B=jnp.asarray(B));pools=[];diagnostics=[]
 for pool,seed in enumerate(config['seeds']):
  begin=perf_counter();state=(jax.random.normal(jax.random.PRNGKey(seed),(4,len(anchor)),dtype=jnp.float64)*1.5,jnp.full(4,.25),jnp.asarray(0));samples=[];accept=[];div=[]
  for block in range((config['warmup']+config['retained_per_chain'])//128):
   checktime()
   if perf_counter()-begin>config['per_pool_seconds']:raise TimeoutError('Frozen per-reference-pool budget')
   state,(z,acc,dd)=base.ready(hmc_kernel(block<config['warmup']//128)(p,state,seed))
   if block>=config['warmup']//128:samples.append(np.asarray(z).transpose(1,0,2));accept.append(np.asarray(acc));div.append(np.asarray(dd))
   if (block+1)%8==0:
    arrays(dest/f'reference_progress_{pool}.npz',z=np.asarray(state[0]),eps=np.asarray(state[1]),t=np.asarray(state[2]),retained_z=np.concatenate(samples,axis=1) if samples else np.empty((4,0,len(anchor))))
    progress(status='running',stage='reference',version=dest.name,pool=pool,transitions=(block+1)*128);print('REFERENCE',dest.name,pool,(block+1)*128,flush=True)
  theta=anchor+np.concatenate(samples,axis=1)@B.T;arrays(dest/f'reference_{pool}.npz',samples=theta);subprocess.run([str(R/'.venv/bin/python'),str(H/'diagnose_reference.py'),str(dest/f'reference_{pool}.npz'),str(dest/f'reference_diagnostic_{pool}.json')],check=True,env=dict(os.environ,MPLCONFIGDIR='/private/tmp/cv2_mpl'));diag=json.loads((dest/f'reference_diagnostic_{pool}.json').read_text());diag.update(acceptance=np.concatenate(accept).mean(0).tolist(),divergences=np.concatenate(div).sum(0).tolist(),seconds=perf_counter()-begin,seed=seed,warmup=config['warmup'],retained=config['retained_per_chain']);diag['reliable']=bool(diag['reliable'] and min(diag['acceptance'])>.5 and sum(diag['divergences'])==0);arrays(dest/f'reference_{pool}.npz',samples=theta);save(dest/f'reference_{pool}.json',diag);pools.append(theta);diagnostics.append(diag)
 pair=validate_reference_pair(*pools,*diagnostics);save(dest/'reference_pair.json',pair)
 if not pair['reliable']:return None
 scoring=[]
 for pool,x in enumerate(pools):
  flat=x.reshape(-1,x.shape[-1]);ix=np.random.default_rng(config['scoring_selection_seed']+pool).choice(len(flat),config['scoring_draws_per_pool'],replace=False);scoring.append(flat[ix])
 return scoring

def run_job(dest,version,X,y,anchor,W,refs,proj,meta,phase,method,n,h,seed,replicas,maxk,checkpoints):
 rid=f'{phase}_{method}_n{n}_h{h}_seed{seed}';file=dest/(rid+'.json')
 if file.exists():return json.loads(file.read_text())
 checktime();key=(method,n,replicas)
 if key not in compiled:
  pp,_=base.prepare(X,y,anchor,1.,'logistic',method,'full',W);ss=base.ready(base.initial_state(pp,replicas));fn=sample_kernel(method,'full',n,.99);t=perf_counter();exe=fn.lower(pp,ss,204499,h).compile();compile_s=perf_counter()-t;base.ready(exe(pp,ss,204499,h));compiled[key]=(exe,compile_s)
 exe,compile_s=compiled[key];t=perf_counter();p,setup=base.prepare(X,y,anchor,1.,'logistic',method,'full',W);state=base.ready(base.initial_state(p,replicas));setup_s=perf_counter()-t;sums=jnp.zeros((replicas,2*len(anchor)));sample_s=0.;functional_s=0.;score_s=0.;points=[]
 for k in range(128,maxk+1,128):
  checktime();t=perf_counter();state,xx=base.ready(exe(p,state,seed,h));sample_s+=perf_counter()-t;t=perf_counter();sums=base.ready(sums+base.sum_functionals(xx,p['anchor'],p['transform']));functional_s+=perf_counter()-t
  if k in checkpoints:
   failed=np.asarray(state[3]);t=perf_counter();q=base.score(np.asarray(xx[-1]),np.asarray(sums),k,refs,anchor,W,proj);score_s+=perf_counter()-t
   if np.any(failed):q=dict(status='failed_ensemble',references=None)
   q.update(iteration=k,failed_replicas=int(np.sum(failed>0)),warm_total_seconds=meta['anchor_seconds']+setup_s+sample_s,cold_total_seconds=meta['anchor_seconds']+setup_s+sample_s+compile_s);points.append(q);arrays(dest/(rid+f'_K{k}.npz'),endpoints=np.asarray(xx[-1]),functional_sums=np.asarray(sums),failed_at=failed)
  if k%512==0:progress(status='running',stage=phase,version=version,method=method,batch=n,h=h,seed=seed,iteration=k)
 failed=np.asarray(state[3]);r=dict(version=version,phase=phase,method=method,batch=n,h=h,seed=seed,replicas=replicas,completed_iterations=maxk,failed_replicas=int(np.sum(failed>0)),failed_at=failed.tolist(),checkpoints=points,setup=setup,setup_seconds=setup_s,compile_seconds=compile_s,sampling_seconds=sample_s,functional_seconds=functional_s,scoring_seconds=score_s);save(file,r);print('SAMPLER',version,phase,method,n,h,seed,'failed',r['failed_replicas'],flush=True);return r
try:
 with threadpool_limits(limits=1):
  for version in c['versions']:
   dest=O/version;a=np.load(dest/'target.npz');X,y,anchor,W,B=[a[k] for k in ['X','y','anchor','W','B']];meta=json.loads((dest/'metadata.json').read_text());progress(status='running',stage='reference',version=version)
   if version=='original':
    pair=json.loads((A/'reference_pair.json').read_text());assert pair['reliable'];rr=np.load(A/'scoring.npz');refs=[rr['first'],rr['second']];save(dest/'reference_pair.json',dict(**pair,provenance='Reused unchanged original target approved references'))
   else:refs=reference(dest,X,y,anchor,B)
   if refs is None:save(dest/'STAGE_B_STATUS.json',dict(status='reference_gate_failed',samplers_started=False));results.append(dict(version=version,status='reference_gate_failed'));continue
   proj=np.load(A/'scoring.npz')['projections'];arrays(dest/'scoring.npz',first=refs[0],second=refs[1],projections=proj)
   pc=c['pilot'];runs=[]
   for method in pc['methods']:
    for n in pc['batch_sizes']:
     for h in pc['h_grid']:
      for seed in pc['seeds']:runs.append(run_job(dest,version,X,y,anchor,W,refs,proj,meta,'pilot',method,n,h,seed,pc['replicas'],pc['max_iterations'],pc['checkpoints']))
   save(dest/'pilot_summary.json',runs);choices=[]
   for method in pc['methods']:
    candidates=[]
    for n in pc['batch_sizes']:
     for h in pc['h_grid']:
      rs=[r for r in runs if (r['method'],r['batch'],r['h'])==(method,n,h)]
      if any(r['failed_replicas'] for r in rs):continue
      for k in pc['checkpoints']:
       ps=[next(q for q in r['checkpoints'] if q['iteration']==k) for r in rs];means=np.mean([[[z[t] for t in ['sw2','variance_nrmse']] for z in q['references']] for q in ps],axis=0).max(0);candidates.append(dict(method=method,batch=n,h=h,K=k,sw2=float(means[0]),variance_nrmse=float(means[1]),warm=float(np.mean([q['warm_total_seconds'] for q in ps])),reached=bool(means[0]<=.3 and means[1]<=.5)))
    reached=[q for q in candidates if q['reached']]
    if reached:choice=min(reached,key=lambda q:(q['warm'],q['batch'],q['h'],q['K']));choice['selection']='pilot_joint_attainment'
    elif candidates:choice=min([q for q in candidates if q['K']==pc['max_iterations']],key=lambda q:(max(q['sw2']/.3,q['variance_nrmse']/.5),q['batch'],q['h']));choice['selection']='fallback_no_pilot_attainment'
    else:choice=dict(method=method,selection='no_stable_candidate')
    choices.append(choice)
   save(dest/'FROZEN_CHOICES.json',choices);vr=[];vc=c['validation']
   for choice in choices:
    if choice['selection']=='no_stable_candidate':continue
    for seed in vc['seeds']:vr.append(run_job(dest,version,X,y,anchor,W,refs,proj,meta,'validation',choice['method'],choice['batch'],choice['h'],seed,vc['replicas'],vc['max_iterations'],vc['checkpoints']))
   save(dest/'validation_summary.json',vr);save(dest/'STAGE_B_STATUS.json',dict(status='complete',pilot_runs=len(runs),validation_runs=len(vr)));results.append(dict(version=version,status='complete',choices=choices))
 progress(status='complete',versions=results)
except Exception:
 progress(status='failed_or_budget_censored',error=traceback.format_exc());raise
