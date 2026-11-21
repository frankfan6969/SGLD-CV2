"""Existing initialization inventory plus bounded matched new diagnostic."""
from pathlib import Path
import sys,time,json
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R/'gpu_replication'),str(R/'p2_screening')]
import numpy as np,jax,jax.numpy as jnp
from scipy.special import expit
from batched_sampler import prepare,initial_state,ready
from kernels import sample_kernel
from common import *
def main():
 c=load(H/'PROTOCOL.json')['li_diagnostic'];F=D/'li_initialization';F.mkdir(exist_ok=True);inventory=[]
 for name in ['stage456_synthetic','german_sliced_heldout_long_reference']:
  f=R/'results'/name;pilots=load(f/'metric_pilots.json');diags=[]
  for p in f.glob('diagnostics_p*.json'):
   obj=load(p);v=obj['diagnostics'];diags.append(dict(file=str(p.relative_to(R)),reliable=v['reliable'],max_rhat=v['max_rhat']))
  inventory.append(dict(study=name,initialization='independent anchor-batch RMSprop pilot,NOT zero',metric_states=[dict(batch=b,min_v=min(q['v']),max_initial_diagonal=max(q['diagonal'])) for b,q in pilots.items()],diagnostics=diags,interpretation='Different historical parameters/targets; not a paired zero-versus-pilot causal comparison'))
 z=np.load(R/c['target']);X,y,anchor,W=[z[k] for k in ['X','y','anchor','transform']];rng=np.random.default_rng(c['pilot_seed']);v=np.zeros(len(anchor));t=time.perf_counter()
 for _ in range(c['pilot_batches']):
  ix=rng.choice(len(X),c['batch'],replace=False);raw=(expit(X[ix]@anchor)-y[ix])@X[ix]/c['batch'];v=c['alpha']*v+(1-c['alpha'])*raw*raw
 pilot_seconds=time.perf_counter()-t;save(F/'FROZEN_INITIALIZATION.json',dict(v=v.tolist(),pilot_seconds=pilot_seconds,pilot_batches=c['pilot_batches'],pilot_seed=c['pilot_seed'],initial_diagonal=(1/(c['epsilon']+np.sqrt(v))).tolist(),scope='Common independent initializer; originaltarget/h/alpha/epsilon untouched'))
 records=[]
 for method in c['methods']:
  params,_=prepare(X,y,anchor,1.,'logistic',method,'li2016',W);fn=sample_kernel(method,'li2016',c['batch'],c['alpha'],c['epsilon']);st=ready(initial_state(params,c['replicas']));exe=fn.lower(params,st,410010,c['h']).compile();ready(exe(params,st,410010,c['h']))
  for seed in c['seeds']:
   for init in c['initializations']:
    dest=F/f'{method}_{init}_{seed}.json'
    if dest.exists():records.append(load(dest));continue
    st=list(initial_state(params,c['replicas']));st[1]=jnp.asarray(np.broadcast_to(v if init=='independent_anchor_pilot' else np.zeros_like(v),(c['replicas'],len(v))));st=ready(tuple(st));trace=[];elapsed=0.
    for _ in range(c['iterations']//128):
     t=time.perf_counter();st,states=ready(exe(params,st,seed,c['h']));elapsed+=time.perf_counter()-t;trace.append(dict(K=int(st[4]),failed=int(np.sum(np.asarray(st[3])>0)),max_metric=float(np.max(1/(c['epsilon']+np.sqrt(np.asarray(st[1])))))))
    fail=np.asarray(st[3]);r=dict(method=method,initialization=init,seed=seed,failed_replicas=int((fail>0).sum()),first_failure=int(fail[fail>0].min()) if (fail>0).any() else None,failed_at=fail.tolist(),sampling_seconds=elapsed,trace=trace);save(dest,r);records.append(r)
 save(F/'summary.json',records);save(E/'LI_INITIALIZATION_AUDIT.json',dict(existing_inventory=inventory,new_matched_diagnostic=records,scope=c['purpose'],new_rng_streams=True,original_failures_preserved=True))
 md=['# Li initialization audit and bounded sensitivity','','P1 deliberately used zero initialization; earlier studies also tested independent anchor-pilot initialization. Neither is silently called a bug fix. Old pilot-initialized studies did not supply an optimized reliable adaptive frontier.','','New matched diagnostic: German target,h=1.3203191532898161e-5,alpha=.99,epsilon=1e-5,batch32,R64,K2048,two fresh seeds. Only v0 changes; common minibatch and Gaussian streams are preserved. These JAX streams do not exactly replay old NumPy failures.','','|Method|Seed|Initialization|Failed/R|First failure|','|---|---:|---|---:|---:|']
 for q in records:md.append(f"|{q['method']}|{q['seed']}|{q['initialization']}|{q['failed_replicas']}/64|{q['first_failure']}|")
 md+=['','Zero observed failures is finite-budget evidence, not mixing,accuracy or an adaptive-efficiency claim. If an alternative initializer improves stability, it remains a declared initialization variant requiring its own tuning/validation before ranking. Main fixed-full-H scope remains unchanged.'];(E/'LI_INITIALIZATION_AUDIT.md').write_text('\n'.join(md)+'\n');print('Li inventory and12 matched diagnostic runs complete',flush=True)
if __name__=='__main__':main()
