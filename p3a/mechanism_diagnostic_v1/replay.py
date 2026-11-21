from pathlib import Path
import sys,json,hashlib
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R/'src'),str(R/'gpu_replication'),str(R/'p2_screening')]
import jax
import jax.numpy as jnp
from jax import lax
from batched_sampler import prepare,initial_state,advance,ready
from kernels import sample_kernel,subset
D=R/'results/p3a_bank_jax_cpu_amendment1_validation_v1';O=R/'exports/p3a_mechanism_diagnostic_v1';HERE=Path(__file__).parent;A=R/'results/p3a_bank_t4_heldout_v1/frozen_bank_assets'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
runs=json.loads((D/'summary.json').read_text());cases=[]
for n,h,seed in [(1,.003125,203804),(16,.00625,203801),(64,.0125,203800)]:
 r=next(r for r in runs if (r['method'],r['batch'],r['h'],r['seed'])==('cv2',n,h,seed));t=min(x for x in r['failed_at'] if x);lane=r['failed_at'].index(t);cases.append(dict(batch=n,h=h,seed=seed,first_failure=t,lane=lane))
protocol=dict(stage='Bounded failure replay extension; fixed before replay',cases=cases,replicas=512,stream='Original sample_kernel fold_in order, same full shape, exact matching checkpoint512 required',scope='Three earliest failures from three distinct batch sizes. No retuning or new accuracy ranking.',source_sha256=sha(R/'p2_screening/kernels.py'),prior_protocol_sha256=sha(HERE/'PROTOCOL.json'))
(HERE/'REPLAY_PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n')
a=np.load(A/'target.npz');X,y,anchor,W=[a[k] for k in ['X','y','anchor','W']];N,d=X.shape;records=[]
with threadpool_limits(limits=1):
 p,_=prepare(X,y,anchor,1.,'logistic','cv2','full',W);L=np.asarray(p['L']);P=np.asarray(p['P']);p0=expit(X@anchor);weights=p0*(1-p0)
 for case in cases:
  n,h,seed,t,lane=[case[k] for k in ['batch','h','seed','first_failure','lane']];state=ready(initial_state(p,512));exe=sample_kernel('cv2','full',n,.99)
  for block in range(4):state,public=ready(exe(p,state,seed,h))
  z=np.load(D/f'checkpoint_cv2_n{n}_h{h}_seed{seed}_K512.npz');diff=float(np.nanmax(abs(np.asarray(public[-1])-z['endpoints'])));assert np.allclose(np.asarray(public[-1]),z['endpoints'],rtol=1e-11,atol=1e-11,equal_nan=True)
  roots=jax.vmap(lambda r:jax.random.fold_in(jax.random.PRNGKey(seed),r))(jnp.arange(512,dtype=jnp.uint32))
  def replay(p,state):
   def body(state,_):
    k=state[4];keys=jax.vmap(lambda key:jax.random.fold_in(key,k))(roots);idx=jax.vmap(lambda key:subset(jax.random.fold_in(key,1),N,n))(keys);normal=jax.vmap(lambda key:jax.random.normal(jax.random.fold_in(key,2),(d,),dtype=jnp.float64))(keys);new,public,parts=advance(p,state,idx,normal,h,'cv2','full','logistic',.99,1e-5)
    return new,(state[0][lane],idx[lane],normal[lane],*[v[lane] for v in parts],new[2][lane])
   return lax.scan(body,state,None,length=t-512)
  final,trace=ready(jax.jit(replay)(p,state));assert int(np.asarray(final[3])[lane])==t
  theta,idx,normal,g,corr,rem,raw,drift,noise,radius,live=[np.asarray(v) for v in trace];details=[]
  for j in range(max(0,len(theta)-10),len(theta)):
   delta=theta[j]-anchor;prob=expit(X@theta[j]);exact=X.T@(prob-y)+theta[j];r1=prob-p0;r2=r1-weights*(X@delta);basis={}
   for m,rr in [('cv1',r1),('cv2',r2)]:
    xx=X*rr[:,None];xx-=xx.mean(0);basis[m]=float(N*N*(N-n)/(n*(N-1))*np.mean(np.sum((xx@L)**2,axis=1)))
   details.append(dict(iteration=513+j,anchor_distance_before=float(np.linalg.norm(delta@W)),radius_after=float(radius[j]),live_after=bool(live[j]),batch_indices=idx[j].tolist(),sampled_max_abs_anchor_logit=float(abs(X[idx[j]]@anchor).max()),sampled_max_abs_delta_logit=float(abs(X[idx[j]]@delta).max()),preconditioned_gradient_error_norm=float(np.linalg.norm((g[j]-exact)@L)),preconditioned_correction_norm=float(np.linalg.norm(corr[j]@L)),preconditioned_random_remainder_norm=float(np.linalg.norm(rem[j]@L)),whitened_drift_norm=float(np.linalg.norm(drift[j]@W)),whitened_noise_norm=float(np.linalg.norm(noise[j]@W)),exact_conditional_noise_mse=basis,ratio_cv2_cv1=basis['cv2']/basis['cv1'] if basis['cv1'] else None))
  name=f"replay_n{n}_h{h}_seed{seed}";np.savez_compressed(O/(name+'.npz'),theta_before=theta,indices=idx,normal=normal,gradient=g,correction=corr,remainder=rem,drift=drift,noise=noise,radius_after=radius,live_after=live)
  rec=dict(**case,checkpoint512_max_absolute_difference=diff,checkpoint512_match=True,first_failure_reproduced=True,failure_threshold=100*np.sqrt(d),last10_steps=details);records.append(rec);print(json.dumps(rec),flush=True)
(O/'failure_replay.json').write_text(json.dumps(records,indent=2)+'\n')
