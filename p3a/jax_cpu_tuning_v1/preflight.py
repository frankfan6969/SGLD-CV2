from pathlib import Path
import sys,json,platform,os
from time import perf_counter
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'gpu_replication'),str(ROOT/'p2_screening')]
import run as base
import numpy as np
import jax
import jax.numpy as jnp
from kernels import sample_kernel,subset
from batched_sampler import gradient_parts,advance
from threadpoolctl import threadpool_limits
from sgld_cv.models import LogisticRegression
from sgld_cv.estimators import prepare_estimator
from sgld_cv.costs import CostLedger
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;ASSETS=ROOT/'results/p3a_bank_t4_heldout_v1/frozen_bank_assets'
assert jax.default_backend()=='cpu' and jax.config.x64_enabled
with threadpool_limits(limits=1):
 a=np.load(ASSETS/'target.npz');X,y,anchor,W=[a[k] for k in ['X','y','anchor','W']];refs=np.load(ASSETS/'scoring.npz')['first'];model=LogisticRegression(X,y,prior_precision=1.);anc=SimpleNamespace(theta=anchor,cost=CostLedger());theta=refs[:4];rng=np.random.default_rng(203501);checks=[]
 for n in [1,4,16,32,64,256]:
  ix=np.array([rng.choice(len(X),n,replace=False) for _ in theta]);noise=rng.normal(size=theta.shape)
  for m in ['cv1','cv2','ula']:
   p,_=base.prepare(X,y,anchor,1.,'logistic',m,'full',W);got=np.asarray(base.ready(gradient_parts(p,jnp.asarray(theta),jnp.asarray(ix),m,'logistic')[0]));est=prepare_estimator(model,m,anc);want=np.array([est.gradient(t,i) for t,i in zip(theta,ix)]);np.testing.assert_allclose(got,want,rtol=1e-9,atol=1e-8)
   state=list(base.initial_state(p,4));state[0]=jnp.asarray(theta);new,public,_=base.ready(advance(p,tuple(state),jnp.asarray(ix),jnp.asarray(noise),.003125,m,'full','logistic',.99,1e-5));expected=theta-.003125*(want@np.asarray(p['P']).T)+np.sqrt(.00625)*(noise@np.asarray(p['L']).T);np.testing.assert_allclose(np.asarray(public),expected,rtol=1e-10,atol=1e-10)
   keys=jax.random.split(jax.random.PRNGKey(203502+n),4);draws=np.asarray(base.ready(jax.jit(jax.vmap(lambda k:subset(k,len(X),n)))(keys)));assert all(len(set(row))==n for row in draws)
   checks.append(dict(method=m,batch=n,max_gradient_error=float(abs(got-want).max()),max_step_error=float(abs(np.asarray(public)-expected).max())))
 print('JAX_CPU_PARITY_PASSED',flush=True);bench=[]
 for m,n in [('cv1',4),('cv2',4),('cv1',32),('cv2',32),('cv1',256),('cv2',256),('ula',32)]:
  p,_=base.prepare(X,y,anchor,1.,'logistic',m,'full',W);state=base.ready(base.initial_state(p,128));fn=sample_kernel(m,'full',n,.99);t=perf_counter();exe=fn.lower(p,state,203503,.003125).compile();compile_s=perf_counter()-t;base.ready(exe(p,state,203503,.003125));t=perf_counter();first=base.ready(exe(p,state,203503,.003125));seconds=perf_counter()-t
  # Exact deterministic checkpoint continuation; no new/randomized RNG stream on resume.
  s,_=first;restored=tuple(jnp.asarray(np.asarray(v)) for v in s);direct=base.ready(exe(p,s,203503,.003125));replayed=base.ready(exe(p,restored,203503,.003125));assert all(np.array_equal(np.asarray(x),np.asarray(y),equal_nan=True) for x,y in zip(direct[0],replayed[0]));assert np.array_equal(np.asarray(direct[1]),np.asarray(replayed[1]),equal_nan=True)
  q=dict(method=m,batch=n,replicas=128,compile_seconds=compile_s,sampler128_seconds=seconds,projected8192_seconds=seconds*64);bench.append(q);print(json.dumps(q),flush=True)
 base.save(HERE/'PREFLIGHT.json',dict(status='passed',backend='JAX CPU float64',jax=jax.__version__,python=sys.version,platform=platform.platform(),devices=[str(x) for x in jax.devices()],xla_flags=os.environ.get('XLA_FLAGS'),formula_and_step_checks=checks,checkpoint_roundtrip='passed bitwise for all benchmark shapes',benchmark=bench,scope='Validation and unscored benchmark only; no tuning/independent-validation seeds consumed'))
