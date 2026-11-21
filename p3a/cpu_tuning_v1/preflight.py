from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import json,platform,sys
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info
from backend import prepare,gradient,transition,inputs,load_assets,save
from sgld_cv.models import LogisticRegression
from sgld_cv.estimators import prepare_estimator
from sgld_cv.costs import CostLedger
OUT=Path(__file__).resolve().parent/'PREFLIGHT.json'
with threadpool_limits(limits=1):
 a,refs,proj,meta=load_assets();p,setup=prepare(a);model=LogisticRegression(a['X'],a['y'],prior_precision=1.);anchor=SimpleNamespace(theta=a['anchor'],cost=CostLedger());rng=np.random.default_rng(203501);theta=refs[0][:4];records=[]
 for n in [1,4,16,32,64,256]:
  ix=np.array([rng.choice(model.N,n,replace=False) for _ in theta]);noise=rng.normal(size=theta.shape)
  for m in ['cv1','cv2','ula']:
   est=prepare_estimator(model,m,anchor);want=np.array([est.gradient(t,i) for t,i in zip(theta,ix)]);got=gradient(p,theta,ix,m);np.testing.assert_allclose(got,want,rtol=1e-9,atol=1e-8)
   h=.003125;actual=transition(p,theta,np.ones(4,bool),np.zeros(4,np.int32),ix,noise,h,m,0)[3];expected=theta-h*want@p['P'].T+np.sqrt(2*h)*noise@p['L'].T;np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)
   records.append(dict(batch=n,method=m,max_gradient_error=float(abs(got-want).max()),max_step_error=float(abs(actual-expected).max())))
 timings=[]
 for method,n in [('cv1',4),('cv2',4),('cv1',32),('cv2',32),('cv1',256),('cv2',256),('ula',32)]:
  R=128;steps=32;theta=np.broadcast_to(p['anchor'],(R,model.d)).copy();live=np.ones(R,bool);failed=np.zeros(R,np.int32);t=perf_counter();ix,noise=inputs(203502,0,R,n,model.N,model.d,method,chunk=steps);rng_s=perf_counter()-t;t=perf_counter()
  for k in range(steps):theta,live,failed,_=transition(p,theta,live,failed,None if ix is None else ix[k],noise[k],.003125,method,k)
  kernel_s=perf_counter()-t;timings.append(dict(method=method,batch=n,replicas=R,steps=steps,rng_seconds=rng_s,kernel_seconds=kernel_s,projected_8192_seconds=(rng_s+kernel_s)*8192/steps))
 # Paired streams across methods and step sizes; uniqueness and restart determinism.
 x,z=inputs(203503,3,8,16,model.N,model.d,'cv1',chunk=4);xx,zz=inputs(203503,3,8,16,model.N,model.d,'cv2',chunk=4);assert np.array_equal(x,xx) and np.array_equal(z,zz);assert all(len(set(row))==16 for row in x.reshape(-1,16))
 save(OUT,dict(status='passed',backend='NumPy CPU float64',python=sys.version,platform=platform.platform(),threads=threadpool_info(),formula_and_step_checks=records,stream_checks='passed',setup=setup,benchmark=timings,scope='Preflight only; no pilot/validation seeds consumed'))
 print(json.dumps(dict(status='passed',timings=timings),indent=2))
