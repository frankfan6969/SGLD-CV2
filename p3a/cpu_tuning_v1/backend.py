"""NumPy float64 CPU backend; same CV formulas and full-H Langevin convention."""
from pathlib import Path
import json,os,sys,time
from time import perf_counter
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from sgld_cv.transport import AccuracyMetric
ASSETS=ROOT/'results/p3a_bank_t4_heldout_v1/frozen_bank_assets'
def save(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));os.replace(tmp,path)
def save_arrays(path,**kw):
 path=Path(path);tmp=path.with_name(path.name+'.tmp')
 with tmp.open('wb') as f:np.savez_compressed(f,**kw)
 os.replace(tmp,path)
def load_assets():
 a=np.load(ASSETS/'target.npz');r=np.load(ASSETS/'scoring.npz');return ({k:a[k] for k in a.files},[r['first'],r['second']],r['projections'],json.loads((ASSETS/'target_geometry.json').read_text()))
def prepare(a):
 t=perf_counter();X,y,anchor,W=[a[k] for k in ['X','y','anchor','W']]
 t0=perf_counter();ph=expit(X@anchor);res=ph-y;ghat=res@X;gradient_s=perf_counter()-t0
 t0=perf_counter();w=ph*(1-ph);Hd=(X.T*w)@X;H=Hd+np.eye(X.shape[1]);hessian_s=perf_counter()-t0
 np.testing.assert_allclose(H,a['H'],rtol=1e-10,atol=1e-8)
 t0=perf_counter();regularized=(Hd+Hd.T)/2+(1+1e-10)*np.eye(len(H));np.linalg.cholesky(regularized);P=np.linalg.solve(regularized,np.eye(len(H)));P=(P+P.T)/2;L=np.linalg.cholesky(P);factor_s=perf_counter()-t0
 LP=float(np.linalg.eigvalsh(L.T@H@L)[-1]);assert abs(LP-1)<1e-7
 return dict(X=X,y=y,anchor=anchor,W=W,res=res,ghat=ghat,w=w,Hd=Hd,P=P,L=L,LP=LP),dict(anchor_gradient_seconds=gradient_s,hessian_seconds=hessian_s,metric_solve_and_noise_factor_seconds=factor_s,total_seconds=perf_counter()-t,scope='Full-H required by every arm; CV2 reuses that unregularized data Hessian, no duplicate construction')
def gradient(p,theta,ix,method):
 if method=='ula':return (expit(theta@p['X'].T)-p['y'])@p['X']+theta
 xb=p['X'][ix];res=expit(np.einsum('rbd,rd->rb',xb,theta))-p['y'][ix]
 if method=='sgld':return len(p['X'])/ix.shape[1]*np.einsum('rbd,rb->rd',xb,res)+theta
 rem=res-p['res'][ix];g=p['ghat']+theta
 if method=='cv2':
  delta=theta-p['anchor'];rem=rem-p['w'][ix]*np.einsum('rbd,rd->rb',xb,delta);g=g+delta@p['Hd'].T
 return g+len(p['X'])/ix.shape[1]*np.einsum('rbd,rb->rd',xb,rem)
def transition(p,theta,live,failed,ix,noise,h,method,t):
 g=gradient(p,theta,ix,method);proposal=theta-h*(g@p['P'].T)+np.sqrt(2*h)*(noise@p['L'].T)
 radius=np.linalg.norm((proposal-p['anchor'])@p['W'],axis=1);good=np.isfinite(proposal).all(1)&np.isfinite(radius)&(radius<=100*np.sqrt(theta.shape[1]));first=live&~good;failed=np.where(first,t+1,failed);live=live&good;theta=np.where(live[:,None],proposal,0.)
 return theta,live,failed,np.where(live[:,None],theta,np.nan)
def inputs(seed,block,R,n,N,d,method,chunk=128):
 ix=None if method=='ula' else np.empty((chunk,R,n),dtype=np.int32);noise=np.empty((chunk,R,d),dtype=np.float64)
 for r in range(R):
  nr=np.random.default_rng(np.random.SeedSequence([seed,r,2,block]));noise[:,r]=nr.standard_normal((chunk,d))
  if method!='ula':
   br=np.random.default_rng(np.random.SeedSequence([seed,r,1,block]));ix[:,r]=br.integers(N,size=(chunk,1)) if n==1 else np.array([br.choice(N,n,replace=False) for _ in range(chunk)],dtype=np.int32)
 return ix,noise
def score(theta,sums,k,p,refs,proj,failed):
 count=int((failed>0).sum())
 if count:return dict(status='failed_ensemble',references=None,failed_replicas=count,failure_iteration=int(failed[failed>0].min()))
 d=theta.shape[1];avg=sums/k;var=avg[:,d:]-avg[:,:d]**2;metric=AccuracyMetric('standardized_sliced_w2',p['anchor'],p['W'],proj);values=[]
 for ref in refs:
  v=((ref-p['anchor'])@p['W']).var(0);values.append(dict(sw2=metric.distance(theta,ref),variance_nrmse=float(np.sqrt(np.mean(((var-v)/v)**2,axis=0)).mean())))
 return dict(status='ok',references=values,failed_replicas=0,failure_iteration=None)
