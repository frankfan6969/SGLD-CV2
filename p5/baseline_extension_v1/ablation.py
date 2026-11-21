"""Small fixed-metric ablation, scheduled after the three-method ULA extension."""
from pathlib import Path
import sys,os,time,traceback,collections
from time import perf_counter
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R/'gpu_replication'),str(R/'p2_screening')]
import numpy as np,jax,jax.numpy as jnp
from scipy.special import expit
from threadpoolctl import threadpool_limits
from batched_sampler import prepare,initial_state,ready,gradient_parts
from metric_kernels import metric_kernel,metric_step
from extension_kernels import sum_functionals
from common import load,save,sha,score,arrays,ranks,H
from analyze import analyze
D=R/'results/p5_metric_ablation_v1';E=R/'exports/p5_metric_ablation_v1';E.mkdir(exist_ok=True);c=load(D/'PROTOCOL.json');started=perf_counter()
for rel,digest in c['input_hashes'].items():assert sha(R/rel)==digest,rel
assert jax.default_backend()=='cpu' and jax.config.x64_enabled
A=R/c['assets'];a=np.load(A/'target.npz');s=np.load(A/'scoring.npz');X,y,anchor,W,Hess=[a[k] for k in ['X','y','anchor','W','H']];refs=[s['first'],s['second']];proj=s['projections'];meta=load(A/'metadata.json')
for f in ['reference_pair.json','reference_0.json','reference_1.json']:assert load(A/f)['reliable']
used=set()
for root in ['results','exports']:
 for p in (R/root).rglob('*summary.json'):
  if 'p5_metric_ablation_v1' in str(p):continue
  try:
   q=load(p)
   if isinstance(q,list):used.update(int(v['seed']) for v in q if isinstance(v,dict) and isinstance(v.get('seed'),(int,float)))
  except (ValueError,TypeError):pass
assert not set(c['pilot']['seeds']+c['validation']['seeds'])&used
save(D/'seed_audit.json',dict(passed=True,seeds=c['pilot']['seeds']+c['validation']['seeds']))
save(D/'CODE_LOCK.json',{str(p.relative_to(R)):sha(p) for p in [H/'ablation.py',H/'metric_kernels.py',H/'analyze.py',D/'PROTOCOL.json']})
statuses=[]
def progress(stage,**kw):save(D/'STATUS.json',dict(status='running',stage=stage,elapsed_seconds=perf_counter()-started,geometries=statuses,pid=os.getpid(),**kw))
def prepare_metric(m,g):
 if g=='full':
  p,parts=prepare(X,y,anchor,1.,'logistic',m,'full',W);return p,dict(**parts,metric_assembly='full Hessian,solve,Cholesky; CV2 cache reuse as original',metric_apply='dense matrix',cv2_full_hessian=(m=='cv2'))
 begin=perf_counter();xx,yy,aa,ww=[jnp.asarray(q,dtype=jnp.float64) for q in [X,y,anchor,W]];ready((xx,yy,aa,ww));transfer=perf_counter()-begin;tt=perf_counter();p0=jax.nn.sigmoid(xx@aa);res=p0-yy;weights=p0*(1-p0);d=len(anchor)
 gh=xx.T@res if m in ['cv1','cv2'] else jnp.zeros(d);HH=(xx.T*weights)@xx if m=='cv2' else jnp.zeros((0,0));ready((gh,HH));estimator=perf_counter()-tt;tt=perf_counter();dd=1/(jnp.einsum('ij,i,ij->j',xx,weights,xx)+1.+1e-10) if g=='diagonal' else jnp.ones(d);ready(dd);metric=perf_counter()-tt
 p=dict(X=xx,y=yy,anchor=aa,prior=jnp.asarray(1.),residual_hat=res,weights_hat=weights,gradient_hat=gh,H_data=HH,D=dd,transform=ww)
 ready(p);return p,dict(transfer_seconds=transfer,estimator_cache_seconds=estimator,metric_assembly_seconds=metric,total_seconds=perf_counter()-begin,metric_assembly='diagonal weighted squared features' if g=='diagonal' else 'identity vector',metric_apply='elementwise drift/noise; common dense failure transform remains',cv2_full_hessian=(m=='cv2'),metric_hessian_contributions=len(X) if g=='diagonal' else 0,cv2_hessian_contributions=len(X) if m=='cv2' else 0)
def run_geometry(g):
 F=D/g;O=E/g;F.mkdir(exist_ok=True);O.mkdir(exist_ok=True);compiled={};records=[];LP=c['LP'][g]
 # Independent NumPy metric, estimator and update check before inference streams.
 rng=np.random.default_rng(430010);theta=refs[0][:3];ix=np.array([rng.choice(len(X),16,replace=False) for _ in theta]);normal=rng.normal(size=theta.shape);checks=[]
 for m in c['methods']:
  p,parts=prepare_metric(m,g);P=np.asarray(p['P']) if g=='full' else np.diag(np.asarray(p['D']));expectedP=np.linalg.inv(Hess+1e-10*np.eye(len(anchor))) if g=='full' else (np.diag(1/(np.diag(Hess)+1e-10)) if g=='diagonal' else np.eye(len(anchor)));np.testing.assert_allclose(P,expectedP,rtol=1e-8,atol=1e-10)
  if m=='ula':want=(expit(theta@X.T)-y)@X+theta
  else:
   p0=expit(X@anchor);gg=[]
   for q,ids in zip(theta,ix):
    delta=q-anchor;rem=expit(X[ids]@q)-p0[ids];correction=np.zeros_like(q)
    if m=='cv2':rem-=p0[ids]*(1-p0[ids])*(X[ids]@delta);correction=((X.T*(p0*(1-p0)))@X)@delta
    gg.append(X.T@(p0-y)+correction+len(X)/16*(X[ids].T@rem)+q)
   want=np.array(gg);got=np.asarray(ready(gradient_parts(p,jnp.asarray(theta),jnp.asarray(ix),m,'logistic')[0]));np.testing.assert_allclose(want,got,rtol=1e-9,atol=1e-8)
  noise_factor=np.linalg.cholesky(P);h=.003125/LP;expected=theta-h*(want@P.T)+np.sqrt(2*h)*(normal@noise_factor.T);state=list(initial_state(p,3));state[0]=jnp.asarray(theta);_,got=ready(metric_step(m,g,c['ula_exact_block'])(p,tuple(state),jnp.asarray(ix),jnp.asarray(normal),h));np.testing.assert_allclose(got,expected,rtol=1e-9,atol=1e-9);checks.append(dict(method=m,geometry=g,metric_error=float(abs(P-expectedP).max()),max_fixed_update_error=float(np.max(np.abs(np.asarray(got)-expected)))))
 save(F/'PREFLIGHT.json',dict(passed=True,checks=checks,LP=LP))
 def job(phase,m,n,h,seed,Rcount,horizon,checkpoints):
  if perf_counter()-started>c['budgets']['total_research_seconds']:raise TimeoutError('Ablation frozen3hour budget exhausted')
  rid=f'{phase}_{m}_n{n}_h{h:.12g}_seed{seed}';dest=F/(rid+'.json')
  if dest.exists():r=load(dest);records.append(r);return r
  key=(m,n,Rcount)
  if key not in compiled:
   p,_=prepare_metric(m,g);st=ready(initial_state(p,Rcount));fn=metric_kernel(m,g,n,c['ula_exact_block']);tt=perf_counter();exe=fn.lower(p,st,430010,h).compile();compile_s=perf_counter()-tt;tt=perf_counter();ready(exe(p,st,430010,h));compiled[key]=(exe,compile_s);save(F/f'compile_{m}_n{n}_R{Rcount}.json',dict(compile_seconds=compile_s,warmup_seconds=perf_counter()-tt))
  exe,compile_s=compiled[key];tt=perf_counter();p,parts=prepare_metric(m,g);state=ready(initial_state(p,Rcount));setup_s=perf_counter()-tt;sums=jnp.zeros((Rcount,2*len(anchor)));sampling=functional=evaluation=0.;points=[];last=0;censored=False
  for k in range(128,horizon+1,128):
   if perf_counter()-started>c['budgets']['total_research_seconds']:raise TimeoutError('Ablation frozen total budget exhausted')
   if sampling>c['budgets']['per_run_sampler_seconds']:censored=True;break
   tt=perf_counter();state,xx=ready(exe(p,state,seed,h));sampling+=perf_counter()-tt;last=k;tt=perf_counter();sums=ready(sums+sum_functionals(xx,p['anchor'],p['transform']));functional+=perf_counter()-tt;failed=np.asarray(state[3])
   if k in checkpoints:
    tt=perf_counter();q=score(np.asarray(xx[-1]),np.asarray(sums),k,anchor,W,proj,refs) if not np.any(failed) else dict(status='failed_ensemble',references=None);evaluation+=perf_counter()-tt;q.update(iteration=k,failed_replicas=int((failed>0).sum()),warm_total_seconds=meta['anchor_seconds']+setup_s+sampling,cold_total_seconds=meta['anchor_seconds']+setup_s+sampling+compile_s);points.append(q);arrays(F/f'{rid}_K{k}.npz',endpoints=np.asarray(xx[-1]),functional_sums=np.asarray(sums),failed_at=failed)
   progress('sampling',geometry=g,phase=phase,method=m,batch=n,seed=seed,iteration=k,runs_completed=len(records))
  r=dict(id=rid,geometry=g,phase=phase,method=m,batch=n,h=h,seed=seed,replicas=Rcount,completed_iterations=last,failed_replicas=int((np.asarray(state[3])>0).sum()),failed_at=np.asarray(state[3]).tolist(),budget_censored=censored,checkpoints=points,compile_seconds=compile_s,setup_seconds=setup_s,setup=parts,sampling_seconds=sampling,functional_seconds=functional,evaluation_seconds=evaluation,anchor_seconds_attributed=meta['anchor_seconds']);save(dest,r);records.append(r);save(F/'summary.json',records);print('RUN',g,len(records),rid,'failed',r['failed_replicas'],flush=True);return r
 jobs=[(m,n,coef/LP,seed) for m in c['methods'] for n in ([len(X)] if m=='ula' else c['batch_sizes']) for coef in c['step_coefficients'] for seed in c['pilot']['seeds']];np.random.default_rng(439000).shuffle(jobs);pilot=[job('pilot',m,n,h,seed,c['pilot']['replicas'],c['pilot']['horizon'],c['pilot']['checkpoints']) for m,n,h,seed in jobs];save(F/'pilot_summary.json',pilot);points=[]
 groups=collections.defaultdict(list)
 for r in pilot:groups[r['method'],r['batch'],r['h']].append(r)
 for key,rr in groups.items():
  if len(rr)!=2 or any(r['failed_replicas'] or r['budget_censored'] or r['completed_iterations']!=c['pilot']['horizon'] for r in rr):continue
  for k in c['pilot']['checkpoints']:
   pp=[next(p for p in r['checkpoints'] if p['iteration']==k) for r in rr];v=np.mean([[[q[z] for z in ['sw2','variance_nrmse']] for q in p['references']] for p in pp],0).max(0);points.append(dict(method=key[0],batch=key[1],h=key[2],K=k,warm=np.mean([p['warm_total_seconds'] for p in pp]),cold=np.mean([p['cold_total_seconds'] for p in pp]),sw2=v[0],variance=v[1]))
 chosen=[]
 for m in c['methods']:
  for n in ([len(X)] if m=='ula' else c['batch_sizes']):
   pp=[q for q in points if q['method']==m and q['batch']==n]
   if not pp:continue
   rr=ranks([[q[k] for k in ['warm','cold','sw2','variance']] for q in pp]);hs=sorted(set(q['h'] for q in pp));scores=[]
   for h in hs:
    ranks_h=[rank for q,rank in zip(pp,rr) if q['h']==h];scores.append(((-sum(x==0 for x in ranks_h),min(ranks_h),-sum(1/(1+x) for x in ranks_h),h),h))
   chosen.extend(dict(method=m,batch=n,h=h) for _,h in sorted(scores)[:2 if m=='ula' else 1])
 freeze=dict(candidates=chosen,max_iterations=c['validation']['horizon'],validation_seeds=c['validation']['seeds']);save(F/'FROZEN_CANDIDATES.json',freeze);jobs=[(q,seed) for q in chosen for seed in c['validation']['seeds']];np.random.default_rng(439001).shuffle(jobs);validation=[job('validation',q['method'],q['batch'],q['h'],seed,c['validation']['replicas'],c['validation']['horizon'],c['validation']['checkpoints']) for q,seed in jobs];save(F/'validation_summary.json',validation);analyze(F,O,c,c['validation']['seeds']);report=O/'REPORT.md';report.write_text(report.read_text().replace('target,full-H metric','target,'+g+' metric'));return dict(geometry=g,status='complete',pilot_runs=len(pilot),validation_runs=len(validation))
with threadpool_limits(limits=1):
 for g in c['geometries']:
  try:status=run_geometry(g)
  except Exception:status=dict(geometry=g,status='failed_or_censored',error=traceback.format_exc());print(status['error'],flush=True)
  statuses.append(status);save(D/g/'STATUS.json',status)
save(D/'STATUS.json',dict(status='complete' if all(q['status']=='complete' for q in statuses) else 'partial',geometries=statuses,elapsed_seconds=perf_counter()-started))
md=['# Fixed-metric ablation','','Log1p Bank; steps scaled by each actual LP, common finite search grid,n in1/16/64,full-data ULA. Diagonal means inverse Hessian diagonal,not diagonal of inverse. CV2 keeps full correction. Separate per-geometry simultaneous bands; not joint across geometry.','']
for q in statuses:
 p=E/q['geometry']/'REPORT.md';md+=['## '+q['geometry'],'',p.read_text() if p.exists() else 'Incomplete; no comparison imputed.','']
(E/'REPORT.md').write_text('\n'.join(md)+'\n');print('FIXED_METRIC_ABLATION_FINISHED',flush=True)
