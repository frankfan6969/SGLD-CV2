from pathlib import Path
import sys,json,hashlib,csv,time
from types import SimpleNamespace
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R/'src'),str(R/'gpu_replication'),str(R/'p2_screening')]
import jax
import jax.numpy as jnp
from batched_sampler import prepare,gradient_parts,advance,initial_state,ready
from sgld_cv.models import LogisticRegression
from sgld_cv.estimators import prepare_estimator
from sgld_cv.preconditioners import fixed_anchor_metric
from sgld_cv.costs import CostLedger
O=R/'exports/p3a_mechanism_diagnostic_v1';O.mkdir(exist_ok=True);HERE=Path(__file__).parent;D=R/'results/p3a_bank_jax_cpu_amendment1_validation_v1';A=R/'results/p3a_bank_t4_heldout_v1/frozen_bank_assets'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,x):(O/name).write_text(json.dumps(x,indent=2)+'\n')
c=json.loads((HERE/'PROTOCOL.json').read_text());assert all(sha(R/p)==h for p,h in c['input_hashes'].items());assert jax.default_backend()=='cpu' and jax.config.x64_enabled
start=time.perf_counter()
with threadpool_limits(limits=1):
 a=np.load(A/'target.npz');X,y,anchor,H,W=[a[k] for k in ['X','y','anchor','H','W']];refs=np.load(A/'scoring.npz');N,d=X.shape;model=LogisticRegression(X,y,prior_precision=1.);anc=SimpleNamespace(theta=anchor,cost=CostLedger());p0=expit(X@anchor);weights=p0*(1-p0);Hd=(X.T*weights)@X;ga=X.T@(p0-y);P=np.linalg.inv(Hd+(1.+1e-10)*np.eye(d));P=(P+P.T)/2;L=np.linalg.cholesky(P)
 np.testing.assert_allclose(H,Hd+np.eye(d),rtol=1e-10,atol=1e-8)
 params={m:prepare(X,y,anchor,1.,'logistic',m,'full',W)[0] for m in ['cv1','cv2','ula']};estimators={m:prepare_estimator(model,m,anc) for m in params};audit=[]
 for m,p in params.items():
  np.testing.assert_allclose(np.asarray(p['P']),P,rtol=1e-9,atol=1e-10);np.testing.assert_allclose(np.asarray(p['L'])@np.asarray(p['L']).T,P,rtol=1e-9,atol=1e-10)
  pm,_,_=fixed_anchor_metric(model,anc,estimators[m]);np.testing.assert_allclose(pm,P,rtol=1e-9,atol=1e-10)
 np.testing.assert_array_equal(np.asarray(params['cv1']['P']),np.asarray(params['cv2']['P']))
 locations=[]
 for i,s in enumerate(c['locations']):
  if s['group']=='anchor':theta=anchor.copy()
  elif 'file' in s:theta=np.load(D/s['file'])['endpoints'][s['lane']]
  else:theta=anchor+s.get('scale',1.)*(refs[s['reference']][s['index']]-anchor)
  assert np.isfinite(theta).all();locations.append((s,theta))
 rng=np.random.default_rng(c['seed']);noise_rows=[];mc=[]
 for loc,(spec,theta) in enumerate(locations):
  delta=theta-anchor;prob=expit(X@theta);res1=prob-p0;res2=res1-weights*(X@delta);full=X.T@(prob-y)+theta;distance=float(np.linalg.norm(delta@W));rowbase=dict(location=loc,group=spec['group'],anchor_distance=distance,gradient_preconditioned_norm=float(np.linalg.norm(full@L)),**{k:v for k,v in spec.items() if k!='group'})
  residuals={'cv1':res1,'cv2':res2};basis={};concentration={}
  for m,res in residuals.items():
   rr=X*res[:,None];center=rr-rr.mean(0);cov=np.cov(rr,rowvar=False,ddof=1);np.testing.assert_allclose(np.trace(P@cov),np.sum((center@L)**2)/(N-1),rtol=1e-10,atol=1e-12)
   basis[m]=(float(np.mean(np.sum(center*center,axis=1))),float(np.mean(np.sum((center@L)**2,axis=1))))
   sq=np.sum((center@L)**2,axis=1);ids=np.argsort(sq);concentration[m]=dict(top1_observation_fraction=float(sq[ids[-1]]/sq.sum()) if sq.sum() else 0.,top10_observation_fraction=float(sq[ids[-10:]].sum()/sq.sum()) if sq.sum() else 0.,largest_observation=int(ids[-1]))
  for n in c['batch_sizes']:
   factor=N*N*(N-n)/(n*(N-1))
   noise_rows.append(dict(**rowbase,batch_size=n,cv1_raw_mse=factor*basis['cv1'][0],cv2_raw_mse=factor*basis['cv2'][0],cv1_preconditioned_mse=factor*basis['cv1'][1],cv2_preconditioned_mse=factor*basis['cv2'][1],cv1_top1_fraction=concentration['cv1']['top1_observation_fraction'],cv2_top1_fraction=concentration['cv2']['top1_observation_fraction'],cv1_top10_fraction=concentration['cv1']['top10_observation_fraction'],cv2_top10_fraction=concentration['cv2']['top10_observation_fraction'],ratio_cv2_cv1=basis['cv2'][1]/basis['cv1'][1] if basis['cv1'][1]>0 else None))
  if loc<17: # independent references and anchor; all batches, same theta/indices/noise for both methods
   for n in c['batch_sizes']:
    idx=rng.choice(N,n,replace=False)[None,:];normal=rng.normal(size=(1,d))
    for m,p in params.items():
     gj,correction,_,_=ready(gradient_parts(p,jnp.asarray(theta[None,:]),jnp.asarray(idx),m,'logistic'));got=np.asarray(gj)[0];want=estimators[m].gradient(theta,idx[0]);np.testing.assert_allclose(got,want,rtol=1e-9,atol=1e-8)
     analytic=full if m=='ula' else ga+(Hd@delta if m=='cv2' else 0)+(N/n)*(X[idx[0]].T@residuals[m][idx[0]])+theta
     np.testing.assert_allclose(got,analytic,rtol=1e-9,atol=1e-8)
     if m=='cv2':np.testing.assert_allclose(np.asarray(correction)[0],Hd@delta,rtol=1e-9,atol=1e-8)
     maxstep=0.
     for h in c['audit']['step_sizes']:
      state=list(initial_state(p,1));state[0]=jnp.asarray(theta[None,:]);_,public,_=ready(advance(p,tuple(state),jnp.asarray(idx),jnp.asarray(normal),h,m,'full','logistic',.99,1e-5));expected=theta-h*(P@want)+np.sqrt(2*h)*(L@normal[0]);np.testing.assert_allclose(np.asarray(public)[0],expected,rtol=1e-9,atol=1e-8);maxstep=max(maxstep,float(abs(np.asarray(public)[0]-expected).max()))
     audit.append(dict(location=loc,method=m,batch=n,max_gradient_absolute_error=float(abs(got-want).max()),max_step_absolute_error=maxstep))
  if spec['group']=='reference' and spec['index']==0:
   for n in c['monte_carlo']['batch_sizes']:
    indices=np.array([rng.choice(N,n,replace=False) for _ in range(c['monte_carlo']['draws'])]);measured={}
    for m in ['cv1','cv2']:
     gradients=ga+(Hd@delta if m=='cv2' else 0)+(N/n)*np.einsum('mbd,mb->md',X[indices],residuals[m][indices])+theta
     
     for j in range(8):np.testing.assert_allclose(gradients[j],estimators[m].gradient(theta,indices[j]),rtol=1e-9,atol=1e-8)
     errors=(gradients-full)@L;sq=np.sum(errors**2,axis=1);exact=N*N*(N-n)/(n*(N-1))*basis[m][1];mean=float(sq.mean());se=float(sq.std(ddof=1)/np.sqrt(len(sq)));z=(mean-exact)/se if se else 0.;print("MC_CHECK",loc,n,m,mean,exact,se,z,flush=True)
     measured[m]=dict(largest_observation=concentration[m]['largest_observation'],largest_observation_hits=int(np.sum(indices==concentration[m]['largest_observation'])),exact_mse=exact,mc_mse=mean,mc_standard_error=se,z_score=z,preconditioned_mean_error_norm=float(np.linalg.norm(errors.mean(0))))
    mc.append(dict(scope='Descriptive MC estimate; empirical SE can miss rare-observation tails. Per-draw gradient algebra checked separately.',location=loc,batch_size=n,draws=len(indices),methods=measured))
  print('LOCATION',loc+1,'/',len(locations),spec['group'],flush=True)
 save('gradient_noise.json',noise_rows);save('implementation_audit.json',dict(status='passed',checks=audit,P_equal_cv1_cv2=True,P_matches_independent_and_P1_fixed_metric=True,noise_covariance_error=float(abs(L@L.T-P).max()),H_data_matches_stored_H_minus_prior=True,backend=str(jax.devices()),jax=jax.__version__,protocol_sha256=sha(HERE/'PROTOCOL.json')));save('monte_carlo_crosscheck.json',mc)
 # Every stored time identity; no redefinition of original timings.
 runs=json.loads((D/'summary.json').read_text());anchor_s=json.loads((A/'target_geometry.json').read_text())['anchor_seconds'];cost=[]
 for r in runs:
  assert not r['timing_interrupted'];last=r['checkpoints'][-1];np.testing.assert_allclose(last['warm_total_seconds'],anchor_s+r['setup_seconds']+r['sampling_seconds'],rtol=1e-12)
  for q in r['checkpoints']:np.testing.assert_allclose(q['cold_total_seconds']-q['warm_total_seconds'],r['compile_seconds'],rtol=1e-12,atol=1e-12)
  cost.append({k:r[k] for k in ['method','batch','h','seed','setup_seconds','compile_seconds','sampling_seconds','functional_seconds','evaluation_seconds']})
 save('cost_audit.json',dict(status='passed',runs_checked=len(cost),anchor_seconds_attributed=anchor_s,records=cost,scope='Warm: legacy anchor attribution + current prepare/state setup + synchronized sampling. Cold adds measured shape compilation. Functionals/evaluation/checkpoint I/O excluded for both. Current cache_metric_seconds aggregates Hessian/factorization; cannot retrospectively split. Full-H CV1 also constructs Hessian; CV2 reuses it. Actual process elapsed is not time-to-accuracy.'))
 # Source parity is byte identity where locked; no claim of bitwise RNG equality across NumPy/JAX.
 lock=json.loads((R/'p2_screening/lock.json').read_text());matches={k:sha(R/k)==v for k,v in lock['files'].items()};save('source_parity.json',dict(p2_locked_files=matches,all_p2_hashes_match=all(matches.values()),P1_numpy_independent_check='Current standalone estimator plus fixed_anchor_metric exercised above',RNG_scope='Fixed input algebra parity, not identical RNG generators across backends'))
 groups={}
 for group in sorted(set(x['group'] for x in noise_rows)):
  rows=[x for x in noise_rows if x['group']==group and x['batch_size']==32 and x['ratio_cv2_cv1'] is not None];rat=np.array([x['ratio_cv2_cv1'] for x in rows]);groups[group]=dict(locations=len(rows),median_ratio=float(np.median(rat)) if len(rat) else None,min_ratio=float(rat.min()) if len(rat) else None,max_ratio=float(rat.max()) if len(rat) else None,p90_ratio=float(np.quantile(rat,.9)) if len(rat) else None,cv2_lower_count=int((rat<1).sum()),mean_cv1_preconditioned_mse=float(np.mean([x['cv1_preconditioned_mse'] for x in rows])) if rows else 0.,mean_cv2_preconditioned_mse=float(np.mean([x['cv2_preconditioned_mse'] for x in rows])) if rows else 0.)
 save('summary.json',dict(status='complete',elapsed_seconds=time.perf_counter()-start,locations=len(locations),audit_checks=len(audit),groups=groups,source_lock_passed=all(matches.values()),protocol_sha256=sha(HERE/'PROTOCOL.json')))
 with (O/'gradient_noise.csv').open('w') as f:
  fields=list(dict.fromkeys(k for row in noise_rows for k in row));w=csv.DictWriter(f,fields);w.writeheader();w.writerows(noise_rows)
 print(json.dumps(groups,indent=2),flush=True)
