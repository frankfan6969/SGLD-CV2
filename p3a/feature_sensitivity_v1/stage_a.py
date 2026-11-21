from pathlib import Path
import sys,json,time,hashlib
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'src'))
from sgld_cv.models import LogisticRegression
from types import SimpleNamespace
def fit_mode(model):
 theta=np.zeros(model.d);start=time.perf_counter()
 for iteration in range(50):
  g=model.full_gradient(theta);norm=float(np.linalg.norm(g))
  if norm<1e-7:break
  step=np.linalg.solve(model.full_hessian(theta),g);energy=model.potential(theta);alpha=1.
  while model.potential(theta-alpha*step)>energy+1e-10:
   alpha*=.5
   if alpha<2**-30:raise RuntimeError('Mode line search failed')
  theta-=alpha*step
 else:raise RuntimeError('Mode residual gate failed')
 return SimpleNamespace(theta=theta,metadata=dict(algorithm='Damped Newton, full Hessian, zero initialization',iterations=iteration,gradient_norm=norm,gradient_norm_gate=1e-7),seconds=time.perf_counter()-start)
H=Path(__file__).parent;O=R/'exports/p3a_feature_sensitivity_v1';A=R/'results/p3a_bank_t4_heldout_v1/frozen_bank_assets';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();save=lambda n,x:(O/n).write_text(json.dumps(x,indent=2)+'\n')
a=np.load(A/'target.npz');meta=json.loads((A/'target_geometry.json').read_text());X,y=a['X'],a['y'];j=meta['features'].index('previous');raw=X[:,j]*meta['preprocessing']['previous']['scale']+meta['preprocessing']['previous']['mean'];np.testing.assert_allclose(raw,np.rint(raw),atol=1e-10);raw=np.rint(raw);assert raw.min()>=0
c=dict(name='Bank exploratory feature sensitivity v1',versions=['original','log1p_previous','winsor999_previous'],transform_column='previous',winsor_quantile=.999,quantile_method='linear',standardization='Each transformed column uses its own full-data mean and population std; all other columns and labels identical',mode_solver='Damped Newton up to50 iterations, full Hessian, backtracking, zero initialization, independent gradient norm <1e-7; scipy Newton-CG precision-loss warning on Original before transformed computation prompted this documented numerical adjustment',prior_precision=1.,targets='Three distinct posteriors, same Gaussian prior form, not target-preserving reparameterizations',seed=204200,directions=32,radii_in_sqrt_d=[1.,2.,4.],location_construction='16 independent normalized Gaussian whitened directions plus their negatives, shared across targets; theta=mode+radius*B*direction, B=symmetric inverse sqrt posterior Hessian',batch=32,mc_draws=512,mc_locations=[0,32,64],stage_a_metrics='Raw/preconditioned residual norm max,q99,q999; exact finite-population noise;29182 share; MC update-noise tail descriptive',stage_b_eligibility='Per variant, at radius sqrt(d), median across directions of max CV2 preconditioned residual norm <= half original and median exact CV2 preconditioned noise <= original. Mechanism screen only, not a statistical test.',stage_b_scope='Only eligible transformed targets plus newly rerun original control. Reference gates unchanged. Separate frozen bounded sampler protocol before stageB.',budget_seconds=600,input_hashes={str(p.relative_to(R)):sha(p) for p in [A/'target.npz',A/'target_geometry.json',R/'src/sgld_cv/models.py',R/'src/sgld_cv/anchors.py']},note='Exploratory protocol frozen before transformed diagnostic computation; no thresholds selected from transformed results')
if (H/'PROTOCOL.json').exists():assert json.loads((H/'PROTOCOL.json').read_text())==c
else:(H/'PROTOCOL.json').write_text(json.dumps(c,indent=2)+'\n')
start=time.perf_counter();rng=np.random.default_rng(c['seed']);u=rng.normal(size=(16,X.shape[1]));u/=np.linalg.norm(u,axis=1)[:,None];dirs=np.concatenate([u,-u]);rows=[];targets=[];cut=float(np.quantile(raw,.999,method='linear'))
with threadpool_limits(limits=1):
 for version in c['versions']:
  xx=X.copy();v=raw if version=='original' else np.log1p(raw) if version=='log1p_previous' else np.minimum(raw,cut);mu=float(v.mean());sd=float(v.std());xx[:,j]=(v-mu)/sd
  if version=='original':np.testing.assert_allclose(xx,X,atol=1e-13);xx=X.copy()
  model=LogisticRegression(xx,y,prior_precision=1.);anchor=fit_mode(model);hd=model.full_hessian(anchor.theta);eig,Q=np.linalg.eigh(hd);W=(Q*np.sqrt(eig))@Q.T;B=(Q/np.sqrt(eig))@Q.T;P=np.linalg.inv(hd+1e-10*np.eye(len(hd)));L=np.linalg.cholesky(P);p0=expit(xx@anchor.theta);weights=p0*(1-p0);dest=O/version;dest.mkdir(exist_ok=True);np.savez_compressed(dest/'target.npz',X=xx,y=y,anchor=anchor.theta,H=hd,W=W,B=B,P=P);cache=dict(version=version,transform_mean=mu,transform_std=sd,winsor_cutoff=cut if version=='winsor999_previous' else None,clipped_observations=int(np.sum(raw>cut)) if version=='winsor999_previous' else 0,mode_metadata=anchor.metadata,anchor_seconds=anchor.seconds,hessian_condition=float(eig[-1]/eig[0]),max_abs_previous=float(abs(xx[:,j]).max()),suspect_transformed_previous=float(xx[29182,j]),N=len(xx),d=len(hd));(dest/'metadata.json').write_text(json.dumps(cache,indent=2)+'\n');targets.append(cache)
  for ri,scale in enumerate(c['radii_in_sqrt_d']):
   for di,direction in enumerate(dirs):
    if time.perf_counter()-start>c['budget_seconds']:raise TimeoutError('StageA cap')
    delta=B@direction*(scale*np.sqrt(len(hd)));theta=anchor.theta+delta;difference=expit(xx@theta)-p0;out={}
    for method,res in [('cv1',difference),('cv2',difference-weights*(xx@delta))]:
     rr=xx*res[:,None];wh=rr@L;norm=np.linalg.norm(rr,axis=1);wn=np.linalg.norm(wh,axis=1);center=wh-wh.mean(0);sq=np.sum(center*center,axis=1);factor=len(xx)**2*(len(xx)-c['batch'])/(c['batch']*(len(xx)-1));out[method]=dict(max_raw_residual_norm=float(norm.max()),q99_raw_residual_norm=float(np.quantile(norm,.99)),q999_raw_residual_norm=float(np.quantile(norm,.999)),max_preconditioned_residual_norm=float(wn.max()),q99_preconditioned_residual_norm=float(np.quantile(wn,.99)),q999_preconditioned_residual_norm=float(np.quantile(wn,.999)),exact_preconditioned_noise_mse=float(factor*sq.mean()),suspect_noise_share=float(sq[29182]/sq.sum()),largest_noise_contributor=int(np.argmax(sq)))
     location=ri*len(dirs)+di
     if location in c['mc_locations']:
      # Same diagnostic seed/indices in all versions/methods. MC tails descriptive; exact expectation above.
      gen=np.random.default_rng(c['seed']+1+location);ix=np.array([gen.choice(len(xx),c['batch'],replace=False) for _ in range(c['mc_draws'])]);errors=len(xx)*(wh[ix].mean(1)-wh.mean(0));en=np.linalg.norm(errors,axis=1);out[method]['mc_noise_norm']=dict(median=float(np.median(en)),q99=float(np.quantile(en,.99)),max=float(en.max()),suspect_batch_hits=int(np.sum(np.any(ix==29182,axis=1))),draws=len(ix))
    rows.append(dict(version=version,radius_scale=scale,direction=di,methods=out));
   print('STAGE_A',version,'radius',scale,'locations',len(rows),flush=True)
 save('mechanism_locations.json',rows);save('targets.json',targets);screen={}
 base=[x for x in rows if x['version']=='original' and x['radius_scale']==1.];bmax=np.median([x['methods']['cv2']['max_preconditioned_residual_norm'] for x in base]);bnoise=np.median([x['methods']['cv2']['exact_preconditioned_noise_mse'] for x in base])
 for v in c['versions']:
  q=[x for x in rows if x['version']==v and x['radius_scale']==1.];mr=float(np.median([x['methods']['cv2']['max_preconditioned_residual_norm'] for x in q])/bmax);nr=float(np.median([x['methods']['cv2']['exact_preconditioned_noise_mse'] for x in q])/bnoise);screen[v]=dict(max_remainder_median_ratio_to_original=mr,noise_median_ratio_to_original=nr,eligible_stage_b=v!='original' and mr<=.5 and nr<=1)
 save('stage_a_summary.json',dict(status='complete',elapsed_seconds=time.perf_counter()-start,locations=len(rows),screen=screen,protocol_sha256=sha(H/'PROTOCOL.json')));print(json.dumps(screen),flush=True)
