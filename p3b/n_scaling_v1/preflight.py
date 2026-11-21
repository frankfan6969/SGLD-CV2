from pathlib import Path
import sys,json,itertools,os,subprocess,ast,hashlib,tempfile
import numpy as np
from scipy.special import expit
from types import SimpleNamespace
R=Path(__file__).resolve().parents[2];H=Path(__file__).parent;sys.path[:0]=[str(R/'src'),str(R/'gpu_replication'),str(R/'p2_screening')]
import jax
import jax.numpy as jnp
from kernels import hmc_kernel
from batched_sampler import ready
from sgld_cv.models import LogisticRegression
from sgld_cv.estimators import prepare_estimator
from sgld_cv.costs import CostLedger
c=json.loads((H/'PROTOCOL.json').read_text());Sigma=np.array(c['data']['Sigma']);beta=np.array(c['data']['beta_star']);np.testing.assert_allclose(np.sqrt(beta@Sigma@beta),2.,atol=1e-12);rng=np.random.default_rng(206090);X=rng.normal(size=(9,3));y=(rng.random(9)<.5).astype(float);theta=rng.normal(size=3)*.2;anchor=rng.normal(size=3)*.1;model=LogisticRegression(X,y,prior_precision=1.);anc=SimpleNamespace(theta=anchor,cost=CostLedger());P=np.linalg.inv(model.full_hessian(anchor)+1e-10*np.eye(3));L=np.linalg.cholesky(P);exact=model.full_gradient(theta);p0=expit(X@anchor);delta=theta-anchor;prob=expit(X@theta);residuals=dict(sgld=prob-y,cv1=prob-p0,cv2=prob-p0-p0*(1-p0)*(X@delta));checks=[];subsets=list(itertools.combinations(range(9),2))
for m,res in residuals.items():
 est=prepare_estimator(model,m,anc);errors=np.array([est.gradient(theta,np.array(ix))-exact for ix in subsets]);vectors=X*res[:,None];center=vectors-vectors.mean(0);factor=9**2*(9-2)/(2*(9-1));raw=factor*np.mean(np.sum(center*center,1));white=factor*np.mean(np.sum((center@L)**2,1));np.testing.assert_allclose(raw,np.mean(np.sum(errors**2,1)),rtol=1e-12,atol=1e-12);np.testing.assert_allclose(white,np.mean(np.sum((errors@L)**2,1)),rtol=1e-12,atol=1e-12);checks.append(dict(method=m,exact_all36_subsets_raw_mse=raw,exact_all36_subsets_preconditioned_mse=white))
ev,Q=np.linalg.eigh(model.full_hessian(anchor)+1e-10*np.eye(3));W=(Q*np.sqrt(ev))@Q.T;B=(Q/np.sqrt(ev))@Q.T;np.testing.assert_allclose(W@B,np.eye(3),atol=1e-12);np.testing.assert_allclose(W@P@W,np.eye(3),atol=1e-12);p=dict(anchor=jnp.zeros(3));state=(jnp.zeros((4,3)),jnp.full(4,.25),jnp.array(0));_,(z,acc,div)=ready(hmc_kernel(False,True)(p,state,206099));assert np.isfinite(np.asarray(z)).all() and np.sum(np.asarray(div))==0
subprocess.run([str(R/'.venv/bin/python'),'-c','import arviz;print(arviz.__version__)'],check=True,env=dict(os.environ,MPLCONFIGDIR='/private/tmp/cv2_mpl'))
for p in H.glob('*.py'):ast.parse(p.read_text())
# Reuse existing real-data fixture solely to verify adapted optimizer-analysis code paths.
fixture=Path(json.loads((R/'p3a/log1p_optimized_v1/SMOKE_CHECK.json').read_text())['fixture']);T=Path(tempfile.mkdtemp(prefix='p3b_preflight_',dir='/private/tmp'));D=fixture/'raw';E=T/'review';E.mkdir();source=(H/'optimized_analyze.py').read_text();source=source.replace("R=Path(__file__).resolve().parents[2]",f"R=Path({str(R)!r})").replace("D=R/'results/p3b_n_scaling_v1'/f'N{N}'/'optimized';O=R/'exports/p3b_n_scaling_v1'/f'N{N}';A=R/'results/p3b_n_scaling_v1'/f'N{N}'/'target'",f"D=Path({str(D)!r});O=Path({str(E)!r});A=R/'exports/p3a_feature_sensitivity_v1/log1p_previous'");(T/'analyze.py').write_text(source);env=dict(os.environ,P3B_N='2000',MPLCONFIGDIR='/private/tmp/cv2_mpl');subprocess.run([sys.executable,str(T/'analyze.py')],check=True,env=env);a=json.loads((E/'analysis.json').read_text());q=next(q for q in a['threshold_sweep'] if q['cost']=='warm' and q['variance_threshold']==.5);assert abs(q['point_speedup_cv1_over_cv2']-.8760303967721421)<1e-10
source=(H/'optimized_plot.py').read_text().replace("R=Path(__file__).resolve().parents[2]",f"R=Path({str(R)!r})").replace("O=R/'exports/p3b_n_scaling_v1'/f'N{N}'",f"O=Path({str(E)!r})");(T/'plot.py').write_text(source);subprocess.run(['/Library/Frameworks/Python.framework/Versions/3.10/bin/python3',str(T/'plot.py')],check=True,env=env)
result=dict(status='passed',backend=str(jax.devices()),x64=jax.config.x64_enabled,finite_population_checks=checks,metric_whitening_identities='passed',Gaussian_HMC_smoke='finite and zero divergence flags',ArviZ_environment='existing .venv import passed',adapted_frontier_analysis='64 original metric reconstructions and known paired result recovered, plots generated',fixture=str(T),scope='No new P3B DGP/pilot/validation seeds consumed');(H/'PREFLIGHT.json').write_text(json.dumps(result,indent=2)+'\n');print('P3B_PREFLIGHT_PASSED')
