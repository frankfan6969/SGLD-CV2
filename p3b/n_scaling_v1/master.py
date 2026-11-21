"""Autonomous P3B driver: targets/references/mechanism/optimized frontiers/report."""
from pathlib import Path
import sys,json,os,time,hashlib,traceback,subprocess,shutil,fcntl,zipfile,platform
from time import perf_counter
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R/'src'),str(R/'gpu_replication'),str(R/'p2_screening'),str(Path(__file__).parent)]
import run as base
import jax
import jax.numpy as jnp
from kernels import hmc_kernel
from sgld_cv.models import LogisticRegression
from sgld_cv.references import validate_reference_pair
from mechanism import run_mechanism
H=Path(__file__).parent;D=R/'results/p3b_n_scaling_v1';E=R/'exports/p3b_n_scaling_v1';c=json.loads((H/'PROTOCOL.json').read_text());save=base.save;start=perf_counter();guard=(D/'.master.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);assert json.loads((H/'PREFLIGHT.json').read_text())['status']=='passed';assert jax.default_backend()=='cpu' and jax.config.x64_enabled;assert all(base.digest(R/p)==v for p,v in c['source_hashes'].items());stage='data_generation';completed=[]
save(D/'environment.json',dict(python=sys.version,numpy=np.__version__,jax=jax.__version__,platform=platform.platform(),devices=[str(x) for x in jax.devices()],backend='JAX CPU float64',pid=os.getpid()))
def progress(status='running',**kw):save(D/'manifest.json',dict(status=status,stage=stage,elapsed_seconds=perf_counter()-start,pid=os.getpid(),completed=completed,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw))
def arrays(path,**kw):
 tmp=path.with_name(path.name+'.tmp')
 with tmp.open('wb') as f:np.savez_compressed(f,**kw)
 os.replace(tmp,path)
def checktime():
 if perf_counter()-start>c['total_budget_seconds']:raise TimeoutError('Frozen16h master cap')
def mode(model):
 t=perf_counter();theta=np.zeros(model.d)
 for it in range(c['mode']['max_iterations']):
  g=model.full_gradient(theta);norm=float(np.linalg.norm(g))
  if norm<c['mode']['gradient_norm_gate']:break
  step=np.linalg.solve(model.full_hessian(theta),g);old=model.potential(theta);alpha=1.
  while model.potential(theta-alpha*step)>old+1e-10:
   alpha*=.5
   if alpha<2**-30:raise RuntimeError('Mode line search failure')
  theta-=alpha*step
 else:raise RuntimeError('Mode residual gate failure')
 return theta,perf_counter()-t,dict(iterations=it,gradient_norm=norm,gate=c['mode']['gradient_norm_gate'])
def reference(A,N,index,X,y,anchor,B):
 pc=c['reference'];p=dict(X=jnp.asarray(X),y=jnp.asarray(y),anchor=jnp.asarray(anchor),B=jnp.asarray(B));pools=[];diagnostics=[]
 for pool in [0,1]:
  seed=pc['seed_base']+index*pc['seed_stride']+pool;t=perf_counter();state=(jax.random.normal(jax.random.PRNGKey(seed),(4,len(anchor)),dtype=jnp.float64)*1.5,jnp.full(4,.25),jnp.asarray(0));samples=[];accept=[];div=[]
  for block in range((pc['warmup_per_chain']+pc['retained_per_chain'])//128):
   checktime()
   if perf_counter()-t>pc['per_pool_seconds']:raise TimeoutError('Frozen per-pool1800s cap')
   state,(z,acc,dd)=base.ready(hmc_kernel(block<pc['warmup_per_chain']//128)(p,state,seed))
   if block>=pc['warmup_per_chain']//128:samples.append(np.asarray(z).transpose(1,0,2));accept.append(np.asarray(acc));div.append(np.asarray(dd))
   if (block+1)%8==0:
    arrays(A/f'reference_progress_{pool}.npz',z=np.asarray(state[0]),eps=np.asarray(state[1]),t=np.asarray(state[2]),retained_z=np.concatenate(samples,1) if samples else np.empty((4,0,len(anchor))),acceptance=np.concatenate(accept) if accept else np.empty((0,4)),divergences=np.concatenate(div) if div else np.empty((0,4)));progress(N=N,pool=pool,reference_transitions_per_chain=(block+1)*128);print('P3B_REFERENCE',N,pool,(block+1)*128,flush=True)
  theta=anchor+np.concatenate(samples,1)@B.T;arrays(A/f'reference_{pool}.npz',samples=theta);env=dict(os.environ,MPLCONFIGDIR='/private/tmp/cv2_mpl');subprocess.run([str(R/'.venv/bin/python'),str(H/'diagnose_reference.py'),str(A/f'reference_{pool}.npz'),str(A/f'reference_diagnostic_{pool}.json')],check=True,env=env);diag=json.loads((A/f'reference_diagnostic_{pool}.json').read_text());diag.update(acceptance=np.concatenate(accept).mean(0).tolist(),divergences=np.concatenate(div).sum(0).tolist(),seed=seed,seconds=perf_counter()-t,warmup=pc['warmup_per_chain'],retained=pc['retained_per_chain']);diag['reliable']=bool(diag['reliable'] and min(diag['acceptance'])>.5 and sum(diag['divergences'])==0);save(A/f'reference_{pool}.json',diag);pools.append(theta);diagnostics.append(diag)
 pair=validate_reference_pair(*pools,*diagnostics);save(A/'reference_pair.json',pair)
 if not pair['reliable']:return False
 refs=[]
 for pool,theta in enumerate(pools):
  flat=theta.reshape(-1,theta.shape[-1]);ix=np.random.default_rng(pc['scoring_seed_base']+index*1000+pool).choice(len(flat),pc['scoring_draws_per_pool'],replace=False);refs.append(flat[ix])
 rng=np.random.default_rng(pc['projection_seed']);proj=rng.normal(size=(len(anchor),pc['projections']));proj/=np.linalg.norm(proj,axis=0);arrays(A/'scoring.npz',first=refs[0],second=refs[1],projections=proj);return True
try:
 with threadpool_limits(limits=1):
  progress();dc=c['data'];Sigma=np.array(dc['Sigma']);beta=np.array(dc['beta_star']);XX=np.random.default_rng(dc['design_seed']).normal(size=(max(dc['N']),dc['d']))@np.linalg.cholesky(Sigma).T;uu=np.random.default_rng(dc['label_uniform_seed']).random(len(XX));yy=(uu<expit(XX@beta)).astype(float);arrays(D/'nested_data.npz',X=XX,y=yy,beta_star=beta,Sigma=Sigma);save(D/'DGP_AUDIT.json',dict(N_max=len(XX),d=dc['d'],data_sha256=base.digest(D/'nested_data.npz'),population_logit_sd=float(np.sqrt(beta@Sigma@beta)),nested=True,no_restandardization=True,label_rate=float(yy.mean())));progress()
  for index,N in enumerate(dc['N']):
   checktime();stage='target_mode';ND=D/f'N{N}';A=ND/'target';O=ND/'optimized';out=E/f'N{N}'
   for folder in [ND,A,O,out]:folder.mkdir(parents=True,exist_ok=True)
   try:
    if shutil.disk_usage(R).free<5*1024**3:raise RuntimeError('Less than5GiB free; preserve results and stop this target')
    progress(N=N);X,y=XX[:N].copy(),yy[:N].copy();assert np.array_equal(X,XX[:N]) and np.array_equal(y,yy[:N]);model=LogisticRegression(X,y,prior_precision=1.);anchor,anchor_s,mode_diag=mode(model);Hess=model.full_hessian(anchor);ev,Q=np.linalg.eigh(Hess+1e-10*np.eye(dc['d']));W=(Q*np.sqrt(ev))@Q.T;B=(Q/np.sqrt(ev))@Q.T;P=B@B.T;L=np.linalg.cholesky(P);LP=float(np.linalg.eigvalsh(L.T@Hess@L)[-1]);arrays(A/'target.npz',X=X,y=y,anchor=anchor,H=Hess,W=W,B=B,P=P,L=L,beta_star=beta);xnorm=np.linalg.norm(X,axis=1);metadata=dict(N=N,d=dc['d'],version='synthetic_AR1_nested',anchor_seconds=anchor_s,mode=mode_diag,LP=LP,hessian_condition=float(ev[-1]/ev[0]),posterior_whitening='symmetric sqrt(H+1e-10I)',label_rate=float(y.mean()),empirical_logit_sd=float(np.std(X@beta)),max_x_norm=float(xnorm.max()),q99_x_norm=float(np.quantile(xnorm,.99)),largest_x_observation=int(np.argmax(xnorm)),max_whitened_design_norm=float(np.linalg.norm(X@B,axis=1).max()),raw_mode_error_against_truth=float(np.linalg.norm(anchor-beta)),white_mode_error_against_truth=float(np.linalg.norm((anchor-beta)@W)));save(A/'metadata.json',metadata);stage='reference';progress(N=N)
    if not reference(A,N,index,X,y,anchor,B):
     completed.append(dict(N=N,status='reference_gate_failed'));save(ND/'STATUS.json',completed[-1]);continue
    stage='gradient_and_remainder_diagnostics';progress(N=N);run_mechanism(N)
    # Numeric grid instantiated from already frozen common coefficients and actual LP.
    pc=json.loads((R/'p3a/log1p_optimized_v1/PROTOCOL.json').read_text());pc.update(name=f'P3B synthetic N={N} optimized accuracy-cost',classification=c['classification'],LP=LP,base_h=[h/LP for h in c['optimized']['base_coefficients']],budget_seconds=c['optimized']['per_N_seconds'],per_run_sampler_seconds=c['optimized']['per_run_sampler_seconds']);pc['pilot']['seeds']=[207100+1000*index,207101+1000*index];pc['validation']['seeds']=list(range(207200+1000*index,207208+1000*index));pc['bootstrap']['seed']=207900+1000*index;pc['seeds']=dict(warmup=207001+1000*index,schedule=207003+1000*index);pc['reference']='Per-N two-pool approved reference, local curvature whitening with ridge1e-10';pc['automatic']='Per-N bounded search/validation/frontiers/report, master continues frozen N sequence';pc['input_hashes']={str(p.relative_to(R)):base.digest(p) for p in [A/n for n in ['target.npz','scoring.npz','metadata.json','reference_pair.json','reference_0.json','reference_1.json']]+[H/'PROTOCOL.json',H/'optimized_pipeline.py',H/'optimized_analyze.py',H/'optimized_plot.py',R/'p2_screening/kernels.py',R/'gpu_replication/batched_sampler.py',R/'p2_screening/run.py',R/'src/sgld_cv/estimators.py',R/'src/sgld_cv/models.py']};save(O/'PROTOCOL.json',pc);stage='optimized_search_validation';progress(N=N);env=dict(os.environ,P3B_N=str(N));log=(ND/'optimized_pipeline.log').open('w');subprocess.run([sys.executable,str(H/'optimized_pipeline.py')],check=True,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,c['total_budget_seconds']-(perf_counter()-start)));completed.append(dict(N=N,status='complete'));save(ND/'STATUS.json',completed[-1]);print('P3B_N_COMPLETE',N,flush=True)
   except Exception:
    error=traceback.format_exc();completed.append(dict(N=N,status='failed_or_budget_censored',error=error));save(ND/'STATUS.json',completed[-1]);print('P3B_N_STOPPED',N,error,flush=True)
 stage='cross_N_analysis';progress();env=dict(os.environ,MPLCONFIGDIR='/private/tmp/cv2_mpl');subprocess.run([sys.executable,str(H/'cross_analyze.py')],check=True,env=env);subprocess.run(['/Library/Frameworks/Python.framework/Versions/3.10/bin/python3',str(H/'cross_plot.py')],check=True,env=env);stage='archive';progress()
 for p in H.iterdir():
  if p.is_file() and p.suffix in ['.py','.json']:(E/p.name).write_bytes(p.read_bytes())
 # Top-level review archive includes per-N summaries, diagnostics and source; large per-N raw archives stay separate.
 files=[p for p in E.iterdir() if p.is_file() and p.suffix!='.zip'];checks={str(p.relative_to(E)):base.digest(p) for p in files};archive=E/'P3B_N_Scaling_Review_v1.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for p in files:z.write(p,p.name)
  for N in dc['N']:
   ND=D/f'N{N}'
   for p in [ND/'STATUS.json',ND/'mechanism_summary.json',ND/'mechanism_locations.json',ND/'target/metadata.json',ND/'target/reference_pair.json',ND/'target/reference_0.json',ND/'target/reference_1.json',E/f'N{N}/REPORT.md',E/f'N{N}/analysis.json']:
    if p.exists():z.write(p,f'N{N}/'+('review_' if p.is_relative_to(E) else '')+p.name)
  z.writestr('checksums.json',json.dumps(checks,indent=2))
 with zipfile.ZipFile(archive) as z:assert z.testzip() is None
 stage='complete';progress(status='complete',all_N_complete=all(x['status']=='complete' for x in completed) and len(completed)==4,archive=str(archive),archive_sha256=base.digest(archive));save(R/'p3b/STATUS.json',dict(stage='P3B complete, report and archives generated',completed=completed,report=str(E/'REPORT.md'),archive=str(archive),archive_sha256=base.digest(archive),persistence='Local only, no active Drive sync'));print('P3B_FULL_WORKFLOW_COMPLETE',archive,flush=True)
except Exception:
 error=traceback.format_exc();progress(status='failed_or_budget_censored',error=error);(E/'REPORT.md').write_text('# P3B incomplete\n\n'+error+'\n\nCompleted per-N results are retained; no missing results imputed.\n');raise
