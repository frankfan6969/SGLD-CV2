"""Frozen, one-shot P4 verification. No performance-based parameter changes."""
import os
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import json, time, hashlib, platform, csv
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sgld_cv.gaussian import make_gaussian_target
from sgld_cv.anchors import supplied_anchor
from sgld_cv.estimators import prepare_estimator

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / 'PROTOCOL.json').read_text())
OUT = ROOT / 'results/p4_gaussian_sanity_v1'
EXP = ROOT / 'exports/p4_gaussian_sanity_v1'


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def symroot(a):
    e, v = np.linalg.eigh(a)
    return (v * np.sqrt(e)) @ v.T


def ot_w2(x, y):
    cost = cdist(x, y, 'sqeuclidean')
    rows, cols = linear_sum_assignment(cost)
    return float(np.sqrt(np.mean(cost[rows, cols])))


def make_kernel(method):
    def kernel(X, y, lam, anchor, residual, gd, Hd, Q, b, P, L, W, h, indices, noises):
        theta = jnp.broadcast_to(anchor, (CFG['replicas'], CFG['d']))
        def step(state, inputs):
            idx, noise = inputs
            if method == 'cv2':
                xb = X[idx]
                r = jnp.einsum('rbd,rd->rb', xb, state) - y[idx]
                delta = state - anchor
                remainder = r - residual[idx] - jnp.einsum('rbd,rd->rb', xb, delta)
                g = gd + delta @ Hd.T + (X.shape[0] / idx.shape[1]) * jnp.einsum('rbd,rb->rd', xb, remainder) + lam * state
            else:
                g = state @ Q.T - b
            nxt = state - h * (g @ P.T) + jnp.sqrt(2*h) * (noise @ L.T)
            return nxt, nxt
        _, path = jax.lax.scan(step, theta, (indices, noises))
        # Reduce full-path verification on device; retain checkpoint endpoints only.
        endpoints = jnp.concatenate([theta[None], path[jnp.array(CFG['checkpoints'][1:])-1]], axis=0)
        radius = jnp.max(jnp.linalg.norm((path-anchor) @ W, axis=-1))
        return endpoints, path, radius
    return jax.jit(kernel)


def main():
    if OUT.exists() or EXP.exists():
        raise RuntimeError('Fresh output directories required; preserve original results.')
    OUT.mkdir(parents=True); EXP.mkdir(parents=True)
    started = time.perf_counter()
    sources = [HERE/'PROTOCOL.json', HERE/'run.py', ROOT/'src/sgld_cv/gaussian.py', ROOT/'src/sgld_cv/estimators.py']
    manifest = {'status':'running', 'protocol':CFG, 'platform':platform.platform(), 'python':sys.version, 'jax':jax.__version__, 'numpy':np.__version__, 'devices':[str(x) for x in jax.devices()], 'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    dump(OUT/'manifest.json',manifest)
    kernels = {m:make_kernel(m) for m in CFG['methods']}
    gradient_rows=[]; runs=[]; scores=[]; theory=[]; controls=[]
    for kappa in CFG['kappas']:
        target=make_gaussian_target(N=CFG['N'],d=CFG['d'],kappa=kappa,seed=CFG['dataset_seed'])
        model=target.model; X=model.X; y=model.y; lam=model.prior_precision
        t=time.perf_counter(); Q=X.T@X+lam*np.eye(model.d); b=X.T@y
        mu=np.linalg.solve(Q,b); C=np.linalg.inv(Q); W=symroot(Q)
        posterior_setup=time.perf_counter()-t
        eig,V=np.linalg.eigh(Q)
        anchor=supplied_anchor(mu,provenance='Exact quadratic posterior mode via recorded dense solve')
        estimators={}; caches={}
        for method in ['sgld','cv1','cv2']:
            t=time.perf_counter(); estimators[method]=prepare_estimator(model,method,anchor); caches[method]=time.perf_counter()-t
        cv2=estimators['cv2']; rng=np.random.default_rng(CFG['gradient_seed'])
        locations=mu+rng.standard_normal((32,model.d))@np.linalg.cholesky(C).T
        shell=rng.standard_normal((3,model.d)); shell/=np.linalg.norm(shell,axis=1)[:,None]
        shell*=np.array([1.,2.,4.])[:,None]*np.sqrt(model.d)
        locations=np.concatenate([mu[None],locations,mu+np.linalg.solve(W,shell.T).T])
        idxsets=rng.integers(model.N,size=(len(locations),128,CFG['batch_size']))
        for method,est in estimators.items():
            err=[]; rel=[]
            for theta,allidx in zip(locations,idxsets):
                full=X.T@(X@theta-y)+lam*theta
                for idx in allidx:
                    g=est.gradient(theta,idx); e=g-full;err.append(e);rel.append(np.linalg.norm(e)/max(1.,np.linalg.norm(full)))
            err=np.array(err)
            for metric in CFG['metrics']:
                L=np.eye(model.d) if metric=='identity' else np.linalg.cholesky(C)
                row={'kappa':kappa,'method':method,'metric':metric,'states':len(locations),'batches_per_state':128,'raw_mse':float(np.mean(np.sum(err**2,axis=1))),'preconditioned_mse':float(np.mean(np.sum((err@L)**2,axis=1))),'max_scaled_relative_error':float(max(rel))}
                gradient_rows.append(row)
            if method=='cv2' and max(rel)>1e-10:raise AssertionError('CV2 gradient exactness gate failed')
        # Independent JAX-versus-repository pointwise CV2 formula audit.
        xb=jnp.asarray(X[idxsets[1]]); pos=jnp.asarray(locations[1]); ids=idxsets[1]
        r=jnp.einsum('kbd,d->kb',xb,pos)-jnp.asarray(y[ids])
        rem=r-jnp.asarray(cv2.residual_hat[ids])-jnp.einsum('kbd,d->kb',xb,pos-jnp.asarray(mu))
        jg=jnp.asarray(cv2.gradient_hat)+jnp.asarray(cv2.hessian_hat)@(pos-jnp.asarray(mu))+(model.N/CFG['batch_size'])*jnp.einsum('kbd,kb->kd',xb,rem)+lam*pos
        expected=np.array([cv2.gradient(locations[1],idx) for idx in ids]); discrepancy=float(np.max(np.abs(np.asarray(jg)-expected)))
        controls.append({'kappa':kappa,'jax_numpy_max_gradient_abs':discrepancy,'covariance_condition':float(np.linalg.cond(C))})
        if not np.allclose(np.asarray(jg),expected,rtol=1e-10,atol=1e-10):raise AssertionError('Independent JAX gradient gate failed')
        rr=np.random.default_rng(CFG['reference_seed'])
        reference=mu+rr.standard_normal((CFG['replicas'],model.d))@np.linalg.cholesky(C).T
        reference2=mu+rr.standard_normal((CFG['replicas'],model.d))@np.linalg.cholesky(C).T
        baseline={'raw':ot_w2(reference,reference2),'whitened':ot_w2((reference-mu)@W,(reference2-mu)@W)}
        np.savez_compressed(OUT/f'target_k{kappa}.npz',X=X,y=y,Q=Q,mean=mu,covariance=C,reference=reference,reference2=reference2)
        for metric in CFG['metrics']:
            t=time.perf_counter()
            P=np.eye(model.d) if metric=='identity' else np.linalg.inv(Q)
            L=np.linalg.cholesky(P); LP=float(np.linalg.eigvalsh(symroot(P)@Q@symroot(P))[-1]); h=CFG['step_coefficient']/LP
            metric_setup=time.perf_counter()-t
            p=np.ones(model.d) if metric=='identity' else 1/eig
            a=1-h*p*eig
            if not np.all(np.abs(a)<1):raise AssertionError('Declared step not stable')
            for K in CFG['checkpoints']:
                variance=2*h*p*(1-a**(2*K))/(1-a*a)
                theory.append({'kappa':kappa,'metric':metric,'K':K,'h':h,'LP':LP,'population_w2':float(np.linalg.norm(np.sqrt(variance)-1/np.sqrt(eig))),'population_whitened_w2':float(np.linalg.norm(np.sqrt(variance*eig)-1)),'population_whitened_cov_error':float(np.linalg.norm(variance*eig-1)/np.sqrt(model.d)),'stationary_w2':float(np.linalg.norm(np.sqrt(2*h*p/(1-a*a))-1/np.sqrt(eig)))})
            for seed in CFG['sampler_seeds']:
                t=time.perf_counter(); brng=np.random.default_rng(np.random.SeedSequence([seed,1])); nrng=np.random.default_rng(np.random.SeedSequence([seed,2]))
                indices=brng.integers(model.N,size=(max(CFG['checkpoints']),CFG['replicas'],CFG['batch_size']),dtype=np.int32)
                noises=nrng.standard_normal((max(CFG['checkpoints']),CFG['replicas'],model.d))
                # Inputs are intentionally generated independently of method and metric.
                input_seconds=time.perf_counter()-t
                args=tuple(jnp.asarray(v) for v in [X,y,lam,mu,cv2.residual_hat,cv2.gradient_hat,cv2.hessian_hat,Q,b,P,L,W,h,indices,noises])
                jax.block_until_ready(args)
                paired={}
                # Reverse timing order by seed to avoid a single fixed method ordering.
                methods=CFG['methods'] if seed%2==0 else list(reversed(CFG['methods']))
                for method in methods:
                    t=time.perf_counter(); executable=kernels[method].lower(*args).compile(); compile_seconds=time.perf_counter()-t
                    t=time.perf_counter(); end,path,radius=executable(*args);jax.block_until_ready((end,path,radius));sample_seconds=time.perf_counter()-t
                    end=np.asarray(end);radius=float(radius);paired[method]=(end,path)
                    failed=bool(not np.isfinite(radius) or radius>100*np.sqrt(model.d))
                    row={'kappa':kappa,'metric':metric,'seed':seed,'method':method,'h':h,'LP':LP,'posterior_setup_s':posterior_setup,'metric_setup_s':metric_setup,'cv_cache_setup_s':caches['cv2'] if method=='cv2' else 0.,'input_rng_s':input_seconds,'compile_s':compile_seconds,'kernel_s':sample_seconds,'max_whitened_radius':radius,'failed':failed}
                    row['warm_total_s']=sum(row[x] for x in ['posterior_setup_s','metric_setup_s','cv_cache_setup_s','input_rng_s','kernel_s']);row['cold_total_s']=row['warm_total_s']+compile_seconds;runs.append(row)
                    if failed:raise AssertionError('P4 failure; preserve and diagnose')
                    tscore=time.perf_counter()
                    for z,K in enumerate(CFG['checkpoints']):
                        white=(end[z]-mu)@W; cov=np.cov(white,rowvar=False,ddof=1)
                        v=2*h*p*(1-a**(2*K))/(1-a*a); finite=(V*(v*eig))@V.T
                        law_cov_error=float(np.linalg.norm(cov-finite,'fro')/np.sqrt(model.d))
                        if K and method=='ula' and law_cov_error>.5:raise AssertionError('Gross finite-step covariance law gate failed')
                        scores.append({'kappa':kappa,'metric':metric,'seed':seed,'method':method,'K':K,'covariance_error_whitened':float(np.linalg.norm(cov-np.eye(model.d),'fro')/np.sqrt(model.d)),'covariance_error_to_finite_law':law_cov_error,'w2_empirical':ot_w2(end[z],reference),'w2_empirical_whitened':ot_w2(white,(reference-mu)@W),'iid_floor':baseline['raw'],'iid_floor_whitened':baseline['whitened']})
                    row['scoring_s']=time.perf_counter()-tscore
                # Compare every transition, not just endpoint archives.
                delta=(paired['cv2'][1]-paired['ula'][1])@jnp.asarray(W)
                maxdiff=float(jnp.max(jnp.linalg.norm(delta,axis=-1)))
                if maxdiff>1e-8:raise AssertionError('Paired full-path exactness failed')
                for row in runs[-2:]:row['paired_max_whitened_path_difference']=maxdiff
                dump(OUT/'partial_results.json',{'gradient_mse':gradient_rows,'pointwise_controls':controls,'runs':runs,'scores':scores,'analytic':theory})
                np.savez_compressed(OUT/f'k{kappa}_{metric}_seed{seed}.npz',cv2=paired['cv2'][0],ula=paired['ula'][0],checkpoints=CFG['checkpoints'])
                dump(OUT/'progress.json',{'kappa':kappa,'metric':metric,'seed':seed,'completed_runs':len(runs),'planned_runs':32,'elapsed_s':time.perf_counter()-started})
                print(f'P4 kappa={kappa} metric={metric} seed={seed}: {len(runs)}/32 runs, paired discrepancy={maxdiff:.3g}',flush=True)
    result={'scope':CFG['classification'],'gradient_mse':gradient_rows,'pointwise_controls':controls,'runs':runs,'scores':scores,'analytic':theory,'elapsed_s':time.perf_counter()-started,'gates_pass':True}
    dump(EXP/'analysis.json',result)
    for name,rows in [('gradient_mse',gradient_rows),('runtime',runs),('endpoint_metrics',scores),('analytic_gaussian',theory)]:
        with (EXP/f'{name}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    manifest.update(status='complete',elapsed_s=result['elapsed_s'],gates_pass=True,run_count=len(runs),failed_runs=sum(r['failed'] for r in runs))
    dump(OUT/'manifest.json',manifest)


if __name__=='__main__':
    main()
