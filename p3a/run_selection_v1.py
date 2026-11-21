"""P3A independent T4 selection; frozen pilot-informed shortlist, untouched held-out seeds."""
from pathlib import Path
import sys,json,time,shutil,zipfile,traceback,hashlib
from time import perf_counter
ROOT=Path('/content/p3a_t4_v1');sys.path[:0]=[str(ROOT/'src'),str(ROOT/'gpu_replication'),str(ROOT/'p2_screening')]
import run as base
import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import expit
from kernels import sample_kernel,subset
from batched_sampler import gradient_parts,advance
from threadpoolctl import threadpool_limits
OUT=ROOT/'results/p3a_bank_t4_selection_v1'
PILOT=ROOT/'results/p3a_bank_t4_pilot_v1'
DRIVE=Path('/content/drive/MyDrive/CV2_Research/P3A_Bank_T4_Selection_v1_2026-10-16')
SOURCE=Path('/content/drive/MyDrive/CV2_Research/P2_Bank_Reference_Refinement_v1_2026-10-15/bank_logistic')

def checks(X,y,anchor,W,H,c,refs):
    rng=np.random.default_rng(202501);theta=refs[0][:2];records=[]
    for n in c['batches']:
        ix=np.array([rng.choice(len(X),n,replace=False) for _ in range(2)])
        keys=jax.random.split(jax.random.PRNGKey(202501+n),16)
        draws=np.asarray(base.ready(jax.jit(jax.vmap(lambda k:subset(k,len(X),n)))(keys)))
        assert all(len(set(a))==n for a in draws) and draws.min()>=0 and draws.max()<len(X)
        for m in ['sgld','cv1','cv2','ula']:
            p,_=base.prepare(X,y,anchor,1.,'logistic',m,'full',W)
            got=np.asarray(base.ready(gradient_parts(p,jnp.asarray(theta),jnp.asarray(ix),m,'logistic')[0]))
            residual=expit(np.einsum('rbd,rd->rb',X[ix],theta))-y[ix]
            if m=='ula':want=(expit(theta@X.T)-y)@X+theta
            elif m=='sgld':want=len(X)/n*np.einsum('rbd,rb->rd',X[ix],residual)+theta
            else:
                ph=expit(X@anchor);dh=(ph-y)@X;rem=residual-(ph-y)[ix];cor=0
                if m=='cv2':
                    Hdata=(X.T*(ph*(1-ph)))@X;delta=theta-anchor;rem-= (ph*(1-ph))[ix]*np.einsum('rbd,rd->rb',X[ix],delta);cor=delta@Hdata.T
                want=dh+cor+len(X)/n*np.einsum('rbd,rb->rd',X[ix],rem)+theta
            np.testing.assert_allclose(got,want,rtol=1e-9,atol=1e-8)
            # Fixed-input one-step parity checks drift, noise geometry and failure bookkeeping.
            normals=rng.normal(size=theta.shape);state=list(base.initial_state(p,2));state[0]=jnp.asarray(theta)
            new,public,_=base.ready(advance(p,tuple(state),jnp.asarray(ix),jnp.asarray(normals),.0001,m,'full','logistic',.99,1e-5))
            expected=theta-.0001*(want@np.asarray(p['P']).T)+np.sqrt(.0002)*(normals@np.asarray(p['L']).T)
            np.testing.assert_allclose(np.asarray(public),expected,rtol=1e-9,atol=1e-8)
            records.append(dict(method=m,batch=n,max_gradient_difference=float(np.max(abs(got-want)))))
    return dict(status='passed',x64=jax.config.x64_enabled,checks=records,scope='unchanged algebra; six exact-uniform Floyd batch shapes; fixed-input CPU/GPU gradient and one-step parity')

def mse(X,y,anchor,W,refs,c):
    rng=np.random.default_rng(202502);states=refs[0][rng.choice(len(refs[0]),24,replace=False)]
    exact=(expit(states@X.T)-y)@X+states;inv=np.linalg.inv(W);rows=[]
    for n in c['batches']:
        ix=np.array([[rng.choice(len(X),n,replace=False) for _ in states] for _ in range(32)])
        for m in ['sgld','cv1','cv2']:
            p,_=base.prepare(X,y,anchor,1.,'logistic',m,'full',W)
            fn=jax.jit(jax.vmap(lambda idx:gradient_parts(p,jnp.asarray(states),idx,m,'logistic')[0]))
            e=np.asarray(base.ready(fn(jnp.asarray(ix))))-exact;raw=np.sum(e*e,axis=-1);white=np.sum((e@inv)**2,axis=-1)
            rows.append(dict(method=m,batch=n,locations=24,paired_batches=32,raw_mse_mean=float(raw.mean()),raw_mse_median=float(np.median(raw)),whitened_mse_mean=float(white.mean()),finite_population_factor=(len(X)-n)/(len(X)-1)))
        base.save(OUT/'gradient_mse.json',rows);base.sync()
    return rows

def main():
    c=json.loads(Path('/content/P3A_SELECTION_PROTOCOL_v1.json').read_text());lock=json.loads((ROOT/'p2_screening/lock.json').read_text())
    assert all(base.digest(ROOT/p)==h for p,h in lock['files'].items())
    dev=jax.devices('gpu')[0];assert 'T4' in dev.device_kind and jax.config.x64_enabled
    assert json.loads((SOURCE/'reference_pair.json').read_text())['reliable']
    assert base.digest(PILOT/'summary.json')==c['pilot_summary_sha256']
    assert json.loads((PILOT/'validation.json').read_text())['status']=='passed'
    old=json.loads((PILOT/'summary.json').read_text());assert not any(r['seed'] in c['selection_seeds'] for r in old)
    OUT.mkdir(parents=True,exist_ok=False);DRIVE.mkdir(parents=True,exist_ok=True);base.OUT,base.DRIVE=OUT,DRIVE
    assets=OUT/'frozen_bank_assets';assets.mkdir()
    for name in ['target.npz','scoring.npz','reference_pair.json','target_geometry.json']:shutil.copyfile(SOURCE/name,assets/name)
    shutil.copyfile('/content/P3A_SELECTION_PROTOCOL_v1.json',OUT/'PROTOCOL.json');shutil.copyfile(__file__,OUT/'run_selection_v1.py')
    base.save(OUT/'seed_audit.json',dict(passed=True,pilot_seeds=sorted(set(r['seed'] for r in old)),selection_seeds=c['selection_seeds'],heldout_consumed=False))
    base.save(OUT/'INPUT_LOCK.json',dict(files={str(p.relative_to(ROOT)):base.digest(p) for p in assets.iterdir()},original_source_lock=lock,runner_sha256=base.digest(__file__)))
    aa=np.load(assets/'target.npz');X,y,anchor,H,W=[aa[k] for k in ['X','y','anchor','H','W']];meta=json.loads((assets/'target_geometry.json').read_text());assert base.digest(assets/'target.npz')==meta['target_sha256']
    aa=np.load(assets/'scoring.npz');refs=[aa['first'],aa['second']];proj=aa['projections'];records=[];start=perf_counter()
    base.save(OUT/'manifest.json',dict(status='validating',device=dev.device_kind,jax=jax.__version__,x64=True,runs_completed=0,total_runs=c['runs'],heldout_consumed=False));base.sync()
    try:
      with threadpool_limits(limits=1):
        base.save(OUT/'validation.json',dict(status='passed',source=str(PILOT/'validation.json'),sha256=base.digest(PILOT/'validation.json'),scope='unchanged validated algebra and six batch shapes; new R128 sampling shape'))
        jobs=[dict(method=q['method'],batch=q['batch'],h=q['h'],seed=seed) for seed in c['selection_seeds'] for q in c['candidates']]
        assert len(jobs)==c['runs'];np.random.default_rng(c['schedule_seed']).shuffle(jobs);base.save(OUT/'schedule.json',jobs);compiled={}
        for job in jobs:
            if perf_counter()-start>c['stage_seconds']:raise TimeoutError('Frozen stage cap; retained partial results, no silent rerun')
            m,n,h,seed=job['method'],job['batch'],job['h'],job['seed'];key=(m,n)
            if key not in compiled:
                t=perf_counter();p,_=base.prepare(X,y,anchor,1.,'logistic',m,'full',W);state=base.ready(base.initial_state(p,c['replicas']));kernel=sample_kernel(m,'full',n,.99)
                begin=perf_counter();exe=kernel.lower(p,state,c['initialization_seed'],h).compile();compile_s=perf_counter()-begin
                begin=perf_counter();base.ready(exe(p,state,c['initialization_seed'],h));micro=perf_counter()-begin
                compiled[key]=(exe,compile_s)
                base.save(OUT/f'initialization_{m}_n{n}.json',dict(compile_seconds=compile_s,sampler_128_seconds=micro,total_research_seconds=perf_counter()-t,batch=n,method=m,scope='unscored first-use initialization; compile attributed to cold; padded measured cost used, no imposed affine cost model'))
            t=perf_counter();p,setup=base.prepare(X,y,anchor,1.,'logistic',m,'full',W);state=base.ready(base.initial_state(p,c['replicas']));setup_s=perf_counter()-t
            L=np.asarray(p['L']);LP=float(np.linalg.eigvalsh(L.T@H@L)[-1]);assert abs(LP-1)<1e-7
            sums=jnp.zeros((c['replicas'],2*len(H)));sampling=functional=evaluation=0.;points=[];exe,compile_s=compiled[key];censored=False
            for block in range(16):
                if sampling>c['per_run_sampler_seconds'] or perf_counter()-start>c['stage_seconds']:censored=True;break
                t=perf_counter();state,xx=base.ready(exe(p,state,seed,h));sampling+=perf_counter()-t
                t=perf_counter();sums=base.ready(sums+base.sum_functionals(xx,p['anchor'],p['transform']));functional+=perf_counter()-t;k=int(state[4]);failed=np.asarray(state[3])
                if k in c['checkpoints']:
                    t=perf_counter();q=base.score(np.asarray(xx[-1]),np.asarray(sums),k,refs,anchor,W,proj);evaluation+=perf_counter()-t
                    count=int((failed>0).sum())
                    if count:q=dict(status='failed_ensemble',references=None)
                    q.update(iteration=k,failed_replicas=count,failure_iteration=int(failed[failed>0].min()) if count else None,warm_total_seconds=meta['anchor_seconds']+setup_s+sampling,cold_total_seconds=meta['anchor_seconds']+setup_s+sampling+compile_s)
                    points.append(q)
            failed=np.asarray(state[3]);r=dict(**job,LP=LP,replicas=c['replicas'],batch_size=n if m!='ula' else len(X),completed_iterations=int(state[4]),checkpoints=points,failed_replicas=int((failed>0).sum()),failed_at=failed.tolist(),budget_censored=censored,setup=setup,setup_seconds=setup_s,compile_seconds=compile_s,sampler_seconds=sampling,functional_seconds=functional,evaluation_seconds=evaluation)
            rid=f'{m}_n{n}_h{h}_seed{seed}';base.save(OUT/f'result_{rid}.json',r);np.savez_compressed(OUT/f'states_{rid}.npz',endpoints=np.asarray(state[0]),functional_sums=np.asarray(sums),failed_at=failed)
            records.append(r);base.save(OUT/'summary.json',records);base.save(OUT/'manifest.json',dict(status='sampling',current=job,current_iteration=int(state[4]),runs_completed=len(records),total_runs=c['runs'],heldout_consumed=False,elapsed_seconds=perf_counter()-start));base.sync()
            print('P3A_T4_SELECTION',len(records),'/',c['runs'],rid,'K',int(state[4]),'failed',r['failed_replicas'],flush=True)
        candidates=[];chosen=[]
        for m,n in sorted(set((q['method'],q['batch']) for q in c['candidates'])):
            eligible=[]
            for h in sorted(set(q['h'] for q in c['candidates'] if (q['method'],q['batch'])==(m,n))):
                rs=[r for r in records if (r['method'],r['batch'],r['h'])==(m,n,h)]
                valid=len(rs)==4 and all(not r['budget_censored'] and r['failed_replicas']==0 and r['completed_iterations']==2048 for r in rs)
                q=dict(method=m,batch=n,h=h,valid=valid,failed_replicas=sum(r['failed_replicas'] for r in rs))
                if valid:
                    vv=np.array([[[v[t] for t in ['sw2','variance_nrmse']] for v in r['checkpoints'][-1]['references']] for r in rs])
                    q.update(sw2=float(vv.mean(0).max(0)[0]),variance_nrmse=float(vv.mean(0).max(0)[1]),warm_seconds=float(np.mean([r['checkpoints'][-1]['warm_total_seconds'] for r in rs])))
                    eligible.append(q)
                candidates.append(q)
            if eligible:
                mins=np.min([[q['sw2'],q['variance_nrmse']] for q in eligible],axis=0)
                for q in eligible:q['selection_score']=float(np.max(np.array([q['sw2'],q['variance_nrmse']])/np.maximum(mins,1e-15)))
                chosen.append(min(eligible,key=lambda q:(q['selection_score'],q['warm_seconds'],q['h'])))
            else:chosen.append(dict(method=m,batch=n,status='no_eligible_configuration'))
        base.save(OUT/'configuration_summary.json',candidates);base.save(OUT/'SELECTED_CONFIGURATIONS.json',chosen)
        base.save(OUT/'manifest.json',dict(status='complete',runs_completed=len(records),total_runs=c['runs'],heldout_consumed=False,elapsed_seconds=perf_counter()-start,scope='independent selection complete; heldout remains unused and needs a separate freeze'))
    except Exception:
        base.save(OUT/'manifest.json',dict(status='failed_or_budget_censored',error=traceback.format_exc(),runs_completed=len(records),total_runs=c['runs'],heldout_consumed=False));raise
    finally:
        archive=OUT/'P3A_Bank_T4_Selection_Results.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in OUT.rglob('*'):
                if p.is_file() and p!=archive:z.write(p,p.relative_to(OUT))
        base.sync();shutil.copyfile(archive,'/content/'+archive.name)
    print('P3A_T4_SELECTION_COMPLETE_AND_BACKED_UP',flush=True)
if __name__=='__main__':main()
