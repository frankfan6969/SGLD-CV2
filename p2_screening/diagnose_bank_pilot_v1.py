"""Preserve Bank pilot; replay four failures with original kernel and RNG streams."""
from pathlib import Path
import sys,json,hashlib,shutil,zipfile,traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'gpu_replication'),str(ROOT/'p2_screening')]
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from scipy.special import expit
import run as pilot
from batched_sampler import advance
from kernels import subset,sample_kernel
from threadpoolctl import threadpool_limits

SOURCE=ROOT/'results/p2_bank_reference_refinement_v1'
BANK=SOURCE/'bank_logistic'
OUT=ROOT/'results/p2_bank_pilot_diagnosis_v1'
DRIVE=Path('/content/drive/MyDrive/CV2_Research/P2_Bank_Pilot_Diagnosis_v1_2026-10-15')

def save(name,value):pilot.save(OUT/name,value)

def replay(record,X,y,anchor,W):
    arm=record['arm'];m,g,alpha=arm['method'],arm['geometry'],arm['alpha']
    seed,h=record['seed'],record['h'];rid=f"{g}_{m}_c{arm['coefficient']}_a{alpha}_seed{seed}"
    stored=np.load(BANK/f'states_{rid}.npz')['failed_at']
    victim=int(np.flatnonzero(stored)[0]);first=int(stored[victim]);block_start=((first-1)//128)*128
    p,_=pilot.prepare(X,y,anchor,1.,'logistic',m,g,W)
    state=pilot.ready(pilot.initial_state(p,64))
    original=sample_kernel(m,g,32,alpha)
    for _ in range(block_start//128):state,_=pilot.ready(original(p,state,seed,h))
    before=state
    original_after,_=pilot.ready(original(p,before,seed,h))
    def instrument(p,state):
        roots=jax.vmap(lambda r:jax.random.fold_in(jax.random.PRNGKey(seed),r))(jnp.arange(64,dtype=jnp.uint32))
        def body(state,_):
            keys=jax.vmap(lambda k:jax.random.fold_in(k,state[4]))(roots)
            idx=jax.vmap(lambda k:subset(jax.random.fold_in(k,1),len(X),32))(keys)
            noise=jax.vmap(lambda k:jax.random.normal(jax.random.fold_in(k,2),(X.shape[1],),dtype=jnp.float64))(keys)
            old=state[0][victim]
            new,public,info=advance(p,state,idx,noise,h,m,g,'logistic',alpha,1e-5)
            return new,(old,idx[victim],noise[victim],tuple(a[victim] for a in info),new[2][victim],new[3][victim])
        return lax.scan(body,state,None,length=128)
    after,trace=pilot.ready(jax.jit(instrument)(p,before))
    theta,indices,normal,info,live,failed=map(lambda x:jax.tree_util.tree_map(np.asarray,x),trace)
    grad,corr,rem,raw,drift,noise,radius=info
    assert np.array_equal(np.asarray(original_after[3]),stored), 'Original kernel failure replay differs'
    fail_match=np.array_equal(np.asarray(after[3]),np.asarray(original_after[3]))
    delta=float(np.max(abs(np.asarray(after[0])-np.asarray(original_after[0]))))
    rows=[]
    Hdata=np.asarray(p['H_data']);P=np.asarray(p['P']);L=np.asarray(p['L'])
    for j in range(min(128,first-block_start)):
        ix=indices[j];t=theta[j];xb=X[ix];r=expit(xb@t)-y[ix]
        exact_data=X.T@(expit(X@t)-y)
        remainder=r-(expit(xb@anchor)-y[ix])-(expit(xb@anchor)*(1-expit(xb@anchor)))*(xb@(t-anchor))
        correction=Hdata@(t-anchor);stochastic=len(X)/32*(xb.T@remainder)
        gn=np.asarray(p['gradient_hat'])+correction+stochastic+t
        dn=-h*(P@gn) if g=='full' else -h*gn
        nn=np.sqrt(2*h)*(L@normal[j]) if g=='full' else np.sqrt(2*h)*normal[j]
        centered=(t-anchor)@W;proposal=t+drift[j]+noise[j]
        rows.append(dict(iteration=block_start+j+1,replica=victim,
            radius_before=float(np.linalg.norm(centered)),proposal_radius=float(radius[j]),
            radius_threshold=float(100*np.sqrt(len(anchor))),gradient_norm=float(np.linalg.norm(grad[j])),
            exact_gradient_norm=float(np.linalg.norm(exact_data+t)),
            gradient_error_norm=float(np.linalg.norm(grad[j]-(exact_data+t))),
            correction_norm=float(np.linalg.norm(corr[j])),remainder_norm=float(np.linalg.norm(rem[j])),
            drift_whitened_norm=float(np.linalg.norm(drift[j]@W)),noise_whitened_norm=float(np.linalg.norm(noise[j]@W)),
            max_abs_batch_remainder=float(abs(remainder).max()),max_batch_row_norm=float(np.linalg.norm(xb,axis=1).max()),
            gradient_numpy_max_difference=float(abs(gn-grad[j]).max()),
            correction_numpy_max_difference=float(abs(correction-corr[j]).max()),
            remainder_numpy_max_difference=float(abs(stochastic-rem[j]).max()),
            drift_numpy_max_difference=float(abs(dn-drift[j]).max()),
            noise_numpy_max_difference=float(abs(nn-noise[j]).max()),
            proposal_finite=bool(np.isfinite(proposal).all()),live_after=bool(live[j]),
            indices=ix.tolist()))
    np.savez_compressed(OUT/f'trace_{rid}.npz',theta_before=theta,indices=indices,normal=normal,
        gradient=grad,correction=corr,remainder=rem,drift=drift,noise=noise,proposal_radius=radius)
    result=dict(configuration=rid,method=m,geometry=g,step_size=h,seed=seed,replica=victim,
        original_failure_iteration=first,original_kernel_reproduced=True,
        instrumented_failure_matches=bool(fail_match),instrumented_final_theta_max_abs_difference=delta,
        reason='standardized_radius_limit' if rows[-1]['proposal_finite'] else 'nonfinite_proposal',
        pre_failure_trace=rows,random_stream='Original replica/iteration/purpose fold_in; original R=64 and exact subset kernel',
        source_record_sha256=pilot.digest(BANK/f'result_{rid}.json'))
    save(f'diagnostic_{rid}.json',result)
    print('FAILURE_DIAGNOSTIC',rid,json.dumps({k:v for k,v in result.items() if k!='pre_failure_trace'}),flush=True)
    return result

def main():
    assert 'A100' in jax.devices('gpu')[0].device_kind
    assert Path('/content/drive/MyDrive').is_dir()
    OUT.mkdir(parents=True,exist_ok=False);DRIVE.mkdir(parents=True,exist_ok=True)
    pilot.OUT,pilot.DRIVE=OUT,DRIVE
    record=json.loads((BANK/'summary.json').read_text());assert len(record)==36
    save('source_hashes.json',{str(p.relative_to(SOURCE)):pilot.digest(p) for p in SOURCE.rglob('*') if p.is_file() and p.suffix!='.zip'})
    shutil.copyfile(__file__,OUT/Path(__file__).name)
    # Copy recorded pilot observations and frozen reference diagnostics for offline tabulation.
    for p in SOURCE.rglob('*.json'):
        dst=OUT/'pilot_source'/p.relative_to(SOURCE);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
    for p in BANK.glob('states_*.npz'):
        with np.load(p) as a:save('failure_indices/'+p.stem+'.json',a['failed_at'].tolist())
    pilot.sync()
    a=np.load(BANK/'target.npz');X,y,anchor,W=[a[k] for k in ['X','y','anchor','W']]
    results=[]
    with threadpool_limits(limits=1):
        for r in record:
            if r['failed_replicas']:
                results.append(replay(r,X,y,anchor,W));pilot.sync()
    save('diagnostic_summary.json',results)
    save('manifest.json',dict(status='complete',diagnostic_runs=len(results),
        reference_recomputed=False,sampler_tuning_performed=False,heldout_consumed=False))
    with zipfile.ZipFile(OUT/'P2_Bank_Pilot_Diagnosis.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file() and p.suffix!='.zip':z.write(p,str(p.relative_to(OUT)))
    pilot.sync()
    shutil.copyfile(OUT/'P2_Bank_Pilot_Diagnosis.zip',Path('/content/P2_Bank_Pilot_Diagnosis.zip'))
    print('DIAGNOSIS_COMPLETE_AND_BACKED_UP',DRIVE,flush=True)

if __name__=='__main__':main()
