"""Bank reference refinement; original gates and sampler grid remain unchanged."""
from pathlib import Path
import sys, json, time, traceback, zipfile, shutil
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'gpu_replication'), str(ROOT/'p2_screening')]
import run as pilot
import numpy as np
import jax
import jax.numpy as jnp
from threadpoolctl import threadpool_limits

ORIGINAL = ROOT/'results/p2_screening_v1'
OUT = ROOT/'results/p2_bank_reference_refinement_v1'
DRIVE = Path('/content/drive/MyDrive/CV2_Research/P2_Bank_Reference_Refinement_v1_2026-10-15')
REFERENCE_SEEDS = [182010, 182011]
WARMUP, RETAINED = 1024, 8192

def reference(X, y, anchor, B, ignored_seed, deadline):
    pool = reference.pool
    reference.pool += 1
    seed = REFERENCE_SEEDS[pool]
    p = dict(X=jnp.asarray(X), y=jnp.asarray(y), anchor=jnp.asarray(anchor), B=jnp.asarray(B))
    state = (jax.random.normal(jax.random.PRNGKey(seed), (4,len(anchor)), dtype=jnp.float64)*1.5,
             jnp.full(4,.25), jnp.asarray(0))
    draws, accepted, divergences = [], [], []
    started = perf_counter()
    for block in range((WARMUP+RETAINED)//128):
        if perf_counter() > deadline:
            raise TimeoutError('Bank reference refinement exceeded original 600-second per-pool cap')
        state, (z, acc, div) = pilot.ready(pilot.hmc_kernel(block < WARMUP//128)(p,state,seed))
        if block >= WARMUP//128:
            draws.append(np.asarray(z).transpose(1,0,2))
            accepted.append(np.asarray(acc))
            divergences.append(np.asarray(div))
        if (block+1)%8 == 0:
            pilot.save(OUT/'reference_progress.json',dict(pool=pool, seed=seed,
                completed_transitions_per_chain=(block+1)*128, warmup=WARMUP,
                retained_target=RETAINED,elapsed_seconds=perf_counter()-started))
            pilot.sync()
            print('BANK_REFERENCE_PROGRESS',pool,(block+1)*128,flush=True)
    theta = anchor + np.concatenate(draws,axis=1) @ B.T
    diag = pilot.chain_diagnostics(theta)
    diag.update(acceptance=np.concatenate(accepted).mean(0).tolist(),
        divergences=np.concatenate(divergences).sum(0).tolist(),seconds=perf_counter()-started,
        kind='full-data MH HMC, unchanged fixed anchor whitening and eight leapfrog steps',
        seed=seed,warmup_per_chain=WARMUP,retained_draws_per_chain=RETAINED,
        final_step_sizes=np.asarray(state[1]).tolist())
    diag['reliable'] = bool(diag['reliable'] and min(diag['acceptance'])>.5 and sum(diag['divergences'])==0)
    print('BANK_REFERENCE_DIAGNOSTIC',pool,json.dumps({k:diag[k] for k in
          ['reliable','max_rhat','min_bulk_ess','min_tail_ess','acceptance','divergences']}),flush=True)
    return theta, diag

reference.pool = 0

def main():
    if not Path('/content/drive/MyDrive').is_dir():
        raise RuntimeError('Authorized Google Drive mount required')
    device = jax.devices('gpu')[0]
    assert 'A100' in device.device_kind
    lock = json.loads((ROOT/'p2_screening/lock.json').read_text())
    assert all(pilot.digest(ROOT/p)==h for p,h in lock['files'].items())
    assert json.loads((ORIGINAL/'validation.json').read_text())['status']=='passed'
    c = json.loads((ROOT/'p2_screening/PROTOCOL.json').read_text())
    spec = next(s for s in c['targets'] if s['id']=='bank_logistic')
    original = ORIGINAL/'bank_logistic'
    meta = json.loads((original/'target_geometry.json').read_text())
    assert pilot.digest(original/'target.npz') == meta['target_sha256']
    with np.load(original/'target.npz') as a:
        X,y = a['X'].copy(),a['y'].copy()
    OUT.mkdir(parents=True,exist_ok=False)
    DRIVE.mkdir(parents=True,exist_ok=True)
    pilot.OUT, pilot.DRIVE = OUT, DRIVE
    pilot.reference = reference
    pilot.data = lambda incoming: (X,y,dict(meta)) if incoming==spec else (_ for _ in ()).throw(ValueError('Bank only'))
    amendment = dict(stage='Bank reference refinement, then original Bank pilot only if gate passes',
        change='warmup 256 -> 1024; retained draws per chain 1024 -> 8192; new independent reference seeds',
        reference_seeds=REFERENCE_SEEDS,chains_per_pool=4,pools=2,
        reference_quality_gates_unchanged=dict(max_rhat_exclusive=1.01,min_bulk_ess=400,
            min_tail_ess=400,min_acceptance_exclusive=.5,divergences=0,
            max_mean_mcse_z_exclusive=4,covariance_relative_difference_exclusive=.2),
        original_protocol=c,source_target_sha256=meta['target_sha256'],
        source_lock_sha256=pilot.digest(ROOT/'p2_screening/lock.json'),
        runner_sha256=pilot.digest(__file__),reference_algorithm_unchanged=True,
        sampler_grid_unchanged=True,pilot_seeds=c['pilot_seeds'],heldout_consumed=False,
        total_budget_seconds=1800,per_reference_budget_seconds=600,
        no_optional_reference_selection='One prespecified longer pair; failure remains failure; no gate lowering',
        drive_destination=str(DRIVE))
    pilot.save(OUT/'PROTOCOL_AMENDMENT.json',amendment)
    shutil.copyfile(__file__,OUT/'refine_bank_reference_v1.py')
    for f in ['reference_0.json','reference_1.json','reference_pair.json','SCREENING_STATUS.json','target_geometry.json']:
        dst=OUT/'original_diagnostics'/f;dst.parent.mkdir(exist_ok=True);shutil.copyfile(original/f,dst)
    pilot.save(OUT/'manifest.json',dict(status='running',stage='reference_refinement',device=device.device_kind,
        heldout_consumed=False,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    pilot.sync()
    started=perf_counter()
    try:
        with threadpool_limits(limits=1):
            pilot.target(spec,c,started+1800)
        status=json.loads((OUT/'bank_logistic/SCREENING_STATUS.json').read_text())
        pilot.save(OUT/'manifest.json',dict(status='complete',bank_status=status,
            elapsed_seconds=perf_counter()-started,heldout_consumed=False))
    except Exception:
        pilot.save(OUT/'manifest.json',dict(status='failed',error=traceback.format_exc(),heldout_consumed=False))
        raise
    finally:
        with zipfile.ZipFile(OUT/'P2_Bank_Reference_Refinement_Results.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in OUT.rglob('*'):
                if p.is_file() and p.suffix!='.zip':z.write(p,str(p.relative_to(OUT)))
        pilot.sync()
    print('BANK_REFERENCE_REFINEMENT_AND_DRIVE_BACKUP_COMPLETE',DRIVE,flush=True)

if __name__=='__main__':
    main()
