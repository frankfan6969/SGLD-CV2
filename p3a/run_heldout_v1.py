"""P3A independent T4 heldout; fixed selection choices and all failures preserved."""
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
OUT=ROOT/'results/p3a_bank_t4_heldout_v1'
SELECTION=ROOT/'results/p3a_bank_t4_selection_v1'
PILOT=ROOT/'results/p3a_bank_t4_pilot_v1'
DRIVE=Path('/content/drive/MyDrive/CV2_Research/P3A_Bank_T4_Heldout_v1_2026-10-16')
SOURCE=Path('/content/drive/MyDrive/CV2_Research/P2_Bank_Reference_Refinement_v1_2026-10-15/bank_logistic')

def main():
    c=json.loads(Path('/content/P3A_HELDOUT_PROTOCOL_v1.json').read_text());lock=json.loads((ROOT/'p2_screening/lock.json').read_text())
    assert all(base.digest(ROOT/p)==h for p,h in lock['files'].items())
    dev=jax.devices('gpu')[0];assert 'T4' in dev.device_kind and jax.config.x64_enabled
    assert json.loads((SOURCE/'reference_pair.json').read_text())['reliable']
    assert base.digest(PILOT/'summary.json')==c['pilot_summary_sha256']
    assert json.loads((PILOT/'validation.json').read_text())['status']=='passed'
    assert base.digest(SELECTION/'summary.json')==c['selection_summary_sha256']
    assert base.digest(SELECTION/'SELECTED_CONFIGURATIONS.json')==c['selected_configurations_sha256']
    assert json.loads((SELECTION/'manifest.json').read_text())['status']=='complete'
    old=json.loads((PILOT/'summary.json').read_text())+json.loads((SELECTION/'summary.json').read_text());assert not any(r['seed'] in c['heldout_seeds'] for r in old)
    OUT.mkdir(parents=True,exist_ok=False);DRIVE.mkdir(parents=True,exist_ok=True);base.OUT,base.DRIVE=OUT,DRIVE
    assets=OUT/'frozen_bank_assets';assets.mkdir()
    for name in ['target.npz','scoring.npz','reference_pair.json','target_geometry.json']:shutil.copyfile(SOURCE/name,assets/name)
    shutil.copyfile('/content/P3A_HELDOUT_PROTOCOL_v1.json',OUT/'PROTOCOL.json');shutil.copyfile(__file__,OUT/'run_heldout_v1.py')
    base.save(OUT/'seed_audit.json',dict(passed=True,pilot_seeds=sorted(set(r['seed'] for r in old)),heldout_seeds=c['heldout_seeds'],heldout_consumed=True))
    base.save(OUT/'INPUT_LOCK.json',dict(files={str(p.relative_to(ROOT)):base.digest(p) for p in assets.iterdir()},original_source_lock=lock,runner_sha256=base.digest(__file__)))
    aa=np.load(assets/'target.npz');X,y,anchor,H,W=[aa[k] for k in ['X','y','anchor','H','W']];meta=json.loads((assets/'target_geometry.json').read_text());assert base.digest(assets/'target.npz')==meta['target_sha256']
    aa=np.load(assets/'scoring.npz');refs=[aa['first'],aa['second']];proj=aa['projections'];records=[];start=perf_counter()
    base.save(OUT/'manifest.json',dict(status='validating',device=dev.device_kind,jax=jax.__version__,x64=True,runs_completed=0,total_runs=c['runs'],heldout_consumed=True));base.sync()
    try:
      with threadpool_limits(limits=1):
        base.save(OUT/'validation.json',dict(status='passed',source=str(PILOT/'validation.json'),sha256=base.digest(PILOT/'validation.json'),scope='unchanged validated algebra and six batch shapes; new R512 sampling shape'))
        jobs=[dict(method=q['method'],batch=q['batch'],h=q['h'],seed=seed) for seed in c['heldout_seeds'] for q in c['candidates']]
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
                    np.savez_compressed(OUT/f"checkpoint_{m}_n{n}_h{h}_seed{seed}_K{k}.npz",endpoints=np.asarray(xx[-1]),functional_sums=np.asarray(sums),failed_at=failed)
                    points.append(q)
            failed=np.asarray(state[3]);r=dict(**job,LP=LP,replicas=c['replicas'],batch_size=n if m!='ula' else len(X),completed_iterations=int(state[4]),checkpoints=points,failed_replicas=int((failed>0).sum()),failed_at=failed.tolist(),budget_censored=censored,setup=setup,setup_seconds=setup_s,compile_seconds=compile_s,sampler_seconds=sampling,functional_seconds=functional,evaluation_seconds=evaluation)
            rid=f'{m}_n{n}_h{h}_seed{seed}';base.save(OUT/f'result_{rid}.json',r);np.savez_compressed(OUT/f'states_{rid}.npz',endpoints=np.asarray(state[0]),functional_sums=np.asarray(sums),failed_at=failed)
            records.append(r);base.save(OUT/'summary.json',records);base.save(OUT/'manifest.json',dict(status='sampling',current=job,current_iteration=int(state[4]),runs_completed=len(records),total_runs=c['runs'],heldout_consumed=True,elapsed_seconds=perf_counter()-start));base.sync()
            print('P3A_T4_HELDOUT',len(records),'/',c['runs'],rid,'K',int(state[4]),'failed',r['failed_replicas'],flush=True)
        base.save(OUT/'manifest.json',dict(status='complete',runs_completed=len(records),total_runs=c['runs'],heldout_consumed=True,elapsed_seconds=perf_counter()-start,scope='independent heldout complete; no step-size or batch retuning; selection-fixed primary comparisons'))
    except Exception:
        base.save(OUT/'manifest.json',dict(status='failed_or_budget_censored',error=traceback.format_exc(),runs_completed=len(records),total_runs=c['runs'],heldout_consumed=True));raise
    finally:
        archive=OUT/'P3A_Bank_T4_Heldout_Results.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in OUT.rglob('*'):
                if p.is_file() and p!=archive:z.write(p,p.relative_to(OUT))
        base.sync();shutil.copyfile(archive,'/content/'+archive.name)
    print('P3A_T4_HELDOUT_COMPLETE_AND_BACKED_UP',flush=True)
if __name__=='__main__':main()
