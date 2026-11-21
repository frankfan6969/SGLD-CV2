"""Frozen synthetic selection; no held-out sampling or retrospective grid extension."""
from pathlib import Path
import sys,json,time,hashlib,zipfile,shutil,traceback
from time import perf_counter
ROOT=Path('/content/p2_synthetic_v1')
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'gpu_replication'),str(ROOT/'p2_screening')]
import run as base
import numpy as np
import jax.numpy as jnp
from threadpoolctl import threadpool_limits
from kernels import sample_kernel
OUT=ROOT/'results/p2_synthetic_selection_v1'
DRIVE=Path('/content/drive/MyDrive/CV2_Research/P2_Synthetic_Selection_v1_2026-10-15')

def main():
    c=json.loads(Path('/content/SYNTHETIC_SELECTION_PROTOCOL_v1.json').read_text())
    assert Path('/content/drive/MyDrive').is_dir()
    assert 'A100' in base.jax.devices('gpu')[0].device_kind
    lock=json.loads((ROOT/'p2_screening/lock.json').read_text())
    assert all(base.digest(ROOT/p)==h for p,h in lock['files'].items())
    sources={name:next(p/name for p in [ROOT/'results/p2_synthetic_pilot_recovery_v1',ROOT/'results/p2_screening_v1'] if (p/name/'summary.json').exists()) for name in c['targets']}
    assert all(json.loads((s/'reference_pair.json').read_text())['reliable'] for s in sources.values())
    assert json.loads((ROOT/'results/p2_synthetic_pilot_recovery_v1/validation.json').read_text())['status']=='passed'
    consumed=[str(p) for s in sources.values() for p in s.glob('result_*.json') if json.loads(p.read_text())['seed'] in c['selection_seeds']]
    assert not consumed
    OUT.mkdir(parents=True,exist_ok=False);DRIVE.mkdir(parents=True,exist_ok=True)
    base.OUT,base.DRIVE=OUT,DRIVE
    base.save(OUT/'PROTOCOL.json',c);shutil.copyfile(__file__,OUT/'run_selection_v1.py')
    hashes={str(p.relative_to(ROOT)):base.digest(p) for s in sources.values() for p in s.iterdir() if p.is_file()}
    hashes.update(lock['files'])
    base.save(OUT/'FREEZE_LOCK.json',dict(input_files=hashes,runner_sha256=base.digest(__file__),protocol_sha256=base.digest('/content/SYNTHETIC_SELECTION_PROTOCOL_v1.json')))
    base.save(OUT/'seed_audit.json',dict(consumed=consumed,passed=True,heldout_consumed=False))
    # Preserve complete primary and control pilots, including all failures and gradient MSE.
    audit=[]
    for name,s in sources.items():
        audit.append(dict(target=name,source=str(s),geometry=json.loads((s/'target_geometry.json').read_text()),
            reference=json.loads((s/'reference_pair.json').read_text()),gradient_mse=json.loads((s/'gradient_mse.json').read_text()),runs=json.loads((s/'summary.json').read_text())))
    base.save(OUT/'ALL_PILOT_AUDIT.json',audit)
    base.save(OUT/'manifest.json',dict(status='preparing',runs_completed=0,total_runs=144,heldout_consumed=False));base.sync()
    records=[];start=perf_counter()
    try:
      with threadpool_limits(limits=1):
       for name,s in sources.items():
        a=np.load(s/'target.npz');X,y,anchor,H,W=[a[k] for k in ['X','y','anchor','H','W']]
        scoring=np.load(s/'scoring.npz');refs=[scoring['first'],scoring['second']];proj=scoring['projections'];meta=json.loads((s/'target_geometry.json').read_text())
        dest=OUT/name;dest.mkdir();compiled={}
        # All methods get an unscored initialization run at actual shapes before measured setup.
        begin=perf_counter()
        for m in c['methods']:
            p,_=base.prepare(X,y,anchor,1.,'logistic',m,'full',W)
            state=base.ready(base.initial_state(p,c['replicas']))
            kernel=sample_kernel(m,'full',32,.99);t=perf_counter();exe=kernel.lower(p,state,c['initialization_seed'],.05).compile();compile_s=perf_counter()-t
            base.ready(exe(p,state,c['initialization_seed'],.05))
            compiled[m]=(exe,compile_s)
        base.save(dest/'shared_initialization.json',dict(seconds=perf_counter()-begin,seed=c['initialization_seed'],scope='unscored initialization research cost; measured method compilation is separately attributed to every cold run'))
        jobs=[dict(method=m,h=h,seed=seed) for seed in c['selection_seeds'] for m in c['methods'] for h in c['step_sizes']]
        np.random.default_rng(c['schedule_seed']).shuffle(jobs);base.save(dest/'schedule.json',jobs)
        for job in jobs:
            if perf_counter()-start>c['stage_seconds']:raise TimeoutError('Frozen stage budget; no silent rerun')
            m,h,seed=job['method'],job['h'],job['seed'];t=perf_counter();p,setup=base.prepare(X,y,anchor,1.,'logistic',m,'full',W)
            state=base.ready(base.initial_state(p,c['replicas']));setup_s=perf_counter()-t
            P,L=np.asarray(p['P']),np.asarray(p['L'])
            assert np.allclose(P@(H+1e-10*np.eye(len(H))),np.eye(len(H)),atol=1e-8)
            assert np.allclose(L@L.T,P,atol=1e-10)
            LP=float(np.linalg.eigvalsh(L.T@H@L)[-1]);assert abs(LP-1)<1e-7
            exe,compile_s=compiled[m];sums=jnp.zeros((c['replicas'],2*len(H)));sampling=functional=scoring_s=0.;points=[]
            for block in range(16):
                t=perf_counter();state,xx=base.ready(exe(p,state,seed,h));sampling+=perf_counter()-t
                t=perf_counter();sums=base.ready(sums+base.sum_functionals(xx,p['anchor'],p['transform']));functional+=perf_counter()-t;k=int(state[4]);failed=np.asarray(state[3])
                if k in c['checkpoints']:
                    t=perf_counter();q=base.score(np.asarray(xx[-1]),np.asarray(sums),k,refs,anchor,W,proj);scoring_s+=perf_counter()-t
                    failures=int(((failed>0)&(failed<=k)).sum())
                    if failures:q=dict(status='failed_ensemble',references=None)
                    q.update(iteration=k,failed_replicas=failures,warm_total_seconds=meta['anchor_seconds']+setup_s+sampling,cold_total_seconds=meta['anchor_seconds']+setup_s+sampling+compile_s)
                    points.append(q)
            r=dict(target=name,**job,geometry='full',LP=LP,replicas=c['replicas'],batch_size=32 if m!='ula' else len(X),
                completed_iterations=int(state[4]),failed_replicas=int((failed>0).sum()),failed_at=failed.tolist(),checkpoints=points,
                setup=setup,setup_seconds=setup_s,compile_seconds=compile_s,sampler_seconds=sampling,functional_seconds=functional,scoring_seconds=scoring_s,anchor_seconds=meta['anchor_seconds'])
            rid=f'{m}_h{h}_seed{seed}';base.save(dest/f'result_{rid}.json',r)
            np.savez_compressed(dest/f'states_{rid}.npz',endpoints=np.asarray(xx[-1]),functional_sums=np.asarray(sums),failed_at=failed)
            records.append(r);base.save(OUT/'summary.json',records)
            base.save(OUT/'manifest.json',dict(status='sampling',target=name,runs_completed=len(records),total_runs=144,current_iteration=k,heldout_consumed=False));base.sync()
            print('SYNTHETIC_SELECTION',len(records),'/144',name,m,h,seed,'failures',r['failed_replicas'],flush=True)
       summaries=[];chosen=[]
       for name in c['targets']:
        for m in c['methods']:
         eligible=[]
         for h in c['step_sizes']:
            rows=[r for r in records if r['target']==name and r['method']==m and r['h']==h]
            valid=len(rows)==4 and all(r['failed_replicas']==0 and len(r['checkpoints'])==3 for r in rows)
            q=dict(target=name,method=m,h=h,valid_all_seeds=valid,failed_replicas=sum(r['failed_replicas'] for r in rows),
                warm_seconds=float(np.mean([r['checkpoints'][-1]['warm_total_seconds'] for r in rows])),cold_seconds=float(np.mean([r['checkpoints'][-1]['cold_total_seconds'] for r in rows])))
            if valid:
                q['accuracy']=np.mean([[z for v in r['checkpoints'][-1]['references'] for z in [v['sw2'],v['variance_nrmse']]] for r in rows],axis=0).tolist();eligible.append(q)
            summaries.append(q)
         if eligible:
            mins=np.min([q['accuracy'] for q in eligible],axis=0)
            for q in eligible:q['selection_score']=float(np.max(np.asarray(q['accuracy'])/np.maximum(mins,1e-15)))
            best=min(eligible,key=lambda q:(q['selection_score'],q['warm_seconds'],q['cold_seconds'],q['h']))
            chosen.append(best)
         else:chosen.append(dict(target=name,method=m,status='no_eligible_configuration'))
       base.save(OUT/'configuration_summary.json',summaries);base.save(OUT/'SELECTED_CONFIGURATIONS.json',chosen)
       base.save(OUT/'manifest.json',dict(status='complete',runs_completed=len(records),total_runs=144,elapsed_seconds=perf_counter()-start,heldout_consumed=False,selection_rule_changed=False))
    except Exception:
        base.save(OUT/'manifest.json',dict(status='failed',runs_completed=len(records),total_runs=144,error=traceback.format_exc(),heldout_consumed=False));raise
    finally:
        archive=OUT/'P2_Synthetic_Selection_Results.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in OUT.rglob('*'):
                if p.is_file() and p!=archive:z.write(p,p.relative_to(OUT))
        base.sync();shutil.copyfile(archive,'/content/'+archive.name)
    print('SYNTHETIC_SELECTION_COMPLETE_AND_BACKED_UP',flush=True)
if __name__=='__main__':main()
