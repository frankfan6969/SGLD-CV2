"""Frozen independent synthetic held-out; no tuning or grid extension."""
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
OUT=ROOT/'results/p2_synthetic_heldout_v1'
DRIVE=Path('/content/drive/MyDrive/CV2_Research/P2_Synthetic_Heldout_v1_2026-10-15')


def summarize(records,c):
    summaries=[];contrasts=[]
    indices=np.random.default_rng(c['bootstrap_seed']).integers(0,8,(19999,8))
    for name in c['targets']:
      for k in c['checkpoints']:
        groups={m:sorted([r for r in records if r['target']==name and r['method']==m],key=lambda r:r['seed']) for m in c['methods']}
        points={m:[next(q for q in r['checkpoints'] if q['iteration']==k) for r in rs] for m,rs in groups.items()}
        for m,pts in points.items():
            valid=len(pts)==8 and all(q['status']=='ok' for q in pts)
            row=dict(target=name,method=m,h=groups[m][0]['h'],iteration=k,valid_all_seeds=valid,
                failed_replicas=sum(q['failed_replicas'] for q in pts),
                warm_seconds=float(np.mean([q['warm_total_seconds'] for q in pts])),cold_seconds=float(np.mean([q['cold_total_seconds'] for q in pts])),
                references=[{metric:float(np.mean([q['references'][j][metric] for q in pts])) for metric in ['sw2','variance_nrmse']} for j in range(2)] if valid else None)
            row['joint_target_attained']=valid and all(q['sw2']<=c['accuracy_targets']['SW2'] and q['variance_nrmse']<=c['accuracy_targets']['variance_NRMSE'] for q in row['references'])
            summaries.append(row)
        for other in ['cv1','ula']:
          if all(q['status']=='ok' for m in ['cv2',other] for q in points[m]):
            for j in range(2):
              for metric in ['sw2','variance_nrmse']:
                delta=np.array([q['references'][j][metric] for q in points['cv2']])-np.array([q['references'][j][metric] for q in points[other]])
                boot=delta[indices].mean(1);primary=k==2048 and other=='cv1' and j==0 and metric=='variance_nrmse'
                contrasts.append(dict(target=name,iteration=k,contrast='cv2_minus_'+other,reference=j,metric=metric,primary=primary,mean_difference=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),family_adjusted_ci=np.quantile(boot,[.00625,.99375]).tolist() if primary else None))
          for cost in ['warm','cold']:
            x=np.array([q[cost+'_total_seconds'] for q in points['cv2']]);y=np.array([q[cost+'_total_seconds'] for q in points[other]])
            boot=x[indices].mean(1)/y[indices].mean(1)
            contrasts.append(dict(target=name,iteration=k,contrast='cv2_over_'+other,metric=cost+'_time_ratio',primary=False,ratio=float(x.mean()/y.mean()),ci95=np.quantile(boot,[.025,.975]).tolist()))
    for r in summaries:
      for cost in ['cold','warm']:
        for metric in ['sw2','variance_nrmse']:
            peers=[q for q in summaries if q['target']==r['target'] and q['method']==r['method'] and q['valid_all_seeds']]
            r['pareto_'+metric+'_'+cost]=r['valid_all_seeds'] and not any(q[cost+'_seconds']<=r[cost+'_seconds'] and q['references'][0][metric]<=r['references'][0][metric] and (q[cost+'_seconds']<r[cost+'_seconds'] or q['references'][0][metric]<r['references'][0][metric]) for q in peers)
    base.save(OUT/'configuration_summary.json',summaries);base.save(OUT/'paired_seed_intervals.json',dict(scope=c['inference'],contrasts=contrasts))
    attainment={name:{m:{cost:min([q for q in summaries if q['target']==name and q['method']==m and q['joint_target_attained']],key=lambda q:q[cost+'_seconds'],default=None) for cost in ['cold','warm']} for m in c['methods']} for name in c['targets']}
    base.save(OUT/'observed_target_attainment.json',dict(target=c['accuracy_targets'],results=attainment,scope='Descriptive observed attainment only; no finite speedup if comparator never reaches target'))
    lines=['# P2 synthetic independent held-out','', 'Frozen selected steps; eight new seeds; 512 independent mode-start replicas. No held-out tuning.','', '| Regime | Method | K | SW2 | Variance NRMSE | Cold s | Warm s | Failed |','| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for q in summaries:
        a=q['references'][0] if q['valid_all_seeds'] else dict(sw2=float('nan'),variance_nrmse=float('nan'))
        lines.append(f"| {q['target']} | {q['method']} | {q['iteration']} | {a['sw2']:.6f} | {a['variance_nrmse']:.6f} | {q['cold_seconds']:.6f} | {q['warm_seconds']:.6f} | {q['failed_replicas']} |")
    lines+=['','Primary comparisons: CV2 minus CV1 variance NRMSE at K2048 for each of four regimes, primary reference. Paired seed bootstrap, 19,999 resamples; 98.75% individual percentile intervals are Bonferroni-adjusted for the four prespecified contrasts. These bootstrap intervals remain approximate with eight seeds. SW2, alternative references, other budgets, ULA and timing contrasts are exploratory. All intervals condition on fixed references and measured compilation; no inner-replica resampling or population-W2 certification.','','Frontiers vary iteration budget at frozen selected steps. They are not globally optimized frontiers. All four checkpoints, including8192, remain in the results. Warm excludes compilation only; shared unscored process/method initialization is separately recorded. Reference, evaluation and Drive costs are excluded from sampler cost. No cross-hardware universal computational-cost claim.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    try:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        colors={'cv1':'#2878b5','cv2':'#d95f02','ula':'#278251'}
        for metric in ['sw2','variance_nrmse']:
          for cost in ['cold','warm']:
            fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
            for ax,name in zip(axs.flat,c['targets']):
              for m in c['methods']:
                pts=[q for q in summaries if q['target']==name and q['method']==m and q['valid_all_seeds']]
                ax.scatter([q[cost+'_seconds'] for q in pts],[q['references'][0][metric] for q in pts],label=m.upper(),color=colors[m])
                front=sorted([q for q in pts if q['pareto_'+metric+'_'+cost]],key=lambda q:q[cost+'_seconds'])
                ax.plot([q[cost+'_seconds'] for q in front],[q['references'][0][metric] for q in front],color=colors[m])
              ax.set_title(name.replace('synthetic_',''));ax.set_xscale('log');ax.set_xlabel(cost+' seconds /512 replicas');ax.set_ylabel(metric);ax.grid(alpha=.2);ax.legend(frameon=False)
            fig.suptitle('Synthetic held-out: frozen-step budget frontiers');fig.savefig(OUT/f'frontier_{metric}_{cost}.png',dpi=160);fig.savefig(OUT/f'frontier_{metric}_{cost}.pdf');plt.close(fig)
    except Exception:base.save(OUT/'plot_error.json',dict(error=traceback.format_exc()))

def main():
    c=json.loads(Path('/content/SYNTHETIC_HELDOUT_PROTOCOL_v1.json').read_text())
    assert Path('/content/drive/MyDrive').is_dir()
    assert 'A100' in base.jax.devices('gpu')[0].device_kind
    lock=json.loads((ROOT/'p2_screening/lock.json').read_text())
    assert all(base.digest(ROOT/p)==h for p,h in lock['files'].items())
    sources={name:next(p/name for p in [ROOT/'results/p2_synthetic_pilot_recovery_v1',ROOT/'results/p2_screening_v1'] if (p/name/'summary.json').exists()) for name in c['targets']}
    assert all(json.loads((s/'reference_pair.json').read_text())['reliable'] for s in sources.values())
    assert json.loads((ROOT/'results/p2_synthetic_pilot_recovery_v1/validation.json').read_text())['status']=='passed'
    selection=ROOT/'results/p2_synthetic_selection_v1'
    assert json.loads((selection/'manifest.json').read_text())['status']=='complete'
    chosen=json.loads((selection/'SELECTED_CONFIGURATIONS.json').read_text())
    assert sorted((r['target'],r['method'],r['h']) for r in chosen)==sorted((r['target'],r['method'],r['h']) for r in c['arms'])
    consumed=[str(p) for parent in [*sources.values(),selection] for p in parent.rglob('result_*.json') if json.loads(p.read_text())['seed'] in c['heldout_seeds']]
    assert not consumed
    OUT.mkdir(parents=True,exist_ok=False);DRIVE.mkdir(parents=True,exist_ok=True)
    base.OUT,base.DRIVE=OUT,DRIVE
    base.save(OUT/'PROTOCOL.json',c);shutil.copyfile(__file__,OUT/'run_heldout_v1.py')
    hashes={str(p.relative_to(ROOT)):base.digest(p) for s in sources.values() for p in s.iterdir() if p.is_file()}
    hashes.update(lock['files'])
    hashes[str((selection/'SELECTED_CONFIGURATIONS.json').relative_to(ROOT))]=base.digest(selection/'SELECTED_CONFIGURATIONS.json')
    base.save(OUT/'FREEZE_LOCK.json',dict(input_files=hashes,runner_sha256=base.digest(__file__),protocol_sha256=base.digest('/content/SYNTHETIC_HELDOUT_PROTOCOL_v1.json')))
    base.save(OUT/'seed_audit.json',dict(consumed=consumed,passed=True,heldout_consumed=False))
    # Preserve complete primary and control pilots, including all failures and gradient MSE.
    audit=[]
    for name,s in sources.items():
        audit.append(dict(target=name,source=str(s),geometry=json.loads((s/'target_geometry.json').read_text()),
            reference=json.loads((s/'reference_pair.json').read_text()),gradient_mse=json.loads((s/'gradient_mse.json').read_text()),runs=json.loads((s/'summary.json').read_text())))
    base.save(OUT/'ALL_PILOT_AUDIT.json',audit)
    base.save(OUT/'manifest.json',dict(status='preparing',runs_completed=0,total_runs=96,heldout_consumed=False));base.sync()
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
        jobs=[dict(method=a['method'],h=a['h'],seed=seed) for seed in c['heldout_seeds'] for a in c['arms'] if a['target']==name]
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
            for block in range(64):
                t=perf_counter();state,xx=base.ready(exe(p,state,seed,h));sampling+=perf_counter()-t
                t=perf_counter();sums=base.ready(sums+base.sum_functionals(xx,p['anchor'],p['transform']));functional+=perf_counter()-t;k=int(state[4]);failed=np.asarray(state[3])
                if k in c['checkpoints']:
                    t=perf_counter();q=base.score(np.asarray(xx[-1]),np.asarray(sums),k,refs,anchor,W,proj);scoring_s+=perf_counter()-t
                    failures=int(((failed>0)&(failed<=k)).sum())
                    if failures:q=dict(status='failed_ensemble',references=None)
                    q.update(iteration=k,failed_replicas=failures,warm_total_seconds=meta['anchor_seconds']+setup_s+sampling,cold_total_seconds=meta['anchor_seconds']+setup_s+sampling+compile_s)
                    points.append(q)
                    rid=f'{m}_h{h}_seed{seed}_K{k}';base.save(dest/f'checkpoint_{rid}.json',q)
                    np.savez_compressed(dest/f'states_{rid}.npz',endpoints=np.asarray(xx[-1]),functional_sums=np.asarray(sums),failed_at=failed)
            r=dict(target=name,**job,geometry='full',LP=LP,replicas=c['replicas'],batch_size=32 if m!='ula' else len(X),
                completed_iterations=int(state[4]),failed_replicas=int((failed>0).sum()),failed_at=failed.tolist(),checkpoints=points,
                setup=setup,setup_seconds=setup_s,compile_seconds=compile_s,sampler_seconds=sampling,functional_seconds=functional,scoring_seconds=scoring_s,anchor_seconds=meta['anchor_seconds'])
            rid=f'{m}_h{h}_seed{seed}';base.save(dest/f'result_{rid}.json',r)
            np.savez_compressed(dest/f'states_{rid}.npz',endpoints=np.asarray(xx[-1]),functional_sums=np.asarray(sums),failed_at=failed)
            records.append(r);base.save(OUT/'summary.json',records)
            base.save(OUT/'manifest.json',dict(status='sampling',target=name,runs_completed=len(records),total_runs=96,current_iteration=k,heldout_consumed=bool(records)));base.sync()
            print('SYNTHETIC_HELDOUT',len(records),'/96',name,m,h,seed,'failures',r['failed_replicas'],flush=True)
       summarize(records,c)
       base.save(OUT/'manifest.json',dict(status='complete',runs_completed=len(records),total_runs=96,elapsed_seconds=perf_counter()-start,heldout_consumed=True,parameters_changed=False))
    except Exception:
        base.save(OUT/'manifest.json',dict(status='failed',runs_completed=len(records),total_runs=96,error=traceback.format_exc(),heldout_consumed=bool(records)));raise
    finally:
        archive=OUT/'P2_Synthetic_Heldout_Results.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in OUT.rglob('*'):
                if p.is_file() and p!=archive:z.write(p,p.relative_to(OUT))
        base.sync();shutil.copyfile(archive,'/content/'+archive.name)
    print('SYNTHETIC_HELDOUT_COMPLETE_AND_BACKED_UP',flush=True)
if __name__=='__main__':main()
