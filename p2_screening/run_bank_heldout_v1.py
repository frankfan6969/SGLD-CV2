"""Frozen Bank held-out: fixed-step budget frontiers, original A100 backend."""
from pathlib import Path
import sys,json,time,shutil,zipfile,traceback
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'gpu_replication'),str(ROOT/'p2_screening')]
import numpy as np
import jax
import jax.numpy as jnp
from threadpoolctl import threadpool_limits
import run as pilot
from kernels import sample_kernel

SOURCE=ROOT/'results/p2_bank_reference_refinement_v1/bank_logistic'
OUT=ROOT/'results/p2_bank_heldout_v1'
DRIVE=Path('/content/drive/MyDrive/CV2_Research/P2_Bank_Heldout_v1_2026-10-15')

def summarize(records,c):
    summaries=[]
    for arm in c['arms']:
        for k in c['checkpoints']:
            group=[r for r in records if r['method']==arm['method']]
            points=[next((q for q in r['checkpoints'] if q['iteration']==k),None) for r in group]
            valid=len(points)==8 and all(q is not None and q['status']=='ok' for q in points)
            costs=[q for q in points if q is not None]
            row=dict(method=arm['method'],h=arm['h'],iteration=k,seeds_observed=len(group),
                valid_all_seeds=valid,failed_replicas=sum(sum(0<t<=k for t in r['failed_at']) for r in group),
                failure_rate=sum(sum(0<t<=k for t in r['failed_at']) for r in group)/(512*8),
                missing_checkpoint_seeds=[r['seed'] for r,q in zip(group,points) if q is None],
                warm_total_seconds=float(np.mean([q['warm_total_seconds'] for q in costs])) if costs else None,
                cold_total_seconds=float(np.mean([q['cold_total_seconds'] for q in costs])) if costs else None,
                references=[{metric:float(np.mean([q['references'][j][metric] for q in points])) for metric in ['sw2','variance_nrmse']} for j in range(2)] if valid else None)
            row['observed_joint_target_attainment']=valid and all(v['sw2']<=c['accuracy_targets']['SW2'] and v['variance_nrmse']<=c['accuracy_targets']['variance_NRMSE'] for v in row['references'])
            summaries.append(row)
    for r in summaries:
        for cost in ['warm','cold']:
            for metric in ['sw2','variance_nrmse']:
                peers=[p for p in summaries if p['method']==r['method'] and p['valid_all_seeds']]
                r[f'pareto_{metric}_{cost}']=r['valid_all_seeds'] and not any(
                    p[f'{cost}_total_seconds']<=r[f'{cost}_total_seconds'] and p['references'][0][metric]<=r['references'][0][metric] and
                    (p[f'{cost}_total_seconds']<r[f'{cost}_total_seconds'] or p['references'][0][metric]<r['references'][0][metric]) for p in peers)
    pilot.save(OUT/'configuration_summary.json',summaries)
    return summaries

def inference(records,c,summaries):
    rng=np.random.default_rng(c['bootstrap_seed']);indices=rng.integers(0,8,(c['bootstrap_replicates'],8))
    comparisons=[]
    for k in c['checkpoints']:
        for other in ['cv1','ula']:
            groups={m:sorted([r for r in records if r['method']==m],key=lambda r:r['seed']) for m in ['cv2',other]}
            if any(len(g)!=8 for g in groups.values()):continue
            points={m:[next((q for q in r['checkpoints'] if q['iteration']==k),None) for r in g] for m,g in groups.items()}
            accuracy_ok=all(q is not None and q['status']=='ok' for g in points.values() for q in g)
            for j in range(2):
                for metric in ['sw2','variance_nrmse']:
                    result=dict(iteration=k,contrast=f'cv2_minus_{other}',reference=j,metric=metric,
                        status='ok' if accuracy_ok else 'invalid_accuracy_failure_or_missing',primary=k==2048 and other=='cv1')
                    if accuracy_ok:
                        a=np.array([q['references'][j][metric] for q in points['cv2']]);b=np.array([q['references'][j][metric] for q in points[other]])
                        boot=(a-b)[indices].mean(1);result.update(mean_difference=float((a-b).mean()),ci95=np.quantile(boot,[.025,.975]).tolist())
                    comparisons.append(result)
            for cost in ['warm','cold']:
                if any(q is None for g in points.values() for q in g):continue
                a=np.array([q[f'{cost}_total_seconds'] for q in points['cv2']]);b=np.array([q[f'{cost}_total_seconds'] for q in points[other]])
                boot=a[indices].mean(1)/b[indices].mean(1)
                comparisons.append(dict(iteration=k,contrast=f'cv2_over_{other}',metric=f'{cost}_time_ratio',status='ok',
                    ratio=float(a.mean()/b.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),primary=k==2048 and other=='cv1',
                    scope='Recorded timing variability across paired seeds; attributed compilation uncertainty is not independently resampled'))
    attainment={}
    for m in ['cv1','cv2','ula']:
        points=[r for r in summaries if r['method']==m and r['observed_joint_target_attainment']]
        attainment[m]={cost:min(points,key=lambda r:r[f'{cost}_total_seconds']) if points else None for cost in ['cold','warm']}
    pilot.save(OUT/'paired_seed_intervals.json',dict(bootstrap_replicates=c['bootstrap_replicates'],
        seed=c['bootstrap_seed'],scope=c['inference'],comparisons=comparisons))
    pilot.save(OUT/'observed_target_attainment.json',dict(target=c['accuracy_targets'],methods=attainment,
        scope='Descriptive first attainable frozen budgets on both references; no finite speedup for unobserved attainment'))

def figures(summaries):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        colors={'cv1':'#2878b5','cv2':'#d95f02','ula':'#278251'}
        for metric in ['sw2','variance_nrmse']:
            fig,axes=plt.subplots(1,2,figsize=(10.5,4.5),layout='constrained')
            for ax,cost in zip(axes,['cold','warm']):
                for m in colors:
                    points=[p for p in summaries if p['method']==m and p['valid_all_seeds']]
                    ax.scatter([p[f'{cost}_total_seconds'] for p in points],[p['references'][0][metric] for p in points],c=colors[m],label=m.upper())
                    front=sorted([p for p in points if p[f'pareto_{metric}_{cost}']],key=lambda p:p[f'{cost}_total_seconds'])
                    ax.plot([p[f'{cost}_total_seconds'] for p in front],[p['references'][0][metric] for p in front],color=colors[m])
                ax.set_xscale('log');ax.set_title(cost+' setup-inclusive');ax.set_xlabel('Seconds / 512 replicas');ax.set_ylabel(metric);ax.grid(alpha=.2);ax.legend(frameon=False)
            bad=sum(not p['valid_all_seeds'] for p in summaries)
            fig.suptitle('Bank independent held-out: frozen-step budget frontier')
            fig.supxlabel(f'{bad} invalid method/budget means excluded by frozen failure rule; all failures retained in tables.',fontsize=9)
            fig.savefig(OUT/f'frontier_{metric}.png',dpi=180);fig.savefig(OUT/f'frontier_{metric}.pdf');plt.close(fig)
    except Exception:pilot.save(OUT/'plot_error.json',dict(error=traceback.format_exc()))

def main():
    c=json.loads((ROOT/'p2_screening/BANK_HELDOUT_PROTOCOL_v1.json').read_text())
    freeze=json.loads((ROOT/'p2_screening/BANK_HELDOUT_LOCK_v1.json').read_text())
    assert all(pilot.digest(ROOT/p)==h for p,h in freeze['files'].items())
    assert Path('/content/drive/MyDrive').is_dir()
    device=jax.devices('gpu')[0];assert 'A100' in device.device_kind
    assert json.loads((SOURCE/'reference_pair.json').read_text())['reliable']
    # Audit actual prior result records rather than matching protocol text mentioning reserved seeds.
    consumed=[]
    for p in (ROOT/'results').rglob('result_*.json'):
        if json.loads(p.read_text()).get('seed') in c['heldout_seeds']:consumed.append(str(p))
    assert not consumed,f'Held-out seed already present: {consumed}'
    OUT.mkdir(parents=True,exist_ok=False);DRIVE.mkdir(parents=True,exist_ok=True)
    pilot.OUT,pilot.DRIVE=OUT,DRIVE
    for name in ['BANK_HELDOUT_PROTOCOL_v1.json','BANK_HELDOUT_SELECTION_v1.json','BANK_HELDOUT_LOCK_v1.json','run_bank_heldout_v1.py']:
        shutil.copyfile(ROOT/'p2_screening'/name,OUT/name)
    pilot.save(OUT/'seed_audit.json',dict(prior_consumed_records=consumed,seeds=c['heldout_seeds'],audit_passed=True))
    jobs=[]
    for seed in c['heldout_seeds']:
        arms=list(c['arms']);np.random.default_rng(c['schedule_seed']+seed).shuffle(arms)
        jobs.extend(dict(seed=seed,**a) for a in arms)
    pilot.save(OUT/'schedule.json',jobs)
    pilot.save(OUT/'manifest.json',dict(status='preflight',device=device.device_kind,jax=jax.__version__,x64=jax.config.x64_enabled,runs_completed=0,total_runs=24))
    pilot.sync()
    a=np.load(SOURCE/'target.npz');X,y,anchor,H,W=[a[k] for k in ['X','y','anchor','H','W']]
    meta=json.loads((SOURCE/'target_geometry.json').read_text());assert pilot.digest(SOURCE/'target.npz')==meta['target_sha256']
    saved=np.load(SOURCE/'scoring.npz');refs=[saved['first'],saved['second']];proj=saved['projections']
    compiled={};records=[];started=perf_counter()
    try:
        with threadpool_limits(limits=1):
            for job in jobs:
                if perf_counter()-started>c['budgets']['stage_seconds']:raise TimeoutError('Frozen overall research budget reached')
                method,h,seed=job['method'],job['h'],job['seed']
                setup_start=perf_counter();p,setup=pilot.prepare(X,y,anchor,1.,'logistic',method,'full',W)
                state=pilot.ready(pilot.initial_state(p,c['replicas']));setup_seconds=perf_counter()-setup_start
                P,L=np.asarray(p['P']),np.asarray(p['L'])
                assert np.isfinite(P).all() and np.allclose(L@L.T,P,rtol=1e-10,atol=1e-12)
                assert np.allclose(P@(H+1e-10*np.eye(len(H))),np.eye(len(H)),atol=1e-9)
                if method not in compiled:
                    kernel=sample_kernel(method,'full',32,.99);begin=perf_counter();exe=kernel.lower(p,state,seed,h).compile()
                    compiled[method]=(exe,perf_counter()-begin)
                exe,compile_seconds=compiled[method];sums=jnp.zeros((512,2*len(H)))
                sampling=functional=scoring=0.;points=[];censored=False
                pilot.save(OUT/'manifest.json',dict(status='sampling',current_method=method,current_seed=seed,
                    runs_completed=len(records),total_runs=24,current_iteration=0));pilot.sync()
                for block in range(64):
                    if sampling>600 or perf_counter()-started>7200:censored=True;break
                    begin=perf_counter();state,xx=pilot.ready(exe(p,state,seed,h));sampling+=perf_counter()-begin
                    begin=perf_counter();sums=pilot.ready(sums+pilot.sum_functionals(xx,p['anchor'],p['transform']));functional+=perf_counter()-begin
                    k=int(state[4])
                    if k in c['checkpoints']:
                        begin=perf_counter();value=pilot.score(np.asarray(xx[-1]),np.asarray(sums),k,refs,anchor,W,proj);scoring+=perf_counter()-begin
                        failed=np.asarray(state[3]);value.update(iteration=k,failed_replicas=int((failed>0).sum()),
                            failure_iteration=int(failed[failed>0].min()) if np.any(failed>0) else None,
                            warm_total_seconds=meta['anchor_seconds']+setup_seconds+sampling,
                            cold_total_seconds=meta['anchor_seconds']+setup_seconds+sampling+compile_seconds)
                        points.append(value)
                        rid=f'{method}_h{h}_seed{seed}_K{k}'
                        np.savez_compressed(OUT/f'states_{rid}.npz',endpoints=np.asarray(xx[-1]),functional_sums=np.asarray(sums),failed_at=failed)
                        pilot.save(OUT/f'checkpoint_{rid}.json',value)
                        pilot.save(OUT/'manifest.json',dict(status='sampling',current_method=method,current_seed=seed,
                            current_iteration=k,runs_completed=len(records),total_runs=24));pilot.sync()
                failed=np.asarray(state[3]);record=dict(method=method,h=h,geometry='full',seed=seed,replicas=512,
                    batch_size=job['batch_size'],checkpoints=points,failed_at=failed.tolist(),failed_replicas=int((failed>0).sum()),
                    failure_iteration=int(failed[failed>0].min()) if np.any(failed>0) else None,
                    failed_replica_indices=np.flatnonzero(failed).tolist(),completed_iterations=int(state[4]),
                    budget_censored=censored,setup=setup,setup_seconds=setup_seconds,compile_seconds=compile_seconds,
                    anchor_seconds_attributed=meta['anchor_seconds'],sampling_seconds=sampling,
                    functional_seconds=functional,scoring_seconds=scoring)
                pilot.save(OUT/f'result_{method}_seed{seed}.json',record);records.append(record)
                pilot.save(OUT/'summary.json',records);pilot.sync()
                print('BANK_HELDOUT_RUN',len(records),'/24',method,seed,'failed',record['failed_replicas'],'sampling_seconds',sampling,flush=True)
            summaries=summarize(records,c);inference(records,c,summaries);figures(summaries)
            pilot.save(OUT/'manifest.json',dict(status='complete',runs_completed=len(records),total_runs=24,
                elapsed_seconds=perf_counter()-started,failed_runs=sum(r['failed_replicas']>0 for r in records),
                heldout_consumed=True,reference_recomputed=False,parameters_changed=False))
    except Exception:
        pilot.save(OUT/'manifest.json',dict(status='failed',runs_completed=len(records),error=traceback.format_exc(),heldout_consumed=bool(records)));raise
    finally:
        with zipfile.ZipFile(OUT/'Bank_Heldout_Results.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in OUT.rglob('*'):
                if p.is_file() and p.suffix!='.zip':z.write(p,str(p.relative_to(OUT)))
        pilot.sync()
        shutil.copyfile(OUT/'Bank_Heldout_Results.zip',Path('/content/Bank_Heldout_Results.zip'))
    print('BANK_HELDOUT_COMPLETE_AND_BACKED_UP',DRIVE,flush=True)

if __name__=='__main__':main()
