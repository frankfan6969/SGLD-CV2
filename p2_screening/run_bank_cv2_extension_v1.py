"""Execute one prespecified CV2 extension; no reference or baseline reruns."""
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
OUT=ROOT/'results/p2_bank_cv2_extension_v1'
DRIVE=Path('/content/drive/MyDrive/CV2_Research/P2_Bank_CV2_Extension_v1_2026-10-15')

def terminal(r):return next((x for x in r['checkpoints'] if x['iteration']==2048),None)
def aggregate(records):
    if not records or any(r['failed_replicas'] or r['completed_iterations']!=2048 or r['budget_censored'] for r in records):
        return dict(valid=False,failed_replicas=sum(r['failed_replicas'] for r in records))
    checks=[terminal(r) for r in records]
    if any(c is None or c['status']!='ok' for c in checks):return dict(valid=False)
    return dict(valid=True,references=[{k:float(np.mean([c['references'][j][k] for c in checks]))
        for k in ['sw2','variance_nrmse']} for j in range(2)],
        warm_total_seconds=float(np.mean([c['warm_total_seconds'] for c in checks])),
        cold_total_seconds=float(np.mean([c['cold_total_seconds'] for c in checks])),failed_replicas=0)

def assess(summary,baseline,frontier):
    if not summary['valid']:return dict(eligible=False,reason='failure, incomplete run or budget censoring')
    ratios=[summary['references'][j][k]/baseline['references'][j][k]
            for j in range(2) for k in ['sw2','variance_nrmse']]
    dominated=[]
    for f in frontier:
        costs=[f['warm_total_seconds'],summary['warm_total_seconds']]
        comparisons=[(f['references'][j][k],summary['references'][j][k]) for j in range(2) for k in ['sw2','variance_nrmse']]+[tuple(costs)]
        if all(a<=b for a,b in comparisons) and any(a<b for a,b in comparisons):dominated.append(f['id'])
    return dict(eligible=max(ratios)<=1 and not dominated,max_accuracy_ratio=max(ratios),
        accuracy_ratios=ratios,dominated_by_recorded_CV1=dominated,
        warm_cost_ratio_to_CV1=summary['warm_total_seconds']/baseline['warm_total_seconds'],
        cold_cost_ratio_to_CV1=summary['cold_total_seconds']/baseline['cold_total_seconds'])

def main():
    c=json.loads((ROOT/'p2_screening/BANK_CV2_EXTENSION_PROTOCOL_v1.json').read_text())
    assert 'A100' in jax.devices('gpu')[0].device_kind
    assert Path('/content/drive/MyDrive').is_dir()
    assert json.loads((SOURCE/'reference_pair.json').read_text())['reliable']
    lock=json.loads((ROOT/'p2_screening/lock.json').read_text())
    assert all(pilot.digest(ROOT/p)==h for p,h in lock['files'].items())
    OUT.mkdir(parents=True,exist_ok=False);DRIVE.mkdir(parents=True,exist_ok=True)
    pilot.OUT,pilot.DRIVE=OUT,DRIVE
    inputs=[SOURCE/'target.npz',SOURCE/'target_geometry.json',SOURCE/'scoring.npz',SOURCE/'reference_pair.json',SOURCE/'reference_0.json',SOURCE/'reference_1.json',SOURCE/'summary.json']
    pilot.save(OUT/'source_hashes.json',{str(p.relative_to(ROOT)):pilot.digest(p) for p in inputs})
    pilot.save(OUT/'PROTOCOL_AMENDMENT.json',c)
    shutil.copyfile(__file__,OUT/Path(__file__).name)
    pilot.save(OUT/'manifest.json',dict(status='running',stage='screening',heldout_consumed=False,reference_recomputed=False))
    pilot.sync()
    data=np.load(SOURCE/'target.npz');X,y,anchor,H,W=[data[k] for k in ['X','y','anchor','H','W']]
    meta=json.loads((SOURCE/'target_geometry.json').read_text())
    assert pilot.digest(SOURCE/'target.npz')==meta['target_sha256']
    scoring_file=np.load(SOURCE/'scoring.npz');refs=[scoring_file['first'],scoring_file['second']];proj=scoring_file['projections']
    original=json.loads((SOURCE/'summary.json').read_text())
    cv1=[r for r in original if r['arm']['method']=='cv1' and r['arm']['geometry']=='full']
    baseline=aggregate([r for r in cv1 if r['h']==.05]);assert baseline['valid']
    frontier=[]
    for h in [.05,.1,.2]:
        for k in c['checkpoints']:
            records=[next(q for q in r['checkpoints'] if q['iteration']==k) for r in cv1 if r['h']==h]
            if len(records)==2 and all(q['status']=='ok' for q in records):
                frontier.append(dict(id=f'CV1_h{h}_K{k}',references=[{metric:float(np.mean([q['references'][j][metric] for q in records])) for metric in ['sw2','variance_nrmse']} for j in range(2)],warm_total_seconds=float(np.mean([q['warm_total_seconds'] for q in records]))))
    pilot.save(OUT/'fixed_comparison_baseline.json',dict(baseline=baseline,cv1_frontier=frontier,source='Original Bank pilot, no reruns'))
    kernel=sample_kernel('cv2','full',32,.99)
    compiled=None;compile_seconds=None;records=[];started=perf_counter()
    def run(h,seed,stage):
        nonlocal compiled,compile_seconds
        begin=perf_counter();p,setup=pilot.prepare(X,y,anchor,1.,'logistic','cv2','full',W)
        state=pilot.ready(pilot.initial_state(p,64));setup_seconds=perf_counter()-begin
        if compiled is None:
            begin=perf_counter();compiled=kernel.lower(p,state,seed,h).compile();compile_seconds=perf_counter()-begin
        sums=jnp.zeros((64,2*len(H)));sampling=evaluation=0.;checkpoints=[];cap=False
        for block in range(16):
            if perf_counter()-started>1200 or sampling>120:cap=True;break
            begin=perf_counter();state,xx=pilot.ready(compiled(p,state,seed,h));sampling+=perf_counter()-begin
            begin=perf_counter();sums=pilot.ready(sums+pilot.sum_functionals(xx,p['anchor'],p['transform']));evaluation+=perf_counter()-begin
            k=int(state[4])
            if k in c['checkpoints']:
                value=pilot.score(np.asarray(xx[-1]),np.asarray(sums),k,refs,anchor,W,proj)
                value.update(iteration=k,warm_total_seconds=meta['anchor_seconds']+setup_seconds+sampling,
                    cold_total_seconds=meta['anchor_seconds']+setup_seconds+sampling+compile_seconds)
                checkpoints.append(value)
        rid=f'{stage}_full_cv2_h{h}_seed{seed}'
        failed=np.asarray(state[3]);r=dict(id=rid,stage=stage,arm=dict(method='cv2',geometry='full',coefficient=h,alpha=.99),
            seed=seed,h=h,replicas=64,batch_size=32,checkpoints=checkpoints,failed_replicas=int((failed>0).sum()),
            failure_iterations=failed[failed>0].tolist(),failed_replica_indices=np.flatnonzero(failed).tolist(),
            completed_iterations=int(state[4]),setup_seconds=setup_seconds,anchor_seconds_attributed=meta['anchor_seconds'],
            sampler_seconds=sampling,functional_seconds=evaluation,compile_seconds=compile_seconds,budget_censored=cap)
        np.savez_compressed(OUT/f'states_{rid}.npz',endpoints=np.asarray(xx[-1]),functional_sums=np.asarray(sums),failed_at=failed)
        pilot.save(OUT/f'result_{rid}.json',r);records.append(r);pilot.save(OUT/'summary.json',records);pilot.sync()
        print('EXTENSION_RUN',rid,'failed',r['failed_replicas'],flush=True)
        return r
    try:
        with threadpool_limits(limits=1):
            jobs=[(h,s) for h in c['step_sizes'] for s in c['screening_seeds']]
            np.random.default_rng(187020).shuffle(jobs)
            for h,s in jobs:run(h,s,'screening')
            candidates=[]
            for h in c['step_sizes']:
                a=aggregate([r for r in records if r['stage']=='screening' and r['h']==h]);assessment=assess(a,baseline,frontier)
                candidates.append(dict(h=h,summary=a,assessment=assessment))
            eligible=[q for q in candidates if q['assessment']['eligible']]
            eligible.sort(key=lambda q:(q['assessment']['max_accuracy_ratio'],q['summary']['warm_total_seconds'],q['h']))
            selected=eligible[0] if eligible else None
            pilot.save(OUT/'candidate_selection.json',dict(candidates=candidates,selected_h=selected['h'] if selected else None,
                selection_rule=c['candidate_rule'],new_seeds_not_yet_consumed=True));pilot.sync()
            if selected:
                pilot.save(OUT/'manifest.json',dict(status='running',stage='fresh_seed_validation',selected_h=selected['h'],heldout_consumed=False));pilot.sync()
                for seed in c['validation_seeds']:run(selected['h'],seed,'validation')
                a=aggregate([r for r in records if r['stage']=='validation']);validation=assess(a,baseline,frontier)
                decision=dict(outcome='candidate_for_freeze_review' if validation['eligible'] else 'stop_validation_failed',
                    selected_h=selected['h'],validation_summary=a,validation_assessment=validation,
                    qualification='Pilot-only; zero observed failures does not establish universal stability or matched-accuracy efficiency.')
            else:decision=dict(outcome='stop_no_eligible_candidate',selected_h=None)
        pilot.save(OUT/'decision.json',decision)
        pilot.save(OUT/'manifest.json',dict(status='complete',decision=decision['outcome'],runs=len(records),
            elapsed_seconds=perf_counter()-started,heldout_consumed=False,reference_recomputed=False))
    except Exception:
        pilot.save(OUT/'manifest.json',dict(status='failed',error=traceback.format_exc(),heldout_consumed=False));raise
    finally:
        with zipfile.ZipFile(OUT/'Bank_CV2_Extension_Results.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in OUT.rglob('*'):
                if p.is_file() and p.suffix!='.zip':z.write(p,str(p.relative_to(OUT)))
        pilot.sync()
        shutil.copyfile(OUT/'Bank_CV2_Extension_Results.zip',Path('/content/Bank_CV2_Extension_Results.zip'))
    print('EXTENSION_COMPLETE_AND_BACKED_UP',json.dumps(decision),flush=True)

if __name__=='__main__':main()
