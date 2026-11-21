from pathlib import Path
import json,hashlib
ROOT=Path('results/p3a_bank_t4_selection_v1')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
selected=json.loads((ROOT/'SELECTED_CONFIGURATIONS.json').read_text());runs=json.loads((ROOT/'summary.json').read_text())
candidates=[dict(method=q['method'],batch=q['batch'],h=q['h']) for q in selected if q.get('valid')]
assert len(candidates)==12
fixed=[]
for mode in ['variance_only','joint']:
 for threshold in [.5,.4,.3,.2]:
  for method in ['cv1','cv2','ula']:
   eligible=[]
   for q in candidates:
    if q['method']!=method:continue
    rs=[r for r in runs if (r['method'],r['batch'],r['h'])==(method,q['batch'],q['h'])]
    assert len(rs)==4 and all(r['failed_replicas']==0 and not r['budget_censored'] for r in rs)
    for k in [128,512,2048]:
     ps=[next(p for p in r['checkpoints'] if p['iteration']==k) for r in rs]
     sw=max(sum(p['references'][ref]['sw2'] for p in ps)/4 for ref in range(2))
     var=max(sum(p['references'][ref]['variance_nrmse'] for p in ps)/4 for ref in range(2))
     if var<=threshold and (mode=='variance_only' or sw<=.3):eligible.append(dict(**q,iteration=k,selection_sw2=sw,selection_variance_nrmse=var,selection_warm_seconds=sum(p['warm_total_seconds'] for p in ps)/4))
   choice=min(eligible,key=lambda q:(q['selection_warm_seconds'],q['batch'],q['h'],q['iteration'])) if eligible else None
   fixed.append(dict(mode=mode,threshold=threshold,method=method,choice=choice,status='frozen_choice' if choice else 'not_reached_in_selection; no retrospective heldout choice'))
c=dict(stage='P3A Bank T4 independent held-out freeze v1',selection_informed=True,selection_summary_sha256=sha(ROOT/'summary.json'),selected_configurations_sha256=sha(ROOT/'SELECTED_CONFIGURATIONS.json'),pilot_summary_sha256=sha(Path('results/p3a_bank_t4_pilot_v1/summary.json')),candidates=candidates,not_selected=[q for q in selected if not q.get('valid')],heldout_seeds=list(range(202800,202808)),replicas=512,checkpoints=[128,512,2048],runs=96,stage_seconds=1500,per_run_sampler_seconds=120,initialization_seed=202810,schedule_seed=202811,selection_seeds=[202700,202701,202702,202703],pilot_seeds=[202600,202601],precision='T4 JAX float64; unchanged validated kernels, uniform minibatches without replacement; mode start for every independent replica',failure='Unchanged permanent nonfinite or whitened radius>100sqrt(d). Any failed lane invalidates ensemble; no survivor-only metrics. Any failed or budget-censored seed excludes complete configuration from formal frontier/time-to-target. Retain all failures and checkpoint states. No grid extension or replacement seed.',metrics='Unchanged standardized replica SW2 and relative variance NRMSE of ergodic variance estimators; both frozen reference clouds; no threshold changes',cost='Same T4-only anchor-attributed plus actual setup plus synchronized sampler; cold adds actual measured compile. First-shape cache effects disclosed. Functional/scoring/backup costs recorded separately. No absolute time comparisons to A100 or to different replica budgets.',primary_fixed_choices=fixed,primary_comparison='Selection-fixed configurations/checkpoints for each threshold; do not replace unsuccessful held-out choice by another held-out configuration. Primary warm-time comparison; cold separately. Paired seed bootstrap 10000 draws seed202812 for uncertainty. Both references must satisfy seed-mean accuracy, no failed/censored seed. Mark unmet or unavailable targets explicitly; no finite speedup if either method not reached.',secondary='Complete held-out point cloud and descriptive batch-optimized Pareto frontier across fixed candidate/checkpoint grid; optimization on heldout explicitly exploratory, not globally optimal or unbiased speedup evidence.',controls='ULA h0.1 is independently optimized full-gradient baseline. Matched-h gradient-noise gap remains a pilot diagnostic from the full original ULA grid, not a formal heldout matched-h comparison. Selected SGLD diagnostic preserved in original pilot.',scope='Bank only; three frozen budgets through K2048, consistent with P3A pilot/selection. No K8192 extension. CV2 n32 has no eligible shortlisted configuration and remains an explicitly reported selection failure, not silently filled by retuning. Selection used R128; heldout uses R512 to improve endpoint-resolution and stability coverage.',reference='Passed frozen Bank assets unchanged; no reference recalculation',next='Report heldout outcome, failure rates and uncertainty; no new tuning after evaluation')
Path('p3a/HELDOUT_PROTOCOL_v1.json').write_text(json.dumps(c,indent=2)+'\n')
print('Frozen 12 configurations x8 seeds x512 replicas to K2048; 96 runs.')
