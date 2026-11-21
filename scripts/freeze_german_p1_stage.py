"""Write a new execution lock before any amendment or selection seed is used."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from sgld_cv.protocol import sha256,verify_lock
from sgld_cv.workflow import save_json,save_csv
from report_german_p1_pilot import paired_summary,shortlist_proposal
from run_german_p1_stage import validate_execution

ROOT=Path(__file__).resolve().parents[1]
PARENT="configs/p1_german_frozen_v2.json"


def freeze(stage):
    c=json.loads((ROOT/PARENT).read_text())
    pilot=ROOT/"results/p1_german_pilot_v1"
    base_lock=json.loads((pilot/"delivery.lock.json").read_text());verify_lock(base_lock,ROOT)
    verify_lock(json.loads((ROOT/"results/p1_german_diagnostic_replay_v1/delivery.lock.json").read_text()),ROOT)
    path=ROOT/f"configs/p1_german_{stage}_v1.json"
    if path.exists():raise FileExistsError("execution configuration already frozen")
    directory=ROOT/f"results/p1_german_{stage}_configuration_v1";directory.mkdir(exist_ok=False)
    e=dict(schema_version=1,stage=stage,parent_config=PARENT,output=f"results/p1_german_{stage}_v1",
        execution_lock=str((directory/"execution.lock.json").relative_to(ROOT)),timing_order_seed=c["seeds"]["timing_order"][0],
        unchanged="target, prior, metric, caches, public h convention, reference pools, projections, functionals, failure rule, Li v0=0/epsilon unchanged",
        heldout_authorized=False)
    if stage=="amendment":
        flags=json.loads((pilot/"report_provenance.json").read_text())["boundary_warning_arms"]
        if not {"psgld","pcv1"}.issubset(flags):raise ValueError("boundary trigger absent")
        candidates=[]
        for arm in ["psgld","pcv1"]:
            old=next(p for p in c["pilot_candidates"] if p["arm"]==arm and p["alpha"]==.99 and p["coefficient"]==.256)
            new=deepcopy(old);new.update(id=f"{arm}_a0.99_c0.512",coefficient=.512,h=old["h"]*2,legacy_step_size=old["legacy_step_size"]*2)
            candidates.extend([old,new])
        e.update(candidates=candidates,seeds=c["seeds"]["pilot_amendment_reserved"],budget=dict(c["budgets"]["pilot"],
            max_stage_elapsed_seconds=1800,maximum_candidate_seed_runs=8),
            rationale="Both pilot seeds place best final variance-NRMSE at upper boundary for pSGLD/pCV1; one factor-two adjacent step per affected arm. Existing boundary rerun as paired control, not another new grid point.",
            amendment_number=1,new_candidates=2,paired_existing_controls=2,
            selection_rule="Keep original pilot points; add eligible new-step mean points from reserved seeds to frozen proxy-coverage shortlist. Reserved-seed old/new contrasts reported separately; no pooling across seed roles. No second amendment even if new boundary remains best.")
    else:
        amendment=ROOT/"results/p1_german_amendment_v1"
        m=json.loads((amendment/"manifest.json").read_text())
        if m["status"] not in ["complete","stage_budget_censored"]:raise ValueError("amendment incomplete")
        verify_lock(json.loads((amendment/"run.lock.json").read_text()),ROOT)
        ae=json.loads((ROOT/"configs/p1_german_amendment_v1.json").read_text())
        records=[json.loads(p.read_text()) for p in pilot.glob("result_*.json")]
        _,original,_=paired_summary(records,c)
        ac=deepcopy(c);ac["pilot_candidates"]=ae["candidates"];ac["seeds"]["pilot"]=ae["seeds"]
        arecords=[json.loads(p.read_text()) for p in amendment.glob("result_*.json")]
        arows,ameans,_=paired_summary(arecords,ac)
        save_csv(directory/"amendment_metrics.csv",arows);save_csv(directory/"amendment_paired_means.csv",ameans)
        added=[p for p in ae["candidates"] if p["coefficient"]==.512]
        cc=deepcopy(c);cc["pilot_candidates"]+=added
        combined=original+[p for p in ameans if p["coefficient"]==.512]
        proposal=shortlist_proposal(combined,cc);save_json(directory/"shortlist_evidence.json",proposal)
        e.update(candidates=[p["candidate"] for arm in proposal["choices"].values() for p in arm],seeds=c["seeds"]["selection"],
            budget=c["budgets"]["selection"],boundary_resolution="One allowed amendment completed/censored and recorded. No further grid expansion; original pilot failures retained. Selection/heldout cannot retune.",
            final_choice_rule="Across all four selection seeds, require full valid seed sets at each prefix; minimum measured mean setup-inclusive cost meeting each frozen marginal/joint target; tie smaller h, shorter T, candidate ID. Preserve mandatory matched cells. Repeat choices/attainment under independent reference; report disagreement. Freeze final choices before heldout.",
            inference="Selection is tuning data. No confirmatory superiority or binding-noise claim; 1999-resample confirmatory intervals belong to heldout evaluation.",
            post_selection_work="Separate overdispersed diagnostics and five independent single-chain timing repetitions for chosen settings; count in selection research cap. Heldout execution requires separate lock.")
        # Freeze completed amendment outcomes too, preventing outcome edits after selection starts.
        for p in amendment.iterdir():
            if p.is_file():base_lock["files"][str(p.relative_to(ROOT))]=sha256(p)
    validate_execution(e,c);save_json(path,e)
    doc=f"""# German P1 {stage} execution freeze

Configuration: `{path.relative_to(ROOT)}`. {len(e['candidates'])} settings,
{len(e['seeds'])} seeds, {e['budget']['replicas']} independent mode-started replicas.
Checkpoints: {e['budget']['iteration_checkpoints']}.
Stage elapsed cap: {e['budget']['max_stage_elapsed_seconds']} seconds;
candidate sampler cap: {e['budget']['max_candidate_sampler_seconds']} seconds.

{e.get('rationale',e.get('boundary_resolution',''))}

No failed replica or seed is filtered. Earlier valid checkpoints are retained;
later failed/censored ensembles remain unavailable. Original pilot and diagnostic
records are immutable. No heldout run is authorized by this configuration.

{e.get('selection_rule',e.get('final_choice_rule',''))}
"""
    (directory/"PROTOCOL.md").write_text(doc)
    files=[path,*[p for p in directory.iterdir() if p.is_file()],ROOT/"scripts/run_german_p1_stage.py",Path(__file__),ROOT/"scripts/report_german_p1_pilot.py"]
    lock=dict(schema=1,files=dict(base_lock["files"]))
    for p in files:lock["files"][str(p.relative_to(ROOT))]=sha256(p)
    save_json(directory/"execution.lock.json",lock);verify_lock(lock,ROOT)
    print(f"Frozen {stage}: {path}, {len(e['candidates'])*len(e['seeds'])} runs")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("stage",choices=["amendment","selection"])
    freeze(parser.parse_args().stage)
