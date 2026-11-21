"""Immutable heldout execution/inference freeze after completed Selection."""
import json
from pathlib import Path
from sgld_cv.protocol import verify_lock,sha256
from sgld_cv.workflow import save_json
from run_german_p1_heldout import validate_execution

ROOT=Path(__file__).resolve().parents[1]


def main():
    c=json.loads((ROOT/"configs/p1_german_frozen_v2.json").read_text())
    selection=json.loads((ROOT/"configs/p1_german_selection_v1.json").read_text())
    choices=json.loads((ROOT/"results/p1_german_selection_v1/final_choices.json").read_text())
    locks=[ROOT/"results/p1_german_selection_v1/final_choices.lock.json",
           ROOT/"results/p1_german_selection_diagnostics_v1/delivery.lock.json",
           ROOT/"results/p1_german_selection_configuration_v1/execution.lock.json"]
    files={}
    for path in locks:
        lock=json.loads(path.read_text());verify_lock(lock,ROOT);files.update(lock["files"])
    diag=json.loads((ROOT/"results/p1_german_selection_diagnostics_v1/manifest.json").read_text())
    if diag["status"]!="complete":raise ValueError("Selection diagnostics unfinished")
    ids={p["choice"]["candidate_id"] for p in choices["choices"] if p["choice"]}|{p["candidate_id"] for p in choices["matched_cells"]}
    folder=ROOT/"results/p1_german_heldout_configuration_v1";folder.mkdir(exist_ok=False)
    path=ROOT/"configs/p1_german_heldout_v1.json"
    if path.exists():raise ValueError("heldout configuration exists")
    e=dict(schema_version=1,stage="heldout",parent_config="configs/p1_german_frozen_v2.json",
        candidates=[p for p in selection["candidates"] if p["id"] in ids],seeds=c["seeds"]["heldout"],
        budget=c["budgets"]["heldout"],timing_order_seed=c["seeds"]["timing_order"][0],
        output="results/p1_german_heldout_v1",execution_lock="results/p1_german_heldout_configuration_v1/execution.lock.json",
        immutable_choices="results/p1_german_selection_v1/final_choices.json",
        scope="Only selected target-specific candidate/prefix pairs and mandatory matched cells are scored for conclusions; extra trajectory prefixes cannot be reoptimized.",
        inference=dict(bootstrap_repeats=1999,paired_seed_clusters_then_whole_replicas=True,
            matched_noise_level=.99375,other_intervals=.95,reference_rows_resampled=False,
            primary_efficiency_target=dict(sw2=c["accuracy"]["sw2_accuracy_targets"][0],variance_NRMSE=.1),
            primary_efficiency_rule="Both frozen configurations have valid full eight-seed sets, meet both targets under both references, accuracy percentile upper bounds meet targets, and paired CV2/CV1 cost-ratio upper bound<=.9. Mean attainment and cost savings also reported separately; other targets pointwise exploratory.",
            bootstrap_limit="Approximate conditional finite-cloud percentile intervals, not guaranteed population-W2 coverage; SW2 resampling bias retained and reported, not corrected post hoc.",
            budget="Sampling+inference active elapsed <=43200s; 1999 draws required per reported interval; incomplete inference case censored; no fewer draws substituted.",
            implementation="Pre-sorted projections and integer replica weights exactly match duplicated-cloud SW2 with S=2R; unit-verified against frozen metric.",
            paper_role_rule="If material noise gap or conservative primary efficiency passes: supporting non-quadratic mechanism/conditional-efficiency benchmark. Otherwise null/limitation result in appendix or robustness section. German N=1000 and prior mixing limitations cannot establish the main large-N ill-conditioned superiority narrative."),
        no_retuning=True,prior_mixing_unverified_retained=True)
    validate_execution(e,c);save_json(path,e)
    (folder/"PROTOCOL.md").write_text(f"""# P1 Held-out Evaluation — German Credit

15 frozen settings x 8 fresh seeds = 120 trajectories. R=512, T prefixes
128/512/2048/8192; public drift -hPg and noise sqrt(2h)chol(P)z.
Maximum active sampling+inference research time: 12 hours; each sampler cap 1800s.
All reference/target/functional/batch/preconditioning and failure rules unchanged.

No heldout seed used before this freeze. No heldout reoptimization or expansion.
Failed seeds invalidate complete paired contrasts; earlier valid prefixes remain.
Selections and independent mixing warnings remain locked.

Four c=.1/.4 x T=2048/8192 noise-gap cells use adjusted 99.375% intervals.
Recovery and remaining gaps use exploratory 95% intervals. Use 1999 hierarchical
paired path bootstraps with fixed reference pools and independent-reference
sensitivity. Integer-weight SW2 is an exact computational acceleration of
resampling whole replicas, not an alternative transport estimator.

Primary efficiency target: SW2<={c['accuracy']['sw2_accuracy_targets'][0]} and
variance NRMSE<=.1. Use frozen CV1/CV2 configurations; report mean threshold
attainment, accuracy intervals, at least 10% paired cost savings, search-inclusive
costs and rare failures separately. Conservative efficiency requires both-reference
accuracy upper bounds within targets as well as cost-ratio upper bound<=.9.
All other marginal/joint comparisons are pointwise exploratory.

Empirical SW2 has finite-cloud/bootstrap bias; percentile intervals are approximate
and do not guarantee population-W2 coverage. No post hoc bias correction.

Final paper-role decision is generated after evaluation. German can support
conditional non-quadratic mechanism/efficiency evidence or a null/limitation
result; it cannot establish large-N/conditioning scaling or universal dominance.
""")
    sources=[Path(__file__),ROOT/"scripts/run_german_p1_heldout.py",ROOT/"scripts/evaluate_german_p1_heldout.py",
        ROOT/"scripts/p1_heldout_bootstrap.py",ROOT/"scripts/german_p1_heldout_pipeline.py",ROOT/"tests/test_p1_heldout.py",
        ROOT/"scripts/p1_bounded_sampler.py",ROOT/"scripts/run_german_p1_pilot.py",path,folder/"PROTOCOL.md"]
    for p in sources:files[str(p.relative_to(ROOT))]=sha256(p)
    lock=dict(schema=1,files=files);save_json(folder/"execution.lock.json",lock);verify_lock(lock,ROOT)
    print(path,flush=True)


if __name__=="__main__":main()
