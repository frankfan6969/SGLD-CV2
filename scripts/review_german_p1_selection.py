"""Readable selection review; all claims descriptive, frozen choices untouched."""
import csv
import json
import os
from pathlib import Path
import numpy as np
from sgld_cv.protocol import sha256,verify_lock
from sgld_cv.workflow import save_json,save_csv
from report_german_p1_pilot import pareto

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"results/p1_german_selection_v1"
OUT=ROOT/"results/p1_german_selection_review_v1"


def main():
    verify_lock(json.loads((SOURCE/"final_choices.lock.json").read_text()),ROOT)
    OUT.mkdir(exist_ok=True)
    locked=json.loads((SOURCE/"final_choices.json").read_text());m=json.loads((SOURCE/"manifest.json").read_text())
    e=json.loads((ROOT/"configs/p1_german_selection_v1.json").read_text())
    rows=[]
    for row in csv.DictReader((SOURCE/"selection_means.csv").open()):
        row=dict(row)
        for k in ["h","sw2","variance_nrmse","alternative_reference_sw2","alternative_reference_variance_nrmse","total_seconds"]:
            row[k]=float(row[k]) if row[k] else None
        row["iteration"]=int(row["iteration"]);rows.append(row)
    contrasts=[]
    for coefficient in [.1,.4]:
        for iteration in [2048,8192]:
            chosen={}
            for arm in ["full_cv1","full_cv2","full_ula"]:
                cid=next(x["id"] for x in e["candidates"] if x["arm"]==arm and x["coefficient"]==coefficient)
                chosen[arm]=next(x for x in rows if x["candidate_id"]==cid and x["iteration"]==iteration)
            for a,b,label in [("full_cv1","full_ula","noise gap"),("full_cv1","full_cv2","CV2 recovery"),("full_cv2","full_ula","remaining gap")]:
                x,y=chosen[a],chosen[b];valid=x["status"]==y["status"]=="ok_all_four_seeds"
                contrast=dict(coefficient=coefficient,iteration=iteration,contrast=label,status="descriptive" if valid else "failed_seed_set")
                for metric in ["sw2","variance_nrmse","alternative_reference_sw2","alternative_reference_variance_nrmse"]:
                    contrast[metric]=x[metric]-y[metric] if valid else None
                contrasts.append(contrast)
    save_csv(OUT/"matched_contrasts.csv",contrasts)
    failures=[]
    for p in SOURCE.glob("result_*.json"):
        r=json.loads(p.read_text())
        failures.extend(dict(candidate_id=r["candidate"]["id"],seed=r["seed"],**f) for f in r["failures"])
    save_json(OUT/"failures.json",failures)
    os.environ.setdefault("MPLCONFIGDIR",str(OUT/"mpl_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors={"full_cv1":"#C45B35","full_cv2":"#286CB0","full_ula":"#222222","psgld":"#7B56A2","pcv1":"#57A088","pcv2":"#A39324","identity_sgld":"#999999"}
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for ax,metric,label in zip(axes,["sw2","variance_nrmse"],["Replica endpoint SW2","Mean coordinate-variance NRMSE"]):
        for arm in ["full_cv1","full_cv2","full_ula"]:
            points=[r for r in rows if r["arm"]==arm and r["status"]=="ok_all_four_seeds"]
            ax.scatter([r["total_seconds"] for r in points],[r[metric] for r in points],s=28,alpha=.6,color=colors[arm],label=arm)
            front=sorted([points[i] for i in pareto([(r["total_seconds"],r[metric]) for r in points])],key=lambda r:r["total_seconds"])
            ax.plot([r["total_seconds"] for r in front],[r[metric] for r in front],color=colors[arm],alpha=.8)
        ax.set_xscale("log");ax.set_yscale("log");ax.set_xlabel("Setup-inclusive 512-replica ensemble seconds")
        ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle("German Selection: four-seed means; tuned finite-grid envelopes, not heldout evidence")
    fig.tight_layout();fig.savefig(OUT/"selection_frontiers.png",dpi=180);plt.close(fig)
    table="\n".join(f"| {r['arm']} | {next(c['coefficient'] for c in e['candidates'] if c['id']==r['candidate_id'])} | {r['iteration']} | {r['sw2']:.4f} | {r['variance_nrmse']:.4f} | {r['total_seconds']:.2f} |" if r["sw2"] is not None else
        f"| {r['arm']} | 0.4 | {r['iteration']} | invalid | invalid | — |" for r in locked["matched_cells"])
    diagnostic=ROOT/"results/p1_german_selection_diagnostics_v1"
    supplemental="Independent diagnostics and single-chain timings are still running."
    if (diagnostic/"manifest.json").exists():
        dm=json.loads((diagnostic/"manifest.json").read_text())
        ds=list(csv.DictReader((diagnostic/"mixing_summary.csv").open()))
        verified=[r["candidate_id"] for r in ds if r["mixing_verified"]=="True"]
        unverified=[r["candidate_id"] for r in ds if r["fixed_metric"]=="True" and r["mixing_verified"]!="True"]
        supplemental=f"""Supplement status: {dm['status']}; {dm['diagnostic_runs']} diagnostic runs,
{dm['timing_runs']} independent R=1 timing runs. Total active Selection research
time including supplement: {dm['total_selection_research_seconds']/3600:.3f} hours.

Fixed-metric configurations passing all four diagnostic-seed gates:
{', '.join(verified) or 'none'}.
Fixed-metric configurations with mixing unverified:
{', '.join(unverified) or 'none'}.
Adaptive ESS remains exploratory regardless of numerical thresholds. These gates
do not erase from-mode accuracy observations or unlock retuning. Detailed ESS per
gradient/second is in diagnostic JSONs; separately measured single-chain times
are in `results/p1_german_selection_diagnostics_v1/single_chain_summary.csv`.

In particular, c=.1 CV1/CV2/ULA have maximum Rhat about 1.012–1.013, above
the frozen 1.01 threshold; c=.05 CV1 has max Rhat=1.0267 and minimum bulk
ESS=236.6. The promising threshold-cost comparison therefore remains a
from-mode estimator result, not validated general mixing throughput. Identity
and adaptive arms show much stronger mixing warnings at this diagnostic budget.
The c=.4 CV2 diagnostic pass does not override its rare replica failures:
these diagnostic chains sample far fewer trajectories than the endpoint experiment.
"""
    (OUT/"REPORT.md").write_text(f"""# German P1 Selection review

## Outcome

68/68 trajectories completed at R=512, four tuning seeds; no budget censoring.
Sampling-stage research time: {m['research_elapsed_seconds']/3600:.3f} hours.
Choices were locked before this review; no heldout seed has been used.

The results support a conditional CV2 benefit in estimator accuracy/cost. They
do not support universal dominance, and they reveal rare longer-horizon failures
at c=.4 that the smaller pilot did not observe.

## Matched-step evidence

| Arm | c | T | SW2 | Variance NRMSE | Ensemble seconds |
|---|---:|---:|---:|---:|---:|
{table}

At c=.1,T=2048, CV2 nearly matches ULA and improves over CV1 in both primary
errors, at about 13% greater matched-step sampler cost. Independent-reference
contrasts retain the direction. At c=.4,T=2048 it closes much of the CV1–ULA gap.
These are tuning-data means without confirmatory intervals.

## Matched-threshold efficiency

For variance NRMSE <=.1 (also SW2 <=.113372), selected CV1 uses c=.05,T=8192:
139.88 seconds, SW2=.1046, variance NRMSE=.0673. Selected CV2 uses c=.1,T=2048:
39.34 seconds, SW2=.1037, variance NRMSE=.0798. Both pass these thresholds under
the independent reference. CV2 is approximately 3.56 times faster for this
threshold pair in Selection; the attained errors are not identical and heldout
must validate the selected settings.

When only SW2 <=.113372 is required, CV1 already passes at c=.05,T=128 in 2.26
seconds; CV2 passes at c=.1,T=128 in 2.51 seconds. Thus SW2-only performance does
not show a universal CV2 cost advantage. Endpoint distribution accuracy and
ergodic functional precision answer different questions.

The strict variance target .05 is unattained by CV1/CV2 on this grid. ULA barely
passes using the primary reference but fails under the independent reference:
this target is reference-sensitive, not robustly established.

## Failures

Full-H CV2 c=.4 failed replica 412 at iteration 4821 for seed142201, and replica
224 at iteration4486 for seed142202. Both triggered the original radius threshold.
This is 2/2048 replicas across the four runs (0.098%), but 2/4 seed ensembles are
invalid at T=8192. Consequently no four-seed c=.4,T=8192 CV2 mean or contrast is
reported. Earlier T=2048 results remain valid. All other shortlisted configurations
had no observed failures over this run length; this is not a stability guarantee.

## Remaining checks and interpretation

{supplemental}

Next, write a separate heldout execution lock using the already locked
target-specific choices and mandatory matched cells. No retuning or grid
expansion is allowed. Confirmatory paired intervals, reference sensitivity,
failure rates and at least 10% matched-threshold cost savings are required for
an efficiency claim. Selection means alone do not satisfy that requirement.
""")
    files=[p for p in OUT.iterdir() if p.is_file() and p.name!="delivery.lock.json"]+[Path(__file__)]
    save_json(OUT/"delivery.lock.json",dict(schema=1,files={str(p.relative_to(ROOT)):sha256(p) for p in files}))
    print(OUT/"REPORT.md")


if __name__=="__main__":main()
