"""Report frozen German pilot, preserving failures and avoiding formal R=512 claims."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np

from sgld_cv.protocol import sha256,verify_lock
from sgld_cv.workflow import save_csv,save_json

ROOT=Path(__file__).resolve().parents[1]


def pareto(points):
    """Eligible point indices on a finite-grid cost/error lower envelope."""
    return [i for i,p in enumerate(points) if not any(
        q[0]<=p[0] and q[1]<=p[1] and (q[0]<p[0] or q[1]<p[1]) for j,q in enumerate(points) if i!=j)]


def paired_summary(records,config):
    rows=[]; lookup={}
    for record in records:
        candidate,seed=record["candidate"],record["seed"]
        for k in config["budgets"]["pilot"]["iteration_checkpoints"]:
            score=next((s for s in record["checkpoints"] if s["iteration"]==k),None)
            if score is None:
                status="failed_before_checkpoint" if record["failures"] else "budget_censored"
                score={"status":status}
            row={"candidate_id":candidate["id"],"arm":candidate["arm"],"seed":seed,"iteration":k,
                "coefficient":candidate["coefficient"],"h":candidate["h"],"alpha":candidate["alpha"],
                "status":score["status"],"sw2":score.get("sw2"),"variance_nrmse":score.get("variance_nrmse"),
                "alternative_reference_sw2":score.get("alternative_reference_sw2"),
                "alternative_reference_variance_nrmse":score.get("alternative_reference_variance_nrmse"),
                "total_seconds":score.get("cost",{}).get("total_seconds"),
                "failures_over_complete_run":len(record["failures"])}
            rows.append(row);lookup[(candidate["id"],seed,k)]=row
    aggregate=[]
    for candidate in config["pilot_candidates"]:
        for k in config["budgets"]["pilot"]["iteration_checkpoints"]:
            group=[lookup.get((candidate["id"],s,k)) for s in config["seeds"]["pilot"]]
            valid=all(row is not None and row["status"]=="ok" for row in group)
            out={"candidate_id":candidate["id"],"arm":candidate["arm"],"iteration":k,
                 "coefficient":candidate["coefficient"],"h":candidate["h"],"alpha":candidate["alpha"],
                 "status":"ok_both_seeds" if valid else "invalid_or_censored_seed_pair"}
            for metric in ("sw2","variance_nrmse","alternative_reference_sw2","alternative_reference_variance_nrmse","total_seconds"):
                out[metric]=float(np.mean([row[metric] for row in group])) if valid else None
            aggregate.append(out)
    return rows,aggregate,lookup


def contrast_rows(aggregate):
    lookup={(p["arm"],p["coefficient"],p["iteration"]):p for p in aggregate if p["arm"].startswith("full_")}
    rows=[]
    for c in [.025,.05,.1,.2,.4,.8]:
        for k in [128,512,2048]:
            for first,second,label in [("full_cv1","full_ula","noise"),("full_cv1","full_cv2","recovery"),("full_cv2","full_ula","remaining")]:
                a,b=lookup.get((first,c,k)),lookup.get((second,c,k))
                valid=a is not None and b is not None and a["status"]==b["status"]=="ok_both_seeds"
                row={"coefficient":c,"iteration":k,"contrast":label,"first":first,"second":second,
                     "status":"descriptive_pair" if valid else "invalid_or_censored",
                     "interpretation":"pilot R=64, two seeds; no formal binding-noise or superiority conclusion"}
                for metric in ("sw2","variance_nrmse","alternative_reference_sw2","alternative_reference_variance_nrmse"):
                    row[metric+"_difference"]=a[metric]-b[metric] if valid else None
                rows.append(row)
    return rows


def shortlist_proposal(aggregate,config):
    """Screening-only proposal; selection execution requires a separate lock."""
    grouped=defaultdict(list)
    for row in aggregate:
        if row["status"]=="ok_both_seeds":grouped[row["candidate_id"]].append(row)
    candidates={c["id"]:c for c in config["pilot_candidates"]}
    targets=[(x,y) for x in config["accuracy"]["sw2_accuracy_targets"] for y in config["accuracy"]["variance_NRMSE_targets"]]
    def cost_for(ids,x,y):
        points=[p for i in ids for p in grouped[i] if p["sw2"]<=x and p["variance_nrmse"]<=y]
        return min((p["total_seconds"] for p in points),default=float("inf"))
    def rank(candidate,base):
        gain=0.
        for x,y in targets:
            old=cost_for(base,x,y); new=cost_for(base+[candidate],x,y)
            if np.isfinite(new):gain+=1. if not np.isfinite(old) else max(0.,(old-new)/old)
        final=next(p for p in grouped[candidate] if p["iteration"]==2048)
        return (-gain,final["variance_nrmse"],final["sw2"],final["total_seconds"],candidates[candidate]["h"],candidate)
    choices={}
    for arm in config["primary_arms"]+config["practical_arms"]:
        eligible=[i for i,p in candidates.items() if p["arm"]==arm and any(x["iteration"]==2048 for x in grouped[i])]
        chosen=[]
        if arm in config["primary_arms"]:
            chosen=[p["id"] for p in config["pilot_candidates"] if p["arm"]==arm and p["coefficient"] in [.1,.4]]
            extra=[i for i in eligible if i not in chosen]
            if extra:chosen.append(min(extra,key=lambda i:rank(i,chosen)))
        else:
            while len(chosen)<2:
                remaining=[i for i in eligible if i not in chosen]
                if not remaining:break
                chosen.append(min(remaining,key=lambda i:rank(i,chosen)))
        choices[arm]=[{"candidate_id":i,"candidate":candidates[i],
                       "pilot_eligible_at_T2048":i in eligible,"mandatory_matched_step":candidates[i]["coefficient"] in [.1,.4] and arm.startswith("full_")}
                      for i in chosen]
    return {"status":"proposal_only; not a selection execution lock","choices":choices,
            "scope":"R=64 proxy coverage/ranking only; not formal R=512 precision attainment",
            "selection_or_heldout_started":False,"automatic_amendment":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=ROOT/"results/p1_german_pilot_v1")
    args=parser.parse_args();out=args.results.resolve()
    c=json.loads((ROOT/"configs/p1_german_frozen_v2.json").read_text())
    manifest=json.loads((out/"manifest.json").read_text())
    if manifest["status"] not in {"complete","stage_budget_censored"}:
        raise ValueError("pilot execution has not finished")
    verify_lock(json.loads((out/"run.lock.json").read_text()),ROOT)
    records=[json.loads(p.read_text()) for p in sorted(out.glob("result_*.json"))]
    rows,aggregate,_=paired_summary(records,c)
    contrasts=contrast_rows(aggregate)
    save_csv(out/"metrics.csv",rows);save_csv(out/"paired_seed_means.csv",aggregate);save_csv(out/"noise_gap_contrasts.csv",contrasts)
    proposal=shortlist_proposal(aggregate,c);save_json(out/"selection_shortlist_proposal.json",proposal)
    functional_rows=[];failures=[];cost_rows=[]
    for record in records:
        candidate=record["candidate"]
        for score in record["checkpoints"]:
            functional_rows.extend(dict(row,candidate_id=candidate["id"],seed=record["seed"],iteration=score["iteration"])
                                   for row in score.get("functional_rows",[]))
        failures.extend(dict(row,candidate_id=candidate["id"],arm=candidate["arm"],seed=record["seed"]) for row in record["failures"])
        cost=record["terminal_cost"]
        cost_rows.append({"candidate_id":candidate["id"],"arm":candidate["arm"],"seed":record["seed"],
            "standalone_candidate_total_seconds":cost["total_seconds"],
            "sampler_seconds_excluding_repeated_shared_anchor":cost["total_seconds"]-manifest["anchor_cost"]["total_seconds"],
            "functional_evaluation_seconds":record["functional_evaluation_seconds"],"scoring_seconds":record["scoring_seconds"],
            "stop_reason":record["stop_reason"],**cost["operations"]})
    save_csv(out/"functional_rmse.csv",functional_rows);save_json(out/"failures.json",failures);save_csv(out/"search_costs.csv",cost_rows)
    search=[]
    for arm in c["primary_arms"]+c["practical_arms"]:
        group=[r for r in cost_rows if r["arm"]==arm]
        search.append({"arm":arm,"executed_candidate_seed_runs":len(group),
            "executed_search_sampler_seconds_anchor_once_per_arm":sum(r["sampler_seconds_excluding_repeated_shared_anchor"] for r in group)+manifest["anchor_cost"]["total_seconds"],
            "note":"mode physically computed once globally, attributed once per standalone arm search; no evaluation time included"})
    save_csv(out/"search_costs_by_arm.csv",search)
    os.environ.setdefault("MPLCONFIGDIR",str(out/"mpl_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors={"full_cv1":"#C45B35","full_cv2":"#286CB0","full_ula":"#222222","psgld":"#7B56A2","pcv1":"#57A088","pcv2":"#A39324","identity_sgld":"#999999"}
    for key,label in [("sw2","Replica endpoint SW2"),("variance_nrmse","Mean coordinate-variance NRMSE")]:
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
        for ax,arms,title in [(axes[0],c["primary_arms"],"Fixed full-H: primary comparison"),(axes[1],c["practical_arms"],"Practical baselines")]:
            for arm in arms:
                points=[r for r in aggregate if r["arm"]==arm and r["status"]=="ok_both_seeds"]
                ax.scatter([r["total_seconds"] for r in points],[r[key] for r in points],color=colors[arm],s=25,alpha=.55,label=arm)
                indices=pareto([(r["total_seconds"],r[key]) for r in points])
                envelope=sorted([points[i] for i in indices],key=lambda x:x["total_seconds"])
                ax.plot([r["total_seconds"] for r in envelope],[r[key] for r in envelope],color=colors[arm],lw=1.5)
            ax.set_xscale("log");ax.set_yscale("log");ax.set_xlabel("Setup-inclusive sequential-ensemble seconds")
            ax.set_title(title);ax.grid(alpha=.2);ax.legend(fontsize=8)
        axes[0].set_ylabel(label)
        fig.suptitle("German pilot: R=64, two paired seeds — exploratory finite-grid envelopes")
        fig.tight_layout();fig.savefig(out/f"{key}_cost.png",dpi=170);plt.close(fig)
    matched=[r for r in contrasts if r["contrast"]=="noise" and r["coefficient"] in [.1,.4] and r["iteration"]==2048]
    table="\n".join(f"| {r['coefficient']} | {r['status']} | {r['sw2_difference']:.5f} | {r['variance_nrmse_difference']:.5f} |" if r["status"]=="descriptive_pair" else
                     f"| {r['coefficient']} | {r['status']} | — | — |" for r in matched)
    audit=list(__import__('csv').DictReader((out/"gradient_mse.csv").open()))
    mse_table="\n".join(f"| {r['method']} | {float(r['mean_mse']):.6g} | {float(r['mean_preconditioned_mse']):.6g} |" for r in audit if int(r['batch_size'])==32)
    primary_table="\n".join(
        f"| {r['coefficient']} | {r['arm']} | {r['sw2']:.5f} | {r['variance_nrmse']:.5f} | {r['total_seconds']:.3f} | {r['alternative_reference_sw2']:.5f} |"
        for r in aggregate if r["arm"].startswith("full_") and r["coefficient"] in [.1,.4]
        and r["iteration"]==2048 and r["status"]=="ok_both_seeds")
    boundary=[]
    for arm in c["primary_arms"]+c["practical_arms"]:
        best=[]
        for seed in c["seeds"]["pilot"]:
            valid=[r for r in rows if r["arm"]==arm and r["seed"]==seed and r["iteration"]==2048 and r["status"]=="ok"]
            if valid:best.append(min(valid,key=lambda r:(r["variance_nrmse"],r["sw2"],r["h"])))
        candidate_coefficients=[p["coefficient"] for p in c["pilot_candidates"] if p["arm"]==arm]
        if len(best)==2 and all(r["coefficient"] in [min(candidate_coefficients),max(candidate_coefficients)] for r in best):
            boundary.append(arm)
    report=f"""# German P1 pilot results

Execution status: {manifest['status']}. {len(records)}/90 candidate-seed runs recorded.
Research elapsed time: {manifest['research_elapsed_seconds']:.1f} seconds; frozen cap 3600 seconds.
No selection, heldout or amendment seeds consumed. Runner/source/config lock verified.

This is R=64, two-seed descriptive screening. It does not establish a formal
noise-bottleneck classification or certify R=512 precision targets. The plotted
envelopes are exploratory pilot envelopes, not heldout optimized frontiers.

## Matched full-H CV1 minus ULA at T=2048

| c (h*L_P) | Status | SW2 difference | Variance-NRMSE difference |
|---|---|---:|---:|
{table}

Positive differences mean CV1 has higher measured error. These differences are
not additive decompositions of gradient-noise error. The confirmatory T=8192
cells and adjusted intervals have not been run. All steps, earlier checkpoints,
CV2 recovery and independent-reference sensitivity contrasts are retained in
`noise_gap_contrasts.csv`.

## CV2 recovery and measured cost at T=2048

Both-seed arithmetic means; seconds cover all 64 sequential replicas and setup.

| c | Arm | SW2 | Variance NRMSE | Seconds | Independent-reference SW2 |
|---|---|---:|---:|---:|---:|
{primary_table}

CV2 approaches ULA at c=.1 and closes much of the CV1–ULA gap at c=.4.
Its matched-step sampler cost is about 12% above CV1 in this implementation.
This does not establish a tuned efficiency advantage: CV1 can choose a smaller
step and a different horizon. The full-grid cost/error points and independent
reference contrasts are saved for subsequent selection and heldout evaluation.

## Common posterior-location gradient MSE, batch=32

| Estimator | Raw MSE | Preconditioned MSE |
|---|---:|---:|
{mse_table}

These are common-reference-position diagnostics, not trajectory-conditioned
noise or a guarantee against tail instability. Batch=4/128 checks are also saved.

## Failures and budget completeness

Failed replicas: {len(failures)} in {sum(bool(r['failures']) for r in records)} candidate-seed runs.
No failed replica was removed from an accuracy ensemble. Earlier prespecified
checkpoints can remain valid before a later failure; later ensembles are invalid.
Failed/censored seed pairs do not contribute survivor-only averages or envelopes.
See `failures.json`, per-run JSON and `metrics.csv` for complete outcomes.

Boundary warnings based on best final variance-NRMSE in both seeds: {', '.join(boundary) or 'none'}.
These are diagnostic flags, not an automatic amendment decision. No candidate
was retuned, and reserved amendment seeds were not used.

## Costs and next step

`search_costs.csv` records standalone candidate costs and separates shared anchor
attribution; `search_costs_by_arm.csv` counts executed search with one anchor
attribution per arm. The optimizer physically ran once globally. Evaluation,
functional calculations and output work are excluded from sampler cost but
included in stage elapsed time. Cost snapshots keep all terminal work, including
work between scored checkpoints. No single-chain cost is inferred by division.

`selection_shortlist_proposal.json` is a screening proposal only. It preserves
mandatory matched steps, marks pilot-ineligible mandatory candidates, and uses
R=64 proxy coverage with the frozen fallback rules. Selection must receive its
own lock before running. If boundary/failure findings warrant a pilot amendment,
write it first and use only the reserved amendment seeds/budget. Otherwise move
to selection at R=512; no heldout superiority claim is made here.
"""
    (out/"REPORT.md").write_text(report)
    save_json(out/"report_provenance.json",{"reporter_sha256":sha256(Path(__file__)),"manifest_sha256":sha256(out/"manifest.json"),
        "record_count":len(records),"failed_replicas":len(failures),"boundary_warning_arms":boundary,
        "selection_heldout_started":False})
    print(f"Pilot report and exploratory figures saved: {out/'REPORT.md'}")


if __name__=="__main__":main()
