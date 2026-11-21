"""Freeze target-specific selection choices; no heldout execution or claims."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from sgld_cv.protocol import sha256,verify_lock
from sgld_cv.workflow import save_json,save_csv

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/p1_german_selection_v1"


def aggregate(records,e):
    lookup={(r["candidate"]["id"],r["seed"]):r for r in records};rows=[]
    for c in e["candidates"]:
        for k in e["budget"]["iteration_checkpoints"]:
            scores=[]
            for seed in e["seeds"]:
                r=lookup.get((c["id"],seed),{})
                s=next((p for p in r.get("checkpoints",[]) if p["iteration"]==k),None)
                scores.append(s)
            valid=all(s is not None and s["status"]=="ok" for s in scores)
            row=dict(candidate_id=c["id"],arm=c["arm"],h=c["h"],iteration=k,
                status="ok_all_four_seeds" if valid else "failed_or_censored_seed_set")
            for name in ["sw2","variance_nrmse","alternative_reference_sw2","alternative_reference_variance_nrmse"]:
                row[name]=float(np.mean([s[name] for s in scores])) if valid else None
            row["total_seconds"]=float(np.mean([s["cost"]["total_seconds"] for s in scores])) if valid else None
            rows.append(row)
    return rows


def choose(rows,arm,x,y,alternative=False):
    prefix="alternative_reference_" if alternative else ""
    eligible=[p for p in rows if p["arm"]==arm and p["status"]=="ok_all_four_seeds" and
              (x is None or p[prefix+"sw2"]<=x) and (y is None or p[prefix+"variance_nrmse"]<=y)]
    if not eligible:return None
    return min(eligible,key=lambda p:(p["total_seconds"],p["h"],p["iteration"],p["candidate_id"]))


def main():
    e=json.loads((ROOT/"configs/p1_german_selection_v1.json").read_text())
    c=json.loads((ROOT/e["parent_config"]).read_text());m=json.loads((OUT/"manifest.json").read_text())
    if m["status"] not in ["complete","stage_budget_censored"]:raise ValueError("selection sampling unfinished")
    verify_lock(json.loads((OUT/"run.lock.json").read_text()),ROOT)
    if (OUT/"final_choices.lock.json").exists():raise ValueError("final selection choices already locked")
    records=[json.loads(p.read_text()) for p in sorted(OUT.glob("result_*.json"))]
    rows=aggregate(records,e);save_csv(OUT/"selection_means.csv",rows)
    targets=[("sw2",x,None) for x in c["accuracy"]["sw2_accuracy_targets"]]
    targets += [("variance",None,y) for y in c["accuracy"]["variance_NRMSE_targets"]]
    targets += [("joint",x,y) for x in c["accuracy"]["sw2_accuracy_targets"] for y in c["accuracy"]["variance_NRMSE_targets"]]
    choices=[]
    for arm in c["primary_arms"]+c["practical_arms"]:
        for kind,x,y in targets:
            best=choose(rows,arm,x,y);alt=choose(rows,arm,x,y,True)
            choices.append(dict(arm=arm,target_kind=kind,sw2_target=x,variance_target=y,
                choice=best,status="attained_in_selection" if best else "unattained",
                independent_reference_choice=alt,
                primary_choice_meets_independent_reference=bool(best and (x is None or best["alternative_reference_sw2"]<=x) and (y is None or best["alternative_reference_variance_nrmse"]<=y))))
    save_json(OUT/"final_choices.json",dict(status="locked tuning choices; no heldout inference",choices=choices,
        matched_cells=[p for p in rows if p["arm"].startswith("full_") and next(q for q in e["candidates"] if q["id"]==p["candidate_id"])["coefficient"] in [.1,.4] and p["iteration"] in [2048,8192]],
        diagnostics_and_single_chain_timings="pending; must be completed within remaining selection research budget before heldout",heldout_started=False))
    failures=[dict(candidate_id=r["candidate"]["id"],seed=r["seed"],**f) for r in records for f in r["failures"]]
    save_json(OUT/"failures.json",failures)
    (OUT/"REPORT.md").write_text(f"""# German P1 Selection sampling

Status: {m['status']}; {len(records)}/{m['planned_runs']} runs recorded.
R=512; four selection seeds; prefixes 128/512/2048/8192.
Elapsed sampler-stage research time: {m['research_elapsed_seconds']:.1f} seconds.
Failed replicas: {len(failures)}. Invalid seed sets are never averaged over survivors.

`selection_means.csv` retains every candidate/prefix, including failed/censored
sets. `final_choices.json` locks marginal/joint target-specific minimum measured
cost choices, unattained targets, mandatory matched cells, and independent-reference
sensitivity before heldout. The selected configurations were tuned on these data;
no confirmatory superiority or equivalence claim is made here.

Separate overdispersed diagnostics and independent single-chain timing repetitions
remain pending within the remaining selection elapsed budget. Selection is not
fully finalized for heldout until those tasks are accounted for. No heldout seed
has been consumed; no grid expansion or heldout reoptimization is permitted.
""")
    paths=[p for p in OUT.iterdir() if p.is_file() and p.name!="final_choices.lock.json"]+[Path(__file__)]
    lock=dict(schema=1,files={str(p.relative_to(ROOT)):sha256(p) for p in paths})
    save_json(OUT/"final_choices.lock.json",lock);verify_lock(lock,ROOT)
    print(OUT/"REPORT.md")


if __name__=="__main__":main()
