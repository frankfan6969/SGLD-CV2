"""Heldout-only inference and paper-role decision; never retune choices."""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from p1_heldout_bootstrap import prepare_projected,weighted_sw2,bootstrap_design,interval
from sgld_cv.functionals import FunctionalSpec
from sgld_cv.protocol import sha256,verify_lock
from sgld_cv.workflow import save_json,save_csv

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/p1_german_heldout_v1"


def classify_gap(ci,margin):
    low=np.asarray(ci["lower"]);high=np.asarray(ci["upper"]);margin=np.asarray(margin)
    if np.all(low>=-margin) and np.all(high<=margin):return "negligible_at_this_cell"
    if np.any(low>margin) and not np.any(high<-margin):return "substantial"
    return "inconclusive_or_mixed"


def main():
    started=perf_counter();e=json.loads((ROOT/"configs/p1_german_heldout_v1.json").read_text())
    c=json.loads((ROOT/e["parent_config"]).read_text());m=json.loads((OUT/"manifest.json").read_text())
    if m["status"] not in ["complete","stage_budget_censored"]:raise ValueError("heldout sampling incomplete")
    verify_lock(json.loads((OUT/"run.lock.json").read_text()),ROOT)
    deadline=started+c["budgets"]["heldout"]["max_stage_elapsed_seconds"]-m["research_elapsed_seconds"]
    report=OUT/"evaluation";report.mkdir(exist_ok=False)
    sampling_lock=dict(schema=1,files={str(p.relative_to(ROOT)):sha256(p) for p in OUT.iterdir() if p.is_file()})
    save_json(report/"sampling.lock.json",sampling_lock)
    selected=json.loads((ROOT/"results/p1_german_selection_v1/final_choices.json").read_text())
    with np.load(ROOT/c["target"]["assets_source"]/"frozen_assets.npz") as f:a={k:f[k] for k in f.files}
    targets=a["function_targets"];scales=a["function_scales"];d=len(a["anchor"])
    fm=json.loads((ROOT/c["target"]["assets_source"]/"functional_reference.json").read_text())
    alt=targets+np.asarray(fm["independent_reference_disagreement_normalized"])*scales
    # Only derive() uses this object; prediction rows are irrelevant here.
    functional=FunctionalSpec(a["anchor"],a["transform"],a["slow_direction"],np.empty((0,d)))
    cases={(q["choice"]["candidate_id"],q["choice"]["iteration"]) for q in selected["choices"] if q["choice"]}
    cases|={(q["candidate_id"],q["iteration"]) for q in selected["matched_cells"]}
    outer,counts=bootstrap_design(8,512,1999,c["seeds"]["uncertainty"][0])
    np.savez_compressed(report/"bootstrap_design.npz",outer=outer,counts=counts)
    refs={}
    for seed in e["seeds"]:
        with np.load(OUT/f"reference_seed{seed}.npz") as z:
            refs[seed]=np.stack([((z[k]-a["anchor"])@a["transform"])@a["projections"] for k in ["scoring","sensitivity"]])
    distributions={};summaries=[]
    with threadpool_limits(limits=1):
        for case_index,(cid,k) in enumerate(sorted(cases)):
            prepared=[];errors=[];observed=[];costs=[];statuses=[]
            for seed in e["seeds"]:
                path=OUT/f"result_{cid}_seed{seed}.json"
                r=json.loads(path.read_text()) if path.exists() else {}
                s=next((s for s in r.get("checkpoints",[]) if s["iteration"]==k),None)
                statuses.append(s["status"] if s else "missing_or_censored")
                if s is None or s["status"]!="ok":continue
                with np.load(OUT/f"states_{cid}_seed{seed}.npz") as z:
                    j=list(z["checkpoints"]).index(k);theta=z["endpoints"][j];values,_=functional.derive(z["functional_averages"][j])
                xp=((theta-a["anchor"])@a["transform"])@a["projections"]
                cache=prepare_projected(xp,refs[seed]);prepared.append(cache)
                err=np.stack([((values[:,d:2*d]-t[d:2*d])/scales[d:2*d])**2 for t in [targets,alt]])
                errors.append(err)
                point=np.array([[s["sw2"],s["variance_nrmse"]],[s["alternative_reference_sw2"],s["alternative_reference_variance_nrmse"]]])
                check=weighted_sw2(*cache,np.ones(512,dtype=np.int16))
                np.testing.assert_allclose(check,point[:,0],rtol=1e-10,atol=1e-12)
                observed.append(point);costs.append(s["cost"]["total_seconds"])
            status="ok" if len(prepared)==8 else "failed_or_censored_seed_set"
            summary=dict(candidate_id=cid,iteration=k,status=status,seed_statuses=statuses)
            if status=="ok":
                draws=np.zeros((1999,2,2));complete=True
                for b in range(1999):
                    if b%32==0 and perf_counter()>=deadline:complete=False;break
                    for slot in range(8):
                        seed_index=outer[b,slot];w=counts[b,slot]
                        draws[b,:,0]+=weighted_sw2(*prepared[seed_index],w)/8
                        draws[b,:,1]+=np.sqrt(np.sum(errors[seed_index]*w[None,:,None],axis=1)/512).mean(axis=-1)/8
                point=np.mean(observed,axis=0)
                if complete:
                    cost_draws=np.asarray(costs)[outer].mean(axis=1)
                    distributions[(cid,k)]=(draws,cost_draws,point,float(np.mean(costs)))
                    np.savez_compressed(report/f"bootstrap_{cid}_T{k}.npz",accuracy=draws,cost=cost_draws,point=point,mean_cost=np.mean(costs))
                    summary.update(accuracy=interval(draws,point,.95),cost=interval(cost_draws,np.mean(costs),.95))
                else:summary["status"]="inference_budget_censored"
            summaries.append(summary);save_json(report/"case_intervals.json",summaries)
            print(f"Inference {case_index+1}/{len(cases)} {cid} T{k}: {summary['status']}",flush=True)
    contrasts=[]
    for coefficient in [.1,.4]:
        for k in [2048,8192]:
            keys={arm:(next(p["id"] for p in e["candidates"] if p["arm"]==arm and p["coefficient"]==coefficient),k)
                  for arm in ["full_cv1","full_cv2","full_ula"]}
            for first,second,label in [("full_cv1","full_ula","noise"),("full_cv1","full_cv2","recovery"),("full_cv2","full_ula","remaining")]:
                row=dict(coefficient=coefficient,iteration=k,contrast=label,status="failed_or_censored")
                if keys[first] in distributions and keys[second] in distributions:
                    x,y=distributions[keys[first]],distributions[keys[second]]
                    ci=interval(x[0]-y[0],x[2]-y[2],.99375 if label=="noise" else .95)
                    row.update(status="ok",interval=ci)
                    if label=="noise":
                        labels=[classify_gap(dict(lower=ci["lower"][i],upper=ci["upper"][i]),[.01,.025]) for i in range(2)]
                        row.update(reference_classifications=labels,classification=labels[0] if labels[0]==labels[1] else "reference_sensitive_inconclusive")
                contrasts.append(row)
    save_json(report/"matched_contrasts.json",contrasts)
    efficiencies=[]
    for q in selected["choices"]:
        if q["arm"]!="full_cv1":continue
        other=next(p for p in selected["choices"] if p["arm"]=="full_cv2" and
            (p["target_kind"],p["sw2_target"],p["variance_target"])==(q["target_kind"],q["sw2_target"],q["variance_target"]))
        row=dict(target_kind=q["target_kind"],sw2_target=q["sw2_target"],variance_target=q["variance_target"],status="selection_target_unattained")
        if q["choice"] and other["choice"]:
            key1=(q["choice"]["candidate_id"],q["choice"]["iteration"]);key2=(other["choice"]["candidate_id"],other["choice"]["iteration"])
            row.update(cv1_choice=key1,cv2_choice=key2,status="failed_or_censored")
            if key1 in distributions and key2 in distributions:
                x,y=distributions[key1],distributions[key2]
                ratio=interval(y[1]/x[1],y[3]/x[3],.95)
                tolerance=np.array([q["sw2_target"] if q["sw2_target"] is not None else np.inf,
                                    q["variance_target"] if q["variance_target"] is not None else np.inf])
                point_ok=bool(np.all(x[2]<=tolerance) and np.all(y[2]<=tolerance))
                upper1=np.asarray(interval(x[0],x[2],.95)["upper"]);upper2=np.asarray(interval(y[0],y[2],.95)["upper"])
                certified=bool(np.all(upper1<=tolerance) and np.all(upper2<=tolerance))
                row.update(status="evaluated",cv2_over_cv1_cost_ratio=ratio,heldout_mean_thresholds_pass_both_references=point_ok,
                    accuracy_upper_intervals_pass_both_references=certified,
                    at_least_10percent_savings_with_paired_interval=bool(ratio["upper"]<=.9),
                    conservative_efficiency_supported=point_ok and certified and ratio["upper"]<=.9,
                    interpretation="pointwise; only the preregistered primary joint target is confirmatory; other targets exploratory")
        efficiencies.append(row)
    save_json(report/"fixed_choice_efficiency.json",efficiencies)
    failures=[]
    for p in OUT.glob("result_*.json"):
        r=json.loads(p.read_text());failures.extend(dict(candidate_id=r["candidate"]["id"],seed=r["seed"],**f) for f in r["failures"])
    save_json(report/"failures.json",failures)
    # Fixed tuning/search cost, not divided by replicas and never fitted on heldout.
    searches=[]
    for arm in c["primary_arms"]+c["practical_arms"]:
        total=0.
        for folder in ["p1_german_pilot_v1","p1_german_amendment_v1","p1_german_selection_v1"]:
            source=ROOT/"results"/folder;sm=json.loads((source/"manifest.json").read_text());anchor=sm["anchor_cost"]["total_seconds"]
            records=[json.loads(p.read_text()) for p in source.glob("result_*.json")]
            group=[r for r in records if r["candidate"]["arm"]==arm]
            if group:total+=sum(r["terminal_cost"]["total_seconds"]-anchor for r in group)+anchor
        searches.append(dict(arm=arm,sampler_search_seconds=total,scope="pilot+amendment+selection sampler work, anchor once per arm/stage; reference/design/evaluation separate"))
    save_csv(report/"search_costs.csv",searches)
    tuning={r["arm"]:r["sampler_search_seconds"] for r in searches}
    for row in efficiencies:
        if row["status"]!="evaluated":continue
        x=distributions[tuple(row["cv1_choice"])];y=distributions[tuple(row["cv2_choice"])]
        row["search_inclusive_cv2_over_cv1_cost_ratio"]=interval(
            (y[1]+tuning["full_cv2"])/(x[1]+tuning["full_cv1"]),
            (y[3]+tuning["full_cv2"])/(x[3]+tuning["full_cv1"]),.95)
    save_json(report/"fixed_choice_efficiency.json",efficiencies)
    primary=next(q for q in efficiencies if q["target_kind"]=="joint" and q["sw2_target"]==c["accuracy"]["sw2_accuracy_targets"][0] and q["variance_target"]==.1)
    robust_noise=any(q.get("classification")=="substantial" for q in contrasts if q["contrast"]=="noise")
    role=("supporting non-quadratic mechanism/conditional-efficiency benchmark; not the main large-N ill-conditioned superiority result"
          if robust_noise or primary.get("conservative_efficiency_supported") else
          "null/limitation benchmark in appendix or brief robustness section; not a primary superiority result")
    reason="Heldout detects a material CV1–ULA gap" if robust_noise else "Material noise-gap evidence is inconclusive or failed at the tested cells"
    if primary.get("conservative_efficiency_supported"):
        reason+="; the primary fixed-choice efficiency comparison passes conservative accuracy/cost checks, with prior mixing limitations retained"
    else:reason+="; primary efficiency is not conservatively established across accuracy uncertainty, cost and reference sensitivity"
    decision=dict(role=role,reason=reason,primary_efficiency=primary,robust_noise_gap_in_at_least_one_cell=robust_noise,
        mixing="Selection diagnostic failures are retained; from-mode results do not validate general mixing throughput",
        scope="German N=1000 cannot establish large-N or conditioning-scaling superiority; a larger non-quadratic target remains necessary")
    save_json(report/"paper_role_decision.json",decision)
    summary=dict(status="complete" if all(q["status"]!="inference_budget_censored" for q in summaries) else "inference_budget_censored",
        inference_research_seconds=perf_counter()-started,total_heldout_research_seconds=m["research_elapsed_seconds"]+perf_counter()-started,
        cases=len(summaries),failed_replicas=len(failures),bootstrap_repeats=1999,heldout_no_retuning=True)
    save_json(report/"manifest.json",summary)
    (report/"REPORT.md").write_text(f"""# German P1 heldout evaluation and paper role

Sampling: {m['status']}, {m['completed_runs']}/{m['planned_runs']} runs.
Inference: {summary['status']}; {len(failures)} failed replicas retained.
Eight new algorithm seeds, R=512, frozen selections/targets/references.

Paper role: **{role}**.
{reason}.

`matched_contrasts.json` reports four matched noise cells with 99.375% intervals,
exploratory recovery/remaining contrasts, and independent-reference classifications.
`fixed_choice_efficiency.json` reports the locked marginal/joint target comparisons;
no heldout reoptimization occurred. The primary target is SW2<={c['accuracy']['sw2_accuracy_targets'][0]}
and variance NRMSE<=.1. Target means, accuracy intervals and paired cost ratios
are separate: a mean passing a target is not confidence-certified attainment.

Percentile bootstrap uncertainty is approximate and conditional on fixed
correlated HMC reference pools. Whole replicas and algorithm-seed clusters are
resampled; time draws and HMC reference rows are not iid-resampled. All 1999
replicates use paired method indices. SW2 remains a finite-cloud statistic with
bootstrap/finite-sample bias; intervals do not guarantee population-W2 coverage.
Failed/censored seed ensembles produce no survivor-only contrasts or CIs.

Previously unverified mixing remains unverified; diagnostic passes cannot erase
rare endpoint failures. No result here establishes uniform CV2 dominance or
large-N/ill-conditioning scaling. A larger non-quadratic primary benchmark is
still needed for that paper narrative.
Distance-to-posterior contrasts are not additive decompositions of total error.

`search_costs.csv` separates tuning work from selected-config costs. Raw per-run
JSON retains operation counters and setup-inclusive cumulative time. Design,
reference and evaluation work remain distinct research costs.
""")
    verify_lock(json.loads((OUT/"run.lock.json").read_text()),ROOT)
    verify_lock(sampling_lock,ROOT)
    files=[p for p in report.iterdir() if p.is_file()]
    save_json(report/"delivery.lock.json",dict(schema=1,files={str(p.relative_to(ROOT)):sha256(p) for p in files}))
    print(report/"REPORT.md",flush=True)


if __name__=="__main__":main()
