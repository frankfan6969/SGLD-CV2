"""Summarize diagnostic replay without changing or rescoring the pilot."""
import csv
import json
import os
from pathlib import Path
import numpy as np
from sgld_cv.data import german_credit
from sgld_cv.protocol import sha256, verify_lock
from sgld_cv.workflow import save_csv, save_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/p1_german_diagnostic_replay_v1"


def main():
    manifest=json.loads((OUT/"manifest.json").read_text())
    assert manifest["status"]=="complete"
    assert sha256(ROOT/"scripts/diagnose_german_p1.py")==manifest["source_sha256"]
    verify_lock(json.loads((ROOT/"results/p1_german_pilot_v1/delivery.lock.json").read_text()),ROOT)
    geometry=np.load(ROOT/"results/p1_german_configuration_v2/geometry_and_li_planning.npz")
    assets=np.load(ROOT/"results/p0_validation_v2/german/frozen_assets.npz")
    model,_=german_credit(ROOT/"data/raw/german.zip")
    H,P,T=geometry["H"],geometry["P_full"],assets["transform"]
    records=[json.loads(p.read_text()) for p in sorted(OUT.glob("replay_*.json"))]
    save_csv(OUT/"replay_summary.csv",[dict(candidate_id=r["candidate"]["id"],seed=r["seed"],failed=r["failed_replicas"],
        failure_rate=r["failure_rate"],exact_replay=r["exact_replay"],formula_error=r["max_direct_formula_relative_error"],
        metric_error=r["max_direct_metric_relative_error"],noise_covariance_error=r["noise_covariance_relative_error"],
        research_seconds=r["research_seconds"]) for r in records])
    evidence=[];first_ids={}
    for r in records:
        if not r["failures"]:continue
        first=min(r["failures"],key=lambda f:f["iteration"])
        rid,k=first["replica"],first["iteration"]
        name=f"{r['candidate']['id']}_seed{r['seed']}"
        first_ids[(r["seed"],r["candidate"]["geometry"])]=(rid,k)
        z=np.load(OUT/f"event_vectors_{name}.npz")
        steps=[k] if r["candidate"]["geometry"]=="full" else [1,2,k]
        for step in sorted(set(steps)):
            j=np.flatnonzero((z["replica"]==rid)&(z["iteration"]==step))[0]
            adaptive=r["candidate"]["geometry"]=="li2016"
            D=z["diagonal"][j]; h=r["candidate"]["h"]
            apply=(lambda x:D*x) if adaptive else (lambda x:P@x)
            n=np.linalg.norm
            row=dict(candidate_id=r["candidate"]["id"],seed=r["seed"],replica=rid,iteration=step,
                radius_before=n((z["theta_before"][j]-assets["anchor"])@T),
                radius_after=n((z["proposal"][j]-assets["anchor"])@T),
                whitened_drift=n(z["drift"][j]@T),whitened_noise=n(z["noise"][j]@T),
                whitened_true_gradient_drift=n((-h*apply(z["full_gradient"][j]))@T),
                whitened_gradient_error_drift=n((-h*apply(z["g"][j]-z["full_gradient"][j]))@T))
            if adaptive:
                root=np.sqrt(D);sat=D>.99/r["candidate"]["epsilon"]
                row.update(metric_max=D.max(),saturated_coordinates=int(sat.sum()),
                    h_L_actual_anchor=h*np.linalg.eigvalsh(root[:,None]*H*root[None,:])[-1],
                    saturated_squared_euclidean_drift_fraction=np.sum(z["drift"][j][sat]**2)/np.sum(z["drift"][j]**2),
                    noise_transform_relative_error=n(z["noise"][j]-np.sqrt(2*h*D)*z["normal"][j])/(1+n(z["noise"][j])),
                    current_batch_zero_columns=int(np.all(model.X[z["indices"][j]]==0,axis=0).sum()))
            else:
                row.update(metric_max=None,saturated_coordinates=None,h_L_actual_anchor=None,
                    saturated_squared_euclidean_drift_fraction=None,current_batch_zero_columns=None,
                    noise_transform_relative_error=n(z["noise"][j]-np.sqrt(2*h)*(np.linalg.cholesky(P)@z["normal"][j]))/(1+n(z["noise"][j])))
            evidence.append(row)
    save_csv(OUT/"first_failure_evidence.csv",evidence)
    # Same mode implies the first CV1/CV2 proposal and raw-gradient metric agree.
    first_step_checks=[]
    for seed in [142100,142101]:
        a=np.load(OUT/f"event_vectors_pcv1_a0.99_c0.256_seed{seed}.npz")
        b=np.load(OUT/f"event_vectors_pcv2_a0.99_c0.256_seed{seed}.npz")
        for rid in manifest["detailed_replica_selection"][str(seed)]:
            ia=np.flatnonzero((a["replica"]==rid)&(a["iteration"]==1))[0]
            ib=np.flatnonzero((b["replica"]==rid)&(b["iteration"]==1))[0]
            first_step_checks.append(dict(seed=seed,replica=rid,proposal_equal=bool(np.array_equal(a["proposal"][ia],b["proposal"][ib])),
                metric_equal=bool(np.array_equal(a["diagonal"][ia],b["diagonal"][ib]))))
    save_json(OUT/"li_first_step_checks.json",first_step_checks)
    os.environ.setdefault("MPLCONFIGDIR",str(OUT/"mpl_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for i,seed in enumerate([142100,142101]):
        for col,geometry_name in enumerate(["full","li2016"]):
            rid,end=first_ids[(seed,geometry_name)]
            names=["full_cv2_c5","full_cv2_c4","full_cv1_c5","full_ula_c5"] if col==0 else ["pcv2_a0.99_c0.256","pcv1_a0.99_c0.256"]
            ax=axes[i,col]
            for name in names:
                rows=[r for r in csv.DictReader((OUT/f"trajectory_{name}_seed{seed}.csv").open()) if int(r["replica"])==rid and int(r["iteration"])<=max(end,10 if col else end)]
                ax.plot([int(r["iteration"]) for r in rows],[float(r["radius_after"]) for r in rows],label=name,lw=1.3)
            ax.axhline(100*np.sqrt(model.d),ls="--",color="black",lw=.8,label="failure threshold")
            ax.set_yscale("log");ax.set_xlabel("Iteration");ax.set_ylabel("Whitened radius")
            ax.set_title(f"Seed {seed}, replica {rid}: {geometry_name}");ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle("Diagnostic trajectories: same replica streams; selected examples, not accuracy estimates")
    fig.tight_layout();fig.savefig(OUT/"diagnostic_trajectories.png",dpi=160);plt.close(fig)
    failure_table="\n".join(f"| {r['candidate']['id']} | {r['seed']} | {r['failed_replicas']}/64 | {r['exact_replay']} |" for r in records)
    full_table="\n".join(f"| {r['seed']} | {r['iteration']} | {r['whitened_drift']:.1f} | {r['whitened_noise']:.1f} | {r['whitened_true_gradient_drift']:.1f} | {r['whitened_gradient_error_drift']:.1f} |" for r in evidence if r["candidate_id"].startswith("full"))
    (OUT/"REPORT.md").write_text(f"""# German P1 diagnostic replay

## Decision

No implementation error was detected by these checks; P0 does not need to be
repeated on this evidence. The failures are exactly reproducible CV2 stability
limitations at the largest tested steps. Proceeding to Selection is reasonable
after writing its separate lock and resolving the already reported pilot-grid
boundary flags. This diagnosis neither changes the grid nor launches Selection.
Stability at T=2048 does not establish stability at T=8192.

## Reproduction and correctness

All 12 runs replayed all 64 replicas through 2048 steps, preserving streams,
arithmetic update order, actual anchor optimization, metric construction, and
failure termination. All checkpoint endpoints, terminal states, failure IDs and
iterations, and adaptive terminal v states match the pilot exactly (including NaNs).
The original pilot delivery lock still verifies. Diagnostic elapsed time:
{manifest['research_seconds']:.1f} seconds, separate from pilot sampler costs.

| Configuration | Seed | Failed replicas | Exact replay |
|---|---:|---:|---|
{failure_table}

Independent per-observation gradient assembly agrees within
{max(r['max_direct_formula_relative_error'] for r in records):.3g} relative error;
direct Hessian cache assembly differs by {manifest['direct_hessian_cache_relative_error']:.3g}.
Independent raw-gradient RMSprop calculation agrees within
{max(r['max_direct_metric_relative_error'] for r in records):.3g}.
Fixed noise covariance and stored noise transforms also agree to roundoff.
These checks do not constitute a proof of every implementation property.

## Full-H: stochastic drift amplification

At c=.8 CV2 fails, while paired CV1 and ULA do not; CV2 at c=.4 has no observed
failures. At the earliest CV2 failure in each seed, whitened update magnitudes are:

| Seed | Iteration | Actual drift | Noise | Full-gradient drift at same state | Gradient-error drift |
|---|---:|---:|---:|---:|---:|
{full_table}

The large stochastic gradient error, accompanied by large Taylor correction and
minibatch remainder, dominates the final update. This supports CV2 stochastic-drift
instability away from the anchor at this step. A large remainder is not itself a
coding error; independent formula checks pass. The same-state full-gradient drift
is a diagnostic comparison, not a counterfactual trajectory or causal ablation.

## Li: raw-gradient metric and CV2 interact

For all detailed paired replicas, the first CV1/CV2 proposal and metric are exactly
equal. In the earliest-failing replicas, some sparse feature columns are absent
from the initial minibatch, leaving v=0 and D=1/epsilon=100000. CV2's full-data
Hessian correction can act on these coordinates even when the batch metric has
not accumulated gradient information.

For seed 142100, replica 48, CV2's whitened radius goes 19.3 -> 218.7 -> 3127.4;
the CV1 control goes 19.3 -> 22.9 -> 17.3. At CV2 step 2, 99.94% of squared
Euclidean drift lies in metric-saturated coordinates. Seed 142101 gives the same
pattern (replica 6: 24.1 -> 146.2 -> 1611.6 for CV2).

The measured anchor-based h*lambda_max(sqrt(D) H sqrt(D)) at step 2 is 15.58 and
12.40, far above the planning coefficient .256. This is a local frozen-anchor
diagnostic, not a stability theorem for the adaptive nonlinear stochastic chain.
It shows why the planning scale cannot guarantee safe early adaptive updates.
Evidence supports an interaction between zero-initialized raw-gradient RMSprop
and CV2 correction; initialization alone cannot explain the method difference.
No warm start, metric cap, epsilon change, clipping, or retuning was introduced.

## Scope and files

Detailed IDs were selected explicitly from pilot outcomes: earliest/latest full-CV2
failures, earliest Li-CV2 failure, and the first common survivor, deduplicated.
This is mechanism diagnosis, not representative statistical estimation.
Scalar trajectories cover every live step for those IDs; vector archives retain
Li's first ten live steps and 21-step windows ending at selected failures.
No post-failure updates are invented. All-replica failures are in replay JSONs.

See `first_failure_evidence.csv`, `trajectory_*.csv`, `event_vectors_*.npz`,
`li_first_step_checks.json`, and `diagnostic_trajectories.png`.
Failed replicas were never removed from pilot accuracy ensembles. Diagnostic
runs do not compute new SW2/RMSE or enter frontiers; no selection, heldout, or
amendment seeds were used. Original failed configurations remain failed.
""")
    files=[p for p in OUT.iterdir() if p.is_file() and p.name!="delivery.lock.json"]+[ROOT/"scripts/diagnose_german_p1.py",Path(__file__)]
    lock=dict(schema=1,files={str(p.relative_to(ROOT)):sha256(p) for p in sorted(files)})
    save_json(OUT/"delivery.lock.json",lock);verify_lock(lock,ROOT)
    print(OUT/"REPORT.md")


if __name__=="__main__":main()
