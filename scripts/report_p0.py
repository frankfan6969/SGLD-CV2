"""Rebuild the revised-plan P0 completion report from frozen saved evidence."""
import argparse
import csv
import json
from pathlib import Path

from sgld_cv.protocol import verify_lock, sha256
from sgld_cv.uncertainty import cost_views
from sgld_cv.workflow import save_csv, save_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT/"results/p0_validation_v2")
    args = parser.parse_args()
    out = args.results
    manifest = json.loads((out/"manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("cannot report incomplete P0 as complete")
    verify_lock(json.loads((out/"protocol.lock.json").read_text()), ROOT)
    verify_lock(json.loads((ROOT/"results/phase1_validation_final/snapshot.lock.json").read_text()), ROOT)
    rows = []
    max_path_error = 0.
    for target in manifest["targets"]:
        folder = out/target
        for check in json.loads((folder/"checks.json").read_text()):
            max_path_error = max(max_path_error, check.get("cv2_ula_max_path_error", 0.))
        for path in sorted(folder.glob("cost_*.json")):
            record = json.loads(path.read_text())
            snapshot = record["snapshots"][-1]
            seconds, operations = snapshot["seconds"], snapshot["operations"]
            row = {"target": target, "run": path.stem[5:], **cost_views(snapshot),
                "scope": "P0 profiled sequential ensemble; no parameter tuning", "tuning_seconds": 0.,
                "anchor_seconds": sum(t for p,t in seconds.items() if p.startswith("anchor")),
                "cache_gradient_seconds": seconds.get("cv_anchor_gradient", 0.),
                "cache_hessian_seconds": seconds.get("cv_anchor_hessian", 0.),
                "metric_hessian_seconds": seconds.get("fixed_metric_hessian", 0.),
                "metric_cache_reuse_seconds": seconds.get("fixed_metric_hessian_reuse", 0.),
                "metric_regularization_spd_seconds": seconds.get("fixed_metric_regularization", 0.),
                "metric_inverse_seconds": seconds.get("fixed_metric_inverse", 0.),
                "noise_factorization_seconds": seconds.get("full_metric_factorization", 0.),
                "adaptive_initialization_seconds": seconds.get("adaptive_state_initialization", 0.),
                "gradient_component_seconds": snapshot.get("component_seconds", {}).get("gradient", 0.),
                "adaptive_update_component_seconds": snapshot.get("component_seconds", {}).get("adaptive_update", 0.),
                "drift_noise_component_seconds": snapshot.get("component_seconds", {}).get("drift_and_noise", 0.),
                "component_timers_are_nested": True, "cache_bytes": record["metadata"]["cache_bytes_excluding_model"]}
            row.pop("component_seconds_nested_not_additive")
            row.update({"count_"+key: value for key,value in operations.items()})
            rows.append(row)
    save_csv(out/"itemized_costs.csv", rows)
    targets = manifest["targets"]
    runs = sum(r["arms"]*r["seeds"] for r in targets.values())
    intervals = sum(r["uncertainty"]["individual_interval_rows"] for r in targets.values())
    contrasts = sum(r["uncertainty"]["paired_interval_rows"] for r in targets.values())
    table = "\n".join(f"| {t} | {r['N']} | {r['d']} | {r['arms']} | {r['seeds']} | {r['metric_li_check_count']} |"
                      for t,r in targets.items())
    report = f"""# P0 completion report — updated experimental plan

Date: 2026-10-14. Status: P0 completed for empirical experimental validity.
P1 has not been run; no pilot/selection/heldout performance seeds were consumed.

## Validated implementation and protocol

| Target | N | d | Arms | Validation seeds | Extra metric/Li checks |
|---|---:|---:|---:|---:|---:|
{table}

- {runs} target/arm/seed smoke runs, including fixed-diagonal CV1/CV2/ULA.
- Full CV2 Taylor correction retained under every sampler metric.
- Public h convention with explicit legacy step_size=2h adapter.
- Frozen prior-scaled metric ridge; SPD failures cannot trigger auto-retuning.
- Gaussian full/diagonal noise covariance and multistep law tests.
- Paired quadratic CV2/ULA maximum path discrepancy: {max_path_error:.3g}.
- Li state/cost checks reconstruct raw-gradient recurrence and repeated final
  state exactly for alpha .9/.99 and epsilon 1e-5/1e-3 at declared small steps.
- {intervals} per-arm pointwise interval rows and {contrasts} paired hierarchical
  contrast rows generated at the final checkpoint. Two-seed, 99-resample smoke
  intervals validate execution, not scientific significance or coverage.
- Setup, gradient, adaptive initialization/update, curvature, matrix and noise
  work have attributable costs. Nested component timers are not double-counted.
- Current frozen protocol/source/assets and historical v1 source snapshot verify.

## Theory audit outcome

`../../P0_THEORY_AUDIT.md` and `../../configs/p0_theory_claims.json` classify
verified algebra, conditional results and claims requiring revision. Key
restrictions are explicit fourth-moment assumptions, corrected W2 accumulation
interpretation, integer batch/iteration floors, and all Hessian/setup work.
Original manuscript theorem constants/proofs are not certified by this audit.
Unsupported complexity/superiority claims are quarantined for P5; P1 empirical
comparisons do not rely on them.

Constant full-H and diagonal metrics have a valid linear-coordinate
interpretation with transformed global constants. Li with omitted Gamma is
an empirical practical baseline, not a fixed-P corollary. Original YearMSD's
Gaussian-likelihood target remains P4 support, not a nonquadratic P2 main target.

## Outputs and limits

`itemized_costs.csv` separates selected-configuration total and search-inclusive
total. P0 tuning cost is zero because no performance-based parameter selection
was performed. Timings are profiled one-thread CPU sequential-ensemble costs,
not single-chain costs or publication frontiers. Reference/bootstrap/evaluation
work is external to sampler cost. Production frontier timing should disable
component profiling and report separately measured tuning effort.

Each target folder includes replica SW2/RMSE, gradient MSE, per-phase work,
adaptive final-state records, paired CI tables and metric/Li checks. Reference
assets and functional MCSE/disagreement are reused from verified v1 assets;
HMC rows remain correlated and are not iid-resampled. Pointwise percentile
intervals are conditional finite-cloud diagnostics, not simultaneous frontier
bands. Negative outcomes cannot be filtered from formal evaluation.

Next permitted work: freeze German P1 production grid/budgets/precision targets
on pilot seeds, then select/evaluate configurations with disjoint seeds. P0
does not establish CV2 efficiency superiority or resolve all publication proofs.
"""
    (out/"REPORT.md").write_text(report)
    save_json(out/"report_provenance.json", {"reporter_sha256": sha256(Path(__file__)),
        "manifest_sha256": sha256(out/"manifest.json"), "protocol_lock_sha256": sha256(out/"protocol.lock.json"),
        "itemized_cost_rows": len(rows)})
    print(f"P0 report rebuilt; current and historical snapshot locks verified: {out/'REPORT.md'}")


if __name__ == "__main__":
    main()
