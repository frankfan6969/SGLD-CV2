"""Freeze German P1 design and numerical grids; never execute sampler trajectories."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from sgld_cv.data import german_credit
from sgld_cv.protocol import sha256, verify_lock
from sgld_cv.transport import AccuracyMetric
from sgld_cv.workflow import save_json, save_csv

ROOT = Path(__file__).resolve().parents[1]
COEFFICIENTS = [.025, .05, .1, .2, .4, .8]
LI_COEFFICIENTS = [.00025, .001, .004, .016, .064, .256]


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


def curvature_scale(H, P):
    H, P = np.asarray(H), np.asarray(P)
    if H.ndim != 2 or H.shape[0] != H.shape[1] or P.shape != H.shape:
        raise ValueError("H and P must have matching square shapes")
    if not all(np.isfinite(x).all() and np.allclose(x, x.T) for x in (H, P)):
        raise ValueError("H and P must be finite and symmetric")
    np.linalg.cholesky(H)
    eigenvalues, vectors = np.linalg.eigh(P)
    if eigenvalues[0] <= 0:
        raise ValueError("P must be SPD")
    sqrt = (vectors*np.sqrt(eigenvalues))@vectors.T
    return float(np.linalg.eigvalsh(sqrt@H@sqrt)[-1])


def raw_second_moment(gradients, batch_size):
    """Exact E[(batch mean gradient)^2] under uniform sampling without replacement."""
    g = np.asarray(gradients, dtype=float)
    if g.ndim != 2 or len(g) < 2 or not 1 <= batch_size <= len(g) or not np.isfinite(g).all():
        raise ValueError("invalid gradient matrix or batch size")
    N = len(g)
    mean = g.mean(axis=0)
    return mean**2+(N-batch_size)/(batch_size*(N-1))*((g-mean)**2).mean(axis=0)


def make_candidates(config):
    candidates = []
    for method in ("cv1", "cv2", "ula"):
        for i, (c, h) in enumerate(zip(COEFFICIENTS, config["fixed_metrics"]["full"]["h_grid"])):
            candidates.append({"id": f"full_{method}_c{i}", "arm": f"full_{method}", "method": method,
                "geometry": "full", "coefficient": c, "h": h, "legacy_step_size": 2*h,
                "batch_size": None if method == "ula" else 32,
                "full_gradient_rows_per_step": config["target"]["N"] if method == "ula" else None,
                "alpha": None, "epsilon": None})
    li = config["li2016"]["planning_by_batch"]["32"]
    for method in ("sgld", "cv1", "cv2"):
        arm = "psgld" if method == "sgld" else "p"+method
        for alpha, coefficients in ((.99, LI_COEFFICIENTS), (.9, [.016, .064])):
            for i,c in enumerate(coefficients):
                h = c/li["L_planning"]
                candidates.append({"id": f"{arm}_a{alpha}_c{c}", "arm": arm, "method": method,
                    "geometry": "li2016", "coefficient": c, "h": h, "legacy_step_size": 2*h,
                    "batch_size": 32, "alpha": alpha, "epsilon": 1e-5})
    for i,c in enumerate([.05,.2,.8]):
        h = c/config["fixed_metrics"]["identity"]["L_P"]
        candidates.append({"id": f"identity_sgld_c{i}", "arm": "identity_sgld", "method": "sgld",
            "geometry": "identity", "coefficient": c, "h": h, "legacy_step_size": 2*h,
            "batch_size": 32, "alpha": None, "epsilon": None})
    return candidates


def validate_config(c, geometry):
    if c["purpose"] != "P1 Configuration Freeze — German Credit" or c["schema_version"] != 1:
        raise ValueError("wrong P1 configuration identity")
    if c["pilot_execution_started"] or c["stage"] != "configuration_frozen":
        raise ValueError("configuration freeze must not execute pilot")
    all_seeds = []
    for name, seeds in c["seeds"].items():
        if not seeds or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError(f"invalid seed role: {name}")
        all_seeds.extend(seeds)
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("seed roles overlap")
    H = geometry["H"]
    for kind in ("identity", "diagonal", "full"):
        spec = c["fixed_metrics"][kind]
        P = geometry["P_"+kind]
        actual = curvature_scale(H, P)
        if not np.isclose(actual, spec["L_P"], rtol=1e-12, atol=1e-12):
            raise ValueError("stored L_P differs from frozen matrices")
        if not np.allclose(spec["h_grid"], np.asarray(COEFFICIENTS)/actual, rtol=1e-12, atol=1e-15):
            raise ValueError("h grid is not c/L_P")
        if array_hash(P) != spec["P_sha256"]:
            raise ValueError("metric hash mismatch")
    expected = make_candidates(c)
    if c["pilot_candidates"] != expected:
        raise ValueError("candidate table differs from resolved grid")
    if len(expected) != 45 or len({x["id"] for x in expected}) != 45:
        raise ValueError("pilot candidate count/uniqueness mismatch")
    if c["li2016"]["initial_v"] != "zeros independently per replica; planning second moment is NOT used as initial state":
        raise ValueError("Li initialization changed")
    for b, spec in c["li2016"]["planning_by_batch"].items():
        second = geometry["raw_second_b"+b]
        D = 1/(1e-5+np.sqrt(second))
        actual = curvature_scale(H, np.diag(D))
        if not np.isclose(actual, spec["L_planning"], rtol=1e-12):
            raise ValueError("Li planning scale mismatch")
        if not np.allclose(spec["h_grid_default"], np.asarray(LI_COEFFICIENTS)/actual):
            raise ValueError("Li planning grid mismatch")
    pilot = c["budgets"]["pilot"]
    if pilot["maximum_candidate_seed_runs"] != len(expected)*len(c["seeds"]["pilot"]):
        raise ValueError("pilot run cap does not match candidate grid")
    for phase in ("pilot", "selection", "heldout"):
        spec = c["budgets"][phase]
        if sorted(set(spec["iteration_checkpoints"])) != spec["iteration_checkpoints"] or spec["replicas"] < 2:
            raise ValueError("invalid phase checkpoints/replicas")
    if c["configuration_version"] >= 2 and c["budgets"]["selection"]["replicas"] != c["budgets"]["heldout"]["replicas"]:
        raise ValueError("selection and heldout must use the same empirical SW2 replica count")
    if c["uncertainty"]["matched_noise_gap_ci_level"] != 1-.05/8:
        raise ValueError("primary matched-cell multiplicity rule changed")
    if c["cost"]["profile_components_in_frontier_runs"]:
        raise ValueError("profiling must be separate from main wall-clock runs")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/p1_german_frozen_v2.json")
    parser.add_argument("--output", type=Path, default=ROOT/"results/p1_german_configuration_v2")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    args.config, args.output = args.config.resolve(), args.output.resolve()
    if args.verify:
        verify_lock(json.loads((args.output/"configuration.lock.json").read_text()), ROOT)
        c = json.loads(args.config.read_text())
        with np.load(args.output/"geometry_and_li_planning.npz") as saved:
            validate_config(c, {k:saved[k] for k in saved.files})
        print("German P1 frozen configuration and numerical grids verified; no pilot run.")
        return
    if args.config.exists() or args.output.exists():
        raise FileExistsError("freeze requires new config and output paths")
    start = perf_counter()
    p0 = ROOT/"results/p0_validation_v2"
    verify_lock(json.loads((p0/"delivery.lock.json").read_text()), ROOT)
    source = p0/"german"
    with np.load(source/"frozen_assets.npz") as saved:
        a = {k:saved[k] for k in saved.files}
    reference_diagnostics = json.loads((source/"reference_diagnostics.json").read_text())
    for diagnostic in (reference_diagnostics["first"], reference_diagnostics["longer"]):
        if not (diagnostic["max_rhat"] < 1.01 and diagnostic["min_bulk_ess"] >= 400 and
                diagnostic["min_tail_ess"] >= 400 and sum(diagnostic["divergences"]) == 0 and
                min(diagnostic["acceptance"]) > .5):
            raise ValueError("reference chain quality gate failed")
    if not reference_diagnostics["agreement"]["reliable"]:
        raise ValueError("independent reference agreement failed")
    fmeta = json.loads((source/"functional_reference.json").read_text())
    ref_quality = {"reliable": bool(reference_diagnostics["reliable"]),
        "first_max_rhat": reference_diagnostics["first"]["max_rhat"],
        "second_max_rhat": reference_diagnostics["longer"]["max_rhat"],
        "first_min_bulk_ess": reference_diagnostics["first"]["min_bulk_ess"],
        "second_min_bulk_ess": reference_diagnostics["longer"]["min_bulk_ess"],
        "first_min_tail_ess": reference_diagnostics["first"]["min_tail_ess"],
        "second_min_tail_ess": reference_diagnostics["longer"]["min_tail_ess"],
        "scoring_max_normalized_functional_mcse": float(np.max(np.asarray(fmeta["functional_mcse"]["scoring"])/a["function_scales"])),
        "calibration_max_normalized_functional_mcse": float(np.max(np.asarray(fmeta["functional_mcse"]["calibration"])/a["function_scales"])),
        "max_normalized_reference_disagreement": float(np.max(np.abs(fmeta["independent_reference_disagreement_normalized"])))}
    if not (ref_quality["reliable"] and ref_quality["scoring_max_normalized_functional_mcse"] <= .02 and
            ref_quality["calibration_max_normalized_functional_mcse"] <= .025 and
            ref_quality["max_normalized_reference_disagreement"] <= .1):
        raise ValueError("reference does not satisfy proposed P1 resolution gates")
    with threadpool_limits(limits=1):
        model, _ = german_credit(ROOT/"data/raw/german.zip")
        np.testing.assert_array_equal(model.X, a["X"])
        np.testing.assert_array_equal(model.y, a["y"])
        H = a["hessian"]
        delta = 1e-10*model.prior_precision
        Pfull = np.linalg.solve(H+delta*np.eye(model.d), np.eye(model.d)); Pfull=(Pfull+Pfull.T)/2
        geometry = {"H":H, "anchor":a["anchor"], "P_identity":np.eye(model.d),
                    "P_diagonal":np.diag(1/(np.diag(H)+delta)), "P_full":Pfull}
        fixed = {}
        for kind in ("identity", "diagonal", "full"):
            P=geometry["P_"+kind]; LP=curvature_scale(H,P)
            fixed[kind]={"L_P":LP, "coefficients":COEFFICIENTS, "h_grid":[c/LP for c in COEFFICIENTS],
                "legacy_step_size_grid":[2*c/LP for c in COEFFICIENTS], "P_sha256":array_hash(P),
                "scope":"anchor curvature scale only, not a global stochastic stability bound",
                "P_definition": {"identity":"I", "diagonal":"inverse diag(H+delta I)", "full":"inverse(H+delta I)"}[kind]}
        gradients=model.data_gradients(a["anchor"], np.arange(model.N))
        planning={}
        for b in (4,32,128):
            moment=raw_second_moment(gradients,b)
            D=1/(1e-5+np.sqrt(moment)); LP=curvature_scale(H,np.diag(D))
            geometry["raw_second_b"+str(b)]=moment
            planning[str(b)]={"L_planning":LP,"second_moment_sha256":array_hash(moment),
                "h_grid_default":[c/LP for c in LI_COEFFICIENTS],"coefficients":LI_COEFFICIENTS}
        metric=AccuracyMetric("standardized_sliced_w2", a["anchor"], a["transform"], a["projections"])
        first=a["reference_calibration"].reshape(-1,model.d); second=a["reference_scoring"].reshape(-1,model.d)
        rng=np.random.default_rng(143600)
        baselines=[metric.distance(first[rng.choice(len(first),512,replace=False)],
                    second[rng.choice(len(second),1024,replace=False)]) for _ in range(3)]
    c={
      "schema_version":1, "purpose":"P1 Configuration Freeze — German Credit", "stage":"configuration_frozen",
      "configuration_version":2,"supersedes":"configs/p1_german_frozen_v1.json; selection and heldout now use the same R=512",
      "pilot_execution_started":False,
      "research_question":"At matched h,T, is CV1 materially less accurate than full-gradient ULA; can CV2 reduce that gap at reasonable total cost?",
      "transition":{"public":"theta_next=theta-h*P*g+sqrt(2*h)*chol(P)*z","legacy_step_size":"2*h","dtype":"float64","blas_threads":1},
      "target":{"dataset":"German Credit","N":model.N,"d":model.d,"prior_precision":model.prior_precision,
                "intercept":False,"archive":"data/raw/german.zip","assets_source":"results/p0_validation_v2/german",
                "anchor_sha256":array_hash(a["anchor"]),"H_sha256":array_hash(H),"ridge_shift":delta},
      "fixed_metrics":fixed,
      "li2016":{"raw_input":"unscaled batch mean likelihood gradient, excluding prior and CV correction",
         "initial_v":"zeros independently per replica; planning second moment is NOT used as initial state",
         "update":"v=alpha*v+(1-alpha)*raw_mean**2; P_t=diag(1/(epsilon+sqrt(v)))",
         "Gamma":"omitted; empirical adaptive baseline; no fixed-P theorem guarantee",
         "default":{"alpha":.99,"epsilon":1e-5},"bounded_alternative":{"alpha":.9,"epsilon":1e-5,"coefficients":[.016,.064]},
         "planning_formula":"E[raw_mean_B^2]=mean(g_i)^2+(N-b)/(b*(N-1))*mean((g_i-mean(g_i))^2)",
         "planning_scope":"anchor expectation only; alpha-independent expectation scale, not actual v_t or stability guarantee",
         "planning_by_batch":planning,"no_full_alpha_epsilon_factorial":True},
      "seeds":{"pilot":[142100,142101],"pilot_amendment_reserved":[142102,142103],
               "selection":[142200,142201,142202,142203],"heldout":list(range(142300,142308)),
               "reference_existing":[7400,7401],"reference_resolution_only":[143600],
               "gradient_audit":[143500,143501],"uncertainty":[143900,143901],"timing_order":[143700],
               "single_chain_timing":list(range(143800,143805))},
      "budgets":{
        "pilot":{"replicas":64,"iteration_checkpoints":[128,512,2048],"maximum_candidate_seed_runs":90,
                 "max_candidate_sampler_seconds":120,"max_stage_elapsed_seconds":3600,
                 "execution":"one trajectory to max T per candidate/seed; reuse prefixes, do not rerun every T"},
        "selection":{"replicas":512,"iteration_checkpoints":[128,512,2048,8192],"max_stage_elapsed_seconds":14400,
                 "max_primary_configs_per_arm":3,"max_li_configs_per_arm":2,"max_identity_sgld_configs":2,
                 "max_candidate_sampler_seconds":600,"maximum_candidate_seed_runs":68},
        "heldout":{"replicas":512,"iteration_checkpoints":[128,512,2048,8192],"max_stage_elapsed_seconds":43200,
                 "max_primary_configs_per_arm":3,"max_li_configs_per_arm":2,"max_identity_sgld_configs":2,
                 "max_candidate_sampler_seconds":1800,"maximum_candidate_seed_runs":136},
        "budget_exhaustion":"right-censor unreached checkpoints; retain last complete ensemble checkpoint; never replace failed/slow seeds",
        "evaluation_time":"record separately; stage elapsed cap includes evaluation and serialization; selected-config cost excludes evaluation"},
      "batch_sizes":{"main":32,"pilot_sampler_sweep":False,"limited_gradient_MSE_check":[4,32,128],
                 "audit_states":32,"fresh_batches_per_state":128,"ULA":"full N=1000; no stochastic batch parameter"},
      "primary_arms":["full_cv1","full_cv2","full_ula"],
      "practical_arms":["identity_sgld","psgld","pcv1","pcv2"],
      "deferred_ablations":["fixed-diagonal CV1/CV2: P2/P4","identity CV1/CV2: P2 or limited amended P1 checks","MALA: optional"],
      "accuracy":{"distribution":"replica endpoint standardized sliced-W2; 512 frozen projections",
         "reference_size":1024,"init":"all replicas at the identical frozen anchor; independent batch/noise streams",
         "functionals":"reuse P0 frozen definitions/scales; online averages over steps 1..T, burn_in=0",
         "primary_functional":"mean coordinate-variance NRMSE (61 equally weighted coordinates)",
         "secondary_functionals":["coordinate means","slow-direction mean/variance","eight prediction probabilities"],
         "sw2_target_rule":"reference-only baseline median plus [.01,.03,.07,.15]; not a floor or subtraction",
         "sw2_accuracy_targets":[float(np.median(baselines)+v) for v in [.01,.03,.07,.15]],
         "variance_NRMSE_targets":[.05,.1,.2,.4],"joint_success":"both endpoints meet the specified target pair; report marginal frontiers too"},
      "matched_noise_gap":{"coefficients":[.1,.4],"iterations":[2048,8192],"maximum_confirmatory_cells":4,
         "contrasts":{"noise":"CV1 minus ULA","recovery":"CV1 minus CV2","remaining":"CV2 minus ULA"},
         "material_margins":{"sw2":.01,"variance_NRMSE":.025},
         "substantial":"multiplicity-adjusted lower CI exceeds margin in >=1 primary endpoint, with no confirmed opposite material effect in the other",
         "negligible":"adjusted CIs wholly within +/- margins for BOTH endpoints at each tested cell; scope limited to these cells",
         "otherwise":"inconclusive, mixed, failed or budget-censored; absence of significance is not equivalence",
         "interpretation":"differences in distance-to-posterior are contrasts, not additive noise/error decomposition",
         "reference_sensitivity":"repeat accepted conclusions using independent reference pool; flip or unresolved reference error => inconclusive"},
      "screening_and_selection":{"primary":"retain mandatory c=.1,.4 plus at most one extra pilot step per arm",
         "extra_step_rule":"maximize improvement in attained target-specific minimum measured cost across pilot prefixes; ties: lower final variance-NRMSE, lower SW2, lower cost, smaller h",
         "practical":"retain at most two configurations per arm by the same target-cost coverage rule; no extra hyperparameter factorial",
         "selection_stage":"evaluate shortlisted configs on selection seeds; lock final per-target optimal-cost choices and all matched cells before heldout",
         "ties":"equal metrics/cost: smaller h, then shorter T, then candidate ID lexicographically",
         "frontier":"finite-grid nondominated configurations; no interpolation, extrapolation or heldout reoptimization",
         "pilot_resolution":"R=64 descriptive screening only; no formal R=512 target attainment or noise-gap inference from pilot",
         "cost_advantage":"at least 10% lower selected-config total cost at matched accuracy, supported by paired uncertainty and reference sensitivity",
         "no_candidate_feasible":"report unattained target, do not expand on selection or heldout outcomes"},
      "uncertainty":{"level_pointwise":.95,"bootstrap_repeats":1999,"matched_noise_gap_ci_level":1-.05/8,
         "multiplicity":"Bonferroni: four prespecified matched cells x two primary endpoints; additional contrasts exploratory pointwise",
         "unit":"whole independent replica paths; paired method IDs; outer algorithm-seed clusters and inner paired replicas",
         "reference":"conditional frozen-pool intervals; no iid resampling of correlated HMC draws; separate MCSE/sensitivity",
         "pilot":"descriptive screening only; no formal superiority claims","heldout_algorithm_seeds":8},
      "reference_quality":{"requirements":{"max_rhat_exclusive":1.01,"min_bulk_ess":400,"min_tail_ess":400,
          "max_divergences":0,"min_acceptance_exclusive":.5,"independent_agreement_required":True,
          "scoring_max_normalized_functional_mcse":.02,"calibration_max_normalized_functional_mcse":.025,
          "max_normalized_functional_disagreement":.1},"observed":ref_quality,
          "reference_only_baselines_R512_S1024":baselines,"limitations":"approximate correlated HMC; baseline is not a lower bound"},
      "failure":{"transition":"nonfinite or whitened radius >100*sqrt(d); stop replica permanently, no clipping/refresh/replacement",
         "checkpoint":"any failed replica invalidates whole corresponding ensemble; no survivor-only scores/CI",
         "aggregate":"failed seed invalidates relevant contrast; preserve all failures and divergence counts",
         "mixing":"separate overdispersed diagnostics; Rhat>=1.01 or min bulk/tail ESS<400 => mixing unverified, not proof endpoint invalid",
         "reference":"failed quality/lock => stop before sampler execution","budget":"right-censored, not numerical failure"},
      "diagnostics":{"selection_finalists_only":True,"chains":4,"initialization":"shared 2x mode-Hessian Gaussian dispersion",
         "warmup":1000,"draws":4000,"seed_derivation":"algorithm seed + 100000, distinct INITIALIZATION stream",
         "adaptive_ESS":"exploratory only","diagnostic_cost":"separate from endpoint/estimator cost, counted in stage research elapsed"},
      "cost":{"primary":"setup-inclusive sequential-ensemble wall time; setup once per collection",
         "single_chain":"separately measure R=1 repetitions for selected estimator configs; never total_ensemble/R",
         "single_chain_timing_repeats":5,"single_chain_research_cap_seconds":1800,
         "profile_components_in_frontier_runs":False,"separate_profiles":"representative configurations only",
         "itemize":["anchor","CV gradient cache","CV2 Hessian cache","metric Hessian/reuse","SPD validation","inverse solve","noise Cholesky","gradient work","dense correction","drift/noise","adaptive init/update"],
         "tuning":"record grid-planning, pilot and selection search work per method; report search-inclusive and selected-config costs separately",
         "shared_design_cost":"reference-only grid construction is a measured research/design cost, not free per-method preprocessing",
         "no_gradient_equivalent_conversion":True},
      "amendments":{"maximum_pilot_amendments":1,"trigger":"no eligible stable candidate or optimum hits grid boundary in BOTH pilot seeds",
         "scope":"one factor-of-two adjacent step per affected arm; max two extra candidates per arm; use reserved amendment seeds only",
         "additional_elapsed_cap_seconds":1800,"requirement":"write a new immutable protocol amendment before executing; no automatic expansion",
         "selection_or_heldout":"no expansion or retuning; new study/version required"}}
    c["pilot_candidates"]=make_candidates(c)
    # Actual output filenames identify consumed algorithm seeds; reserved P0 roles are allowed.
    consumed=set()
    for path in (ROOT/"results").rglob("*"):
        consumed.update(int(s) for s in re.findall(r"seed(\d+)",path.name))
        if path.name == "manifest.json":
            meta=json.loads(path.read_text())
            consumed.update(meta.get("validation_seeds_consumed", []))
    for phase in ("pilot","pilot_amendment_reserved","selection","heldout"):
        if consumed.intersection(c["seeds"][phase]):
            raise ValueError(f"{phase} seeds already consumed")
    validate_config(c,geometry)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/"geometry_and_li_planning.npz",**geometry)
    save_json(args.config,c)
    save_csv(args.output/"pilot_candidates.csv",c["pilot_candidates"])
    save_json(args.output/"freeze_validation.json",{"status":"passed","pilot_execution_started":False,
        "pilot_candidates":45,"pilot_candidate_seed_runs":90,"all_h_grids_resolved":True,
        "seed_consumption_check":"pilot/selection/heldout disjoint and no saved seed outputs",
        "grid_planning_seconds":perf_counter()-start,"planning_observation_gradients":model.N,
        "L_P":{k:v["L_P"] for k,v in fixed.items()}})
    files=[args.config,Path(__file__),args.output/"geometry_and_li_planning.npz",
        ROOT/"P1_GERMAN_CONFIGURATION.md",ROOT/"P0_THEORY_AUDIT.md",ROOT/"configs/p0_theory_claims.json",ROOT/"SGLD_CV2_Updated_Experimental_Plan.md",
        ROOT/"data/raw/german.zip",source/"frozen_assets.npz",source/"functional_reference.json",source/"reference_diagnostics.json"]
    files+=sorted((ROOT/"src/sgld_cv").glob("*.py"))
    lock={"schema":1,"files":{str(p.relative_to(ROOT)):sha256(p) for p in files}}
    save_json(args.output/"configuration.lock.json",lock)
    verify_lock(lock,ROOT)
    print("Frozen German P1 configuration; no sampler trajectories executed.")
    print("L_P:",{k:v["L_P"] for k,v in fixed.items()})


if __name__=="__main__":
    main()
