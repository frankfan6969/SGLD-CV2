"""Summarize the CV1 chain/horizon and CV2 half-step diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np

from sgld_cv.workflow import save_json


def run(config_path, folder):
    config = json.loads(Path(config_path).read_text())
    folder = Path(folder)
    seeds = config["seeds"]
    formal = Path(config["source_heldout"])
    ref = json.loads((Path(config["source_reference"]) /
                      "reference_diagnostics.json").read_text())
    with np.load(Path(config["source_pilot"]) / "geometry.npz") as data:
        whitening = np.linalg.cholesky(data["hessian"])
    target_mean, target_cov = np.asarray(ref["reference_mean"]),np.asarray(ref["reference_covariance"])
    rng = np.random.default_rng(3709)

    def summarize(label, k):
        records = []
        states = []
        for seed in seeds:
            record = json.loads((folder / f"result_{label}_seed{seed}.json").read_text())
            records.append(next(x for x in record["checkpoints"] if x["iteration"] == k))
            with np.load(folder / f"states_{label}_seed{seed}.npz") as data:
                index = list(data["checkpoints"]).index(k)
                states.append(data["states"][index])
        x = np.concatenate(states,axis=0)
        mean_error = float(np.linalg.norm((x.mean(axis=0)-target_mean) @ whitening) /
            np.sqrt(x.shape[1]) + ref["standardized_mean_difference"])
        cov_error = float(np.linalg.norm(whitening.T @
            (np.cov(x,rowvar=False,ddof=0)-target_cov) @ whitening) /
            x.shape[1] + ref["standardized_covariance_difference"])
        return {"mean_w2":float(np.mean([v["w2"] for v in records])),
            "seed_w2":[v["w2"] for v in records],
            "pooled_guarded_mean_error":mean_error,
            "pooled_guarded_covariance_error":cov_error,
            "mean_total_seconds":float(np.mean([v["cost"]["total_seconds"] for v in records]))}

    cv1 = {str(k):summarize("cv1_extended",k)
           for k in config["cv1"]["replica_checkpoints"] if k}
    cv2_half = {str(k):summarize("cv2_half",k)
                for k in config["cv2"]["replica_checkpoints"] if k}
    cv2_full = json.loads((formal / "summary.json").read_text())["methods"]["cv2"]
    cv2_full_k8192 = {"mean_w2":cv2_full["mean_w2"],
        "seed_w2":cv2_full["seed_w2"],
        "pooled_guarded_mean_error":cv2_full["pooled_guarded_mean_error"],
        "pooled_guarded_covariance_error":cv2_full["pooled_guarded_covariance_error"],
        "mean_total_seconds":cv2_full["mean_total_seconds"]}

    def pair(a,b):
        values = np.asarray(a["seed_w2"])-np.asarray(b["seed_w2"])
        draws = rng.choice(values,size=(config["bootstrap"]["seed_resamples"],len(values)),
                           replace=True).mean(axis=1)
        return {"mean_w2_difference":float(values.mean()),
            "seed_differences":values,
            "bootstrap_q025":float(np.quantile(draws,.025)),
            "bootstrap_q975":float(np.quantile(draws,.975)),
            "mean_cost_ratio":a["mean_total_seconds"]/b["mean_total_seconds"]}

    comparisons = {
        "cv1_8192_to_16384":pair(cv1["16384"],cv1["8192"]),
        "cv1_16384_to_32768":pair(cv1["32768"],cv1["16384"]),
        "cv1_8192_to_32768":pair(cv1["32768"],cv1["8192"]),
        "cv2_half_vs_full_at_8192":pair(cv2_half["8192"],cv2_full_k8192),
        "cv2_half_16384_vs_full_8192_matched_hK":pair(cv2_half["16384"],cv2_full_k8192)}
    cv1_chain = json.loads(Path("results/logistic_cv1_chain_extension/summary.json").read_text())
    cv2_chain = json.loads(Path("results/logistic_cv2_half_chain_diagnostic/summary.json").read_text())
    extended_path = Path("results/logistic_cv2_half_chain_extended/summary.json")
    cv2_chain_extended = json.loads(extended_path.read_text()) if extended_path.exists() else None
    if cv2_chain_extended is not None:
        first = cv2_chain_extended["retained_prefixes"]["120000"]
        original = cv2_chain["retained_prefixes"]["120000"]
        if first != original:
            raise ValueError("extended CV2 half-step chain failed to reproduce 120k prefix")
    result = {"protocol":config,"cv1_replica":cv1,
        "cv1_chain":cv1_chain,"cv2_half_replica":cv2_half,
        "cv2_full_8192":cv2_full_k8192,"cv2_half_chain":cv2_chain,
        "cv2_half_chain_extended":cv2_chain_extended,
        "paired_comparisons":comparisons,
        "interpretation":"Post-held-out sensitivity only; all seeds were already inspected. Bootstrap is conditional on fixed HMC pools and is not fresh confirmation of a changed step size."}
    save_json(folder / "summary.json",result)
    lines=["# CV1 length and CV2 half-step diagnostics", "",
        "Exploratory follow-up to the completed eight-seed fixed-K experiment. These same seeds are not a new held-out evaluation.","",
        "## CV1 at unchanged step size", "",
        "| Replica K | Mean exact W2 | Pooled guarded mean/cov error | Mean ensemble seconds |",
        "| ---: | ---: | ---: | ---: |"]
    for k,v in cv1.items():
        lines.append(f"| {int(k):,} | {v['mean_w2']:.4f} | {v['pooled_guarded_mean_error']:.4f} / {v['pooled_guarded_covariance_error']:.4f} | {v['mean_total_seconds']:.1f} |")
    lines += ["", "The separate four-chain diagnostic reuses the failing seed and extends retained draws without changing h:","",
        "| Retained draws/chain | Max R-hat | Min bulk ESS | Guarded mean/cov error |",
        "| ---: | ---: | ---: | ---: |"]
    for k,v in cv1_chain["retained_prefixes"].items():
        d=v["diagnostics"]
        lines.append(f"| {int(k):,} | {d['max_rhat']:.4f} | {d['min_bulk_ess']:.0f} | {v['guarded_mean_error']:.4f} / {v['guarded_covariance_error']:.4f} |")
    lines += ["", "## CV2 at half step size", "",
        "| Setting | Mean exact W2 | Pooled guarded mean/cov error | Mean ensemble seconds |",
        "| --- | ---: | ---: | ---: |"]
    for label,v in [("full h, K=8,192",cv2_full_k8192),
                    ("h/2, K=8,192",cv2_half["8192"]),
                    ("h/2, K=16,384",cv2_half["16384"])]:
        lines.append(f"| {label} | {v['mean_w2']:.4f} | {v['pooled_guarded_mean_error']:.4f} / {v['pooled_guarded_covariance_error']:.4f} | {v['mean_total_seconds']:.1f} |")
    lines += ["", "Paired W2 changes (negative favors the first setting):", "",
        "| Contrast | Mean difference | 95% seed-bootstrap interval | Mean cost ratio |",
        "| --- | ---: | ---: | ---: |"]
    for name,v in comparisons.items():
        lines.append(f"| {name} | {v['mean_w2_difference']:+.4f} | [{v['bootstrap_q025']:+.4f}, {v['bootstrap_q975']:+.4f}] | {v['mean_cost_ratio']:.2f} |")
    lines += ["", "CV2 four-chain diagnostic at the same chain seed:", "",
        "| Step and retained draws/chain | Max R-hat | Min bulk ESS | Guarded mean/cov error |",
        "| --- | ---: | ---: | ---: |"]
    full = cv2_chain["full_step_comparison"]
    lines.append(f"| full h, {full['draws_per_chain']:,} | {full['diagnostics']['max_rhat']:.4f} | {full['diagnostics']['min_bulk_ess']:.0f} | {full['guarded_mean_error']:.4f} / {full['guarded_covariance_error']:.4f} |")
    prefixes = dict(cv2_chain["retained_prefixes"])
    if cv2_chain_extended is not None:
        prefixes.update(cv2_chain_extended["retained_prefixes"])
    for k,v in sorted(prefixes.items(),key=lambda pair:int(pair[0])):
        d=v["diagnostics"]
        lines.append(f"| h/2, {int(k):,} | {d['max_rhat']:.4f} | {d['min_bulk_ess']:.0f} | {v['guarded_mean_error']:.4f} / {v['guarded_covariance_error']:.4f} |")
    lines += ["", "A smaller h is a sensitivity analysis; adopting it would require new held-out seeds.",""]
    (folder / "RESULTS.md").write_text("\n".join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,
        default=Path("configs/logistic_postheldout_diagnostics.json"))
    parser.add_argument("--folder",type=Path,
        default=Path("results/logistic_postheldout_replica_sensitivity"))
    args=parser.parse_args()
    run(args.config,args.folder)


if __name__ == "__main__":
    main()
