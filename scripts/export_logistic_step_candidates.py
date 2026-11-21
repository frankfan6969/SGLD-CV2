"""Export numerically checked 2D logistic steps without declaring a final horizon."""
import argparse
import json
from pathlib import Path

from sgld_cv.workflow import METHODS, save_json


def run(pilot, validation, output):
    pilot, validation, output = map(Path, (pilot, validation, output))
    old = json.loads((pilot / "frozen_selection.json").read_text())
    checked = json.loads(validation.read_text())
    if not checked["all_numerical_checks_pass"] or set(checked["results"]) != set(METHODS):
        raise ValueError("all six steps must pass fresh-seed numerical checks")
    selection = {}
    for method in METHODS:
        source = old["selection"][method]
        result = checked["results"][method]
        selection[method] = {"batch_size": source["batch_size"],
            "step_size": result["step_size"]}
    artifact = {k: v for k, v in old.items() if k not in
        {"selection", "criterion", "budget"}}
    artifact.update({"selection": selection,
        "criterion": "exploratory fresh-seed Rhat/ESS and guarded standardized-moment gate",
        "budget": "method-specific diagnostic chain lengths; no equal-cost method ranking",
        "status": "diagnostic step candidates; final_horizon deliberately absent",
        "source_pilot_selection": str(pilot / "frozen_selection.json"),
        "source_validation": str(validation),
        "reference_guard": checked["reference_diagnostics"],
        "moment_threshold_each": checked["threshold_each"],
        "adaptive_ess_policy": checked["adaptive_ess_policy"],
        "horizon_probe": {"seeds": [124, 125],
            "note": "inspected retuned-horizon diagnostic seeds"}})
    save_json(output, artifact)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path,
        default=Path("results/stage456_validation/synthetic_logistic"))
    parser.add_argument("--validation", type=Path,
        default=Path("results/logistic_step_validation/validated_steps.json"))
    parser.add_argument("--output", type=Path,
        default=Path("results/logistic_step_validation/diagnostic_selection.json"))
    args = parser.parse_args()
    run(args.pilot, args.validation, args.output)


if __name__ == "__main__":
    main()
