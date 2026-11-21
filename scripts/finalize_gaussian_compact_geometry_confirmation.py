"""Check the frozen Gaussian factorial is complete and internally paired."""
import json
from pathlib import Path

import numpy as np

from sgld_cv.workflow import save_json


FOLDER = Path("results/gaussian_compact_geometry_confirmation")


def main():
    protocol = json.loads((FOLDER / "protocol.json").read_text())
    c = protocol["config"]
    total = 0
    for d in c["dimensions"]:
        for kappa in c["target_condition_numbers"]:
            fingerprints = set()
            for ds in c["design_seeds"]:
                target_id = f"d{d}_k{kappa}_design{ds}"
                target = json.loads((FOLDER / f"target_{target_id}.json").read_text())
                if not np.isclose(target["target"]["actual_kappa"], kappa, rtol=1e-7):
                    raise ValueError(f"target condition number changed: {target_id}")
                fingerprints.add(target["metric_fingerprint"])
                for ss in c["sampler_seeds"]:
                    baseline = json.loads((FOLDER / f"baseline_{target_id}_seed{ss}.json").read_text())
                    if not np.isfinite(baseline["exact_vs_exact_sw2"]):
                        raise ValueError(f"invalid exact baseline: {target_id} {ss}")
                    rows = []
                    for method in c["methods"]:
                        path = FOLDER / f"result_{target_id}_{method}_seed{ss}.json"
                        row = json.loads(path.read_text())
                        if (row["method"] != method or row["d"] != d or row["kappa"] != kappa or
                                row["design_seed"] != ds or row["sampler_seed"] != ss or
                                row["metric_fingerprint"] != target["metric_fingerprint"] or
                                row["batch_size"] != c["batch_size"] or
                                row["step_size"] != c["step_size"]):
                            raise ValueError(f"mismatched result: {path}")
                        if not np.isfinite([row["standardized_sw2_512_vs_1024"],
                                            row["standardized_mean_error"],
                                            row["standardized_covariance_error"],
                                            row["cost"]["total_seconds"]]).all():
                            raise ValueError(f"invalid result: {path}")
                        rows.append(row)
                        total += 1
                    if len({r["metric_fingerprint"] for r in rows}) != 1:
                        raise ValueError(f"unpaired score calibration: {target_id} {ss}")
            if len(fingerprints) != 1:
                raise ValueError(f"score calibration changed across design seeds in d={d}")
    expected = (len(c["dimensions"]) * len(c["target_condition_numbers"]) *
                len(c["design_seeds"]) * len(c["sampler_seeds"]) * len(c["methods"]))
    if total != expected or not (FOLDER / "summary.json").exists() or not (FOLDER / "mechanism.png").exists():
        raise ValueError("incomplete Gaussian confirmation")
    save_json(FOLDER / "finalization_status.json", {"phase": "complete",
        "completed": total, "expected": expected,
        "target_cells": len(c["dimensions"]) * len(c["target_condition_numbers"]),
        "design_seeds_per_cell": len(c["design_seeds"]),
        "sampler_seeds_per_design": len(c["sampler_seeds"]),
        "report": str(FOLDER / "RESULTS.md")})
    print("Gaussian compact geometry confirmation complete:", total, "runs", flush=True)


if __name__ == "__main__":
    main()
