"""Common six-method runner for Gaussian, logistic, German and Year studies."""
import argparse
import json
import os
import tempfile
from pathlib import Path
from threadpoolctl import threadpool_limits
from sgld_cv.workflow import run_study

def main(expected_problem=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--frozen-selection",type=Path)
    a=p.parse_args()
    config=json.loads(a.config.read_text())
    if expected_problem is not None and config["problem"]["kind"] != expected_problem:
        p.error(f"this entry point requires problem.kind={expected_problem}")
    os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"cv2_matplotlib"))
    # Fair, deterministic BLAS threading; the protocol is sequential CPU.
    with threadpool_limits(limits=1):
        run_study(config,a.output,frozen_selection=a.frozen_selection)

if __name__ == "__main__":
    main()
