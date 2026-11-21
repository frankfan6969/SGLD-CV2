"""Validate chains and publish German results after all frozen replica arms finish."""
import json
import subprocess
import sys
import time
from pathlib import Path

from sgld_cv.workflow import save_json


FOLDER = Path("results/german_frozen_geometry_confirmation")
STATUS = FOLDER / "finalization_status.json"


def command(script):
    subprocess.run([sys.executable, script], check=True)


def main():
    config = json.loads((FOLDER / "protocol.json").read_text())["config"]
    arms, seeds = config["arms"], config["method_seeds"]
    expected = len(arms) * len(seeds)
    deadline = time.monotonic() + 8 * 3600
    previous = -1
    try:
        while True:
            done = sum((FOLDER / f"result_{arm}_seed{seed}.json").exists() and
                       (FOLDER / f"states_{arm}_seed{seed}.npz").exists()
                       for arm in arms for seed in seeds)
            if done != previous:
                save_json(STATUS, {"phase": "replicas", "completed": done,
                                   "expected": expected})
                print(f"German replicas {done}/{expected}", flush=True)
                previous = done
            if done == expected:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("German replica run exceeded eight hours")
            time.sleep(30)
        save_json(STATUS, {"phase": "chains", "completed": expected,
                           "expected": expected})
        command("scripts/validate_german_frozen_geometry_chains.py")
        save_json(STATUS, {"phase": "report", "completed": expected,
                           "expected": expected})
        command("scripts/report_german_frozen_geometry_confirmation.py")
        save_json(STATUS, {"phase": "tests", "completed": expected,
                           "expected": expected})
        subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True)
        save_json(STATUS, {"phase": "complete", "completed": expected,
                           "expected": expected,
                           "report": str(FOLDER / "RESULTS.md")})
        print("German frozen geometry confirmation complete", flush=True)
    except Exception as error:
        save_json(STATUS, {"phase": "failed", "error": str(error),
                           "completed": previous, "expected": expected})
        raise


if __name__ == "__main__":
    main()
