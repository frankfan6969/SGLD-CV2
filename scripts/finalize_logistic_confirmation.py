"""Finish the frozen fresh-seed run and its prespecified CV2 extension."""
import json
import subprocess
import sys
import time
from pathlib import Path

from sgld_cv.workflow import save_json


FOLDER = Path("results/logistic_postdiagnostic_confirmation")
STATUS = FOLDER / "finalization_status.json"


def command(script):
    subprocess.run([sys.executable, script], check=True)


def main():
    config = json.loads((FOLDER / "protocol.json").read_text())["config"]
    seeds, methods = config["method_seeds"], config["methods"]
    expected = len(seeds) * len(methods)
    previous = -1
    deadline = time.monotonic() + 12 * 3600
    try:
        while True:
            done = sum((FOLDER / f"result_{m}_seed{s}.json").exists() and
                       (FOLDER / f"states_{m}_seed{s}.npz").exists()
                       for s in seeds for m in methods)
            if done != previous:
                save_json(STATUS, {"phase": "fixed_k", "completed": done,
                                   "expected": expected})
                print(f"fixed K {done}/{expected}", flush=True)
                previous = done
            if done == expected:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("fresh-seed fixed-K run exceeded 12 hours")
            time.sleep(30)
        save_json(STATUS, {"phase": "fixed_k_report", "completed": expected,
                           "expected": expected})
        subprocess.run([sys.executable, "scripts/report_logistic_fixed8192_heldout.py",
                        str(FOLDER)], check=True)
        save_json(STATUS, {"phase": "cv2_long", "completed": expected,
                           "expected": expected, "cv2_long_completed": 0})
        command("scripts/extend_logistic_confirmation_cv2.py")
        long_done = sum((FOLDER / f"result_cv2_long_seed{s}.json").exists() and
                        (FOLDER / f"states_cv2_long_seed{s}.npz").exists()
                        for s in seeds)
        if long_done != len(seeds):
            raise ValueError("incomplete CV2 extension")
        save_json(STATUS, {"phase": "final_report", "completed": expected,
                           "expected": expected, "cv2_long_completed": long_done})
        command("scripts/report_logistic_confirmation.py")
        save_json(STATUS, {"phase": "tests", "completed": expected,
                           "expected": expected, "cv2_long_completed": long_done})
        subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True)
        save_json(STATUS, {"phase": "complete", "completed": expected,
                           "expected": expected, "cv2_long_completed": long_done,
                           "report": str(FOLDER / "CONFIRMATION.md")})
        print("fresh-seed confirmation complete", flush=True)
    except Exception as error:
        save_json(STATUS, {"phase": "failed", "error": str(error),
                           "completed": previous, "expected": expected})
        raise


if __name__ == "__main__":
    main()
