"""Wait for all frozen replica runs, then validate chains and publish the report."""
import json
import subprocess
import sys
import time
from pathlib import Path

from sgld_cv.workflow import save_json


def main():
    folder = Path("results/logistic_fixed8192_heldout")
    status_path = folder / "finalization_status.json"
    config = json.loads((folder / "protocol.json").read_text())["config"]
    expected = [(m, s) for s in config["method_seeds"] for m in config["methods"]]
    deadline = time.monotonic() + 8 * 3600
    previous = -1
    try:
        while True:
            completed = sum((folder / f"result_{m}_seed{s}.json").exists() and
                            (folder / f"states_{m}_seed{s}.npz").exists()
                            for m, s in expected)
            if completed != previous:
                save_json(status_path, {"phase": "replicas", "completed": completed,
                                        "expected": len(expected)})
                print(f"replicas {completed}/{len(expected)}", flush=True)
                previous = completed
            if completed == len(expected):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("replica experiment did not complete within eight hours")
            time.sleep(30)
        save_json(status_path, {"phase": "chain_validation", "completed": completed,
                                "expected": len(expected)})
        subprocess.run([sys.executable, "scripts/validate_logistic_fixed8192_chains.py"],
                       check=True)
        save_json(status_path, {"phase": "report", "completed": completed,
                                "expected": len(expected)})
        subprocess.run([sys.executable, "scripts/report_logistic_fixed8192_heldout.py"],
                       check=True)
        save_json(status_path, {"phase": "tests", "completed": completed,
                                "expected": len(expected)})
        subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True)
        save_json(status_path, {"phase": "complete", "completed": completed,
                                "expected": len(expected),
                                "report": str(folder / "RESULTS.md")})
        print("held-out experiment complete", flush=True)
    except Exception as error:
        save_json(status_path, {"phase": "failed", "error": str(error),
                                "completed": previous, "expected": len(expected)})
        raise


if __name__ == "__main__":
    main()
