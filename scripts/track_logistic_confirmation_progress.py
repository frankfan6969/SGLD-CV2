"""Keep the extra-cost CV2 progress count current while finalization runs."""
import json
import time
from pathlib import Path

from sgld_cv.workflow import save_json


folder = Path("results/logistic_postdiagnostic_confirmation")
status_path = folder / "finalization_status.json"
progress_path = folder / "progress.json"
protocol = json.loads((folder / "protocol.json").read_text())["config"]
seeds = protocol["method_seeds"]
while True:
    status = json.loads(status_path.read_text())
    if status["phase"] != "cv2_long":
        break
    count = sum((folder / f"result_cv2_long_seed{s}.json").exists() and
                (folder / f"states_cv2_long_seed{s}.npz").exists() for s in seeds)
    if not progress_path.exists() or json.loads(progress_path.read_text()).get(
            "cv2_long_completed") != count:
        save_json(progress_path, {"phase": "cv2_long", "fixed_k_completed": 48,
                                  "fixed_k_expected": 48,
                                  "cv2_long_completed": count,
                                  "cv2_long_expected": len(seeds)})
        print(f"CV2 long {count}/{len(seeds)}", flush=True)
    time.sleep(15)
