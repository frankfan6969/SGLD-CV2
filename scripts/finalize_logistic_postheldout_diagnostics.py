"""Finish CV2 chain sensitivity and report after all replica diagnostics land."""
import json
import subprocess
import sys
import time
from pathlib import Path

from sgld_cv.workflow import save_json


def main():
    folder=Path("results/logistic_postheldout_replica_sensitivity")
    config=json.loads(Path("configs/logistic_postheldout_diagnostics.json").read_text())
    expected=[(label,seed) for label in ("cv2_half","cv1_extended")
              for seed in config["seeds"]]
    status=folder/"finalization_status.json"
    deadline=time.monotonic()+8*3600
    previous=-1
    try:
        while True:
            n=sum((folder/f"result_{label}_seed{seed}.json").exists() and
                  (folder/f"states_{label}_seed{seed}.npz").exists()
                  for label,seed in expected)
            if n!=previous:
                save_json(status,{"phase":"replicas","completed":n,
                                  "expected":len(expected)})
                print(f"replicas {n}/{len(expected)}",flush=True)
                previous=n
            if n==len(expected): break
            if time.monotonic()>=deadline:
                raise TimeoutError("post-held-out replica run did not finish in eight hours")
            time.sleep(30)
        save_json(status,{"phase":"cv2_chain","completed":n,"expected":len(expected)})
        subprocess.run([sys.executable,"scripts/diagnose_logistic_cv2_half_chains.py"],
                       check=True)
        save_json(status,{"phase":"report","completed":n,"expected":len(expected)})
        subprocess.run([sys.executable,"scripts/report_logistic_postheldout_diagnostics.py"],
                       check=True)
        save_json(status,{"phase":"tests","completed":n,"expected":len(expected)})
        subprocess.run([sys.executable,"-m","pytest","-q"],check=True)
        save_json(status,{"phase":"complete","completed":n,
                          "expected":len(expected),"report":str(folder/"RESULTS.md")})
        print("post-held-out diagnostics complete",flush=True)
    except Exception as exc:
        save_json(status,{"phase":"failed","completed":previous,
                          "expected":len(expected),"error":str(exc)})
        raise


if __name__=="__main__":
    main()
