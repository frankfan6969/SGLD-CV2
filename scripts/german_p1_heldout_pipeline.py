"""Run frozen heldout sampling then evaluation; persist process ID for power guard."""
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


if __name__=="__main__":
    Path("/tmp/p1_german_heldout_pipeline.pid").write_text(str(os.getpid()))
    subprocess.run([sys.executable,str(ROOT/"scripts/run_german_p1_heldout.py"),"--config",str(ROOT/"configs/p1_german_heldout_v1.json")],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/"scripts/evaluate_german_p1_heldout.py")],cwd=ROOT,check=True)
