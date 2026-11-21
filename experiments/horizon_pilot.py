"""Estimate a provisional common horizon on a representative logistic target."""
import argparse
import json
import os
import tempfile
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"cv2_matplotlib"))
from threadpoolctl import threadpool_limits
from sgld_cv.horizon import run_horizon_probe


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--frozen-selection",type=Path)
    a=p.parse_args()
    with threadpool_limits(limits=1):
        result=run_horizon_probe(json.loads(a.config.read_text()),a.output,
                                 frozen_selection=a.frozen_selection)
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
