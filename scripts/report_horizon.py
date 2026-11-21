"""Plot the saved horizon probe without rerunning samplers."""
import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"cv2_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sgld_cv.workflow import METHODS


def report(folder):
    folder=Path(folder)
    with (folder/"horizon_curves.csv").open() as file: rows=list(csv.DictReader(file))
    summary=json.loads((folder/"horizon_summary.json").read_text())
    calibration=summary.get("accuracy_metric",{})
    (folder/"HORIZON_RESULTS.md").write_text(
        f"# Horizon probe\n\nPhase: {summary['phase']}. Metric: {summary['metric']}.\n\n"
        f"Scoring calibration: {calibration.get('calibration','not recorded')}. "
        f"Calibration SHA-256: {calibration.get('calibration_sha256') or 'not applicable'}.\n\n"
        f"Provisional common K: {summary['provisional_common_k'] if summary['provisional_common_k'] is not None else 'undetermined'}.\n\n"
        f"{summary['interpretation']}\n")
    for column,label in (("accuracy",summary["metric"]),
                         ("mean_error_std","Standardized mean error"),
                         ("covariance_error_std","Standardized covariance error")):
        fig,ax=plt.subplots(figsize=(7.5,4.8))
        for method in METHODS:
            sub=[r for r in rows if r["method"]==method]
            points=[]
            for k in sorted({int(r["iteration"]) for r in sub}):
                group=[r for r in sub if int(r["iteration"])==k]
                values=[float(r[column]) for r in group if r["status"]=="ok"]
                points.append((k,np.median(values) if len(values)==len(group) else np.nan))
            x,y=np.array(points).T
            ax.plot(x,y,marker="o",markersize=3,label=method,
                    linestyle="--" if method.startswith("p") else "-")
        if column=="accuracy":
            ax.axhline(summary["reference_baseline_q75"],color="gray",linestyle=":",
                       label="reference baseline q75")
        ax.set(xlabel="Iteration",ylabel=label,title=f"{summary['phase']} horizon probe; provisional only")
        ax.grid(alpha=.2);ax.legend(ncol=2,fontsize=8);fig.tight_layout()
        fig.savefig(folder/f"horizon_{column}.png",dpi=180);plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results",type=Path)
    report(parser.parse_args().results)
