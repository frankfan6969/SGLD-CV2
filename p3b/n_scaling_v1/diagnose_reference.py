from pathlib import Path
import sys,json,numpy as np
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'src'))
from sgld_cv.diagnostics import chain_diagnostics
s=np.load(sys.argv[1])['samples'];Path(sys.argv[2]).write_text(json.dumps(chain_diagnostics(s),indent=2)+'\n')
