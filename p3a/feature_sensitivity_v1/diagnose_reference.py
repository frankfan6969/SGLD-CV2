"""Reference diagnostics in the existing ArviZ-enabled environment."""
from pathlib import Path
import sys,json,numpy as np
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'src'))
from sgld_cv.diagnostics import chain_diagnostics
from sgld_cv.references import validate_reference_pair
samples=np.load(sys.argv[1])['samples'];diag=chain_diagnostics(samples)
Path(sys.argv[2]).write_text(json.dumps(diag,indent=2)+'\n')
