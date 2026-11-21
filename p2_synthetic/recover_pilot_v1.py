"""Complete only the two synthetic pilots that failed before sampler evaluation."""
from pathlib import Path
import sys,json,time,zipfile,traceback,shutil
from time import perf_counter
ROOT=Path('/content/p2_synthetic_v1')
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'gpu_replication'),str(ROOT/'p2_screening')]
import run as pilot
import numpy as np
from scipy.optimize import minimize
from sgld_cv.anchors import Anchor
from sgld_cv.costs import CostLedger
from threadpoolctl import threadpool_limits
OUT=ROOT/'results/p2_synthetic_pilot_recovery_v1'
DRIVE=Path('/content/drive/MyDrive/CV2_Research/P2_Synthetic_Pilot_Recovery_v1_2026-10-15')

def precise_anchor(model):
    ledger=CostLedger()
    def g(x):
        ledger.count('observation_gradient_evaluations',model.N)
        return model.full_gradient(x)
    def H(x):
        ledger.count('observation_hessian_evaluations',model.N)
        return model.full_hessian(x)
    with ledger.measure('anchor_optimization'):
        result=minimize(model.potential,np.zeros(model.d),jac=g,hess=H,method='Newton-CG',options={'xtol':1e-8,'maxiter':1000})
        x=result.x.copy(); before=float(np.linalg.norm(g(x))); steps=0
        for _ in range(12):
            grad=g(x)
            if np.linalg.norm(grad)<=1e-8:break
            matrix=H(x)
            assert np.linalg.eigvalsh(matrix)[0]>0
            delta=np.linalg.solve(matrix,grad)
            proposals=[x-delta*(.5**j) for j in range(12)]
            best=min(proposals,key=lambda q:np.linalg.norm(g(q)))
            if np.linalg.norm(g(best))>=np.linalg.norm(grad):break
            x=best;steps+=1
        norm=float(np.linalg.norm(g(x))); eigenvalues=np.linalg.eigvalsh(H(x))
        assert np.isfinite(x).all() and norm<=1e-8 and eigenvalues[0]>0,(norm,eigenvalues)
    metadata=dict(source='original Newton-CG plus bounded analytic Newton gradient-norm refinement',optimized=True,
        original_success=bool(result.success),original_message=str(result.message),before_gradient_norm=before,
        gradient_norm=norm,refinement_steps=steps,gradient_gate=1e-8,min_hessian_eigenvalue=float(eigenvalues[0]))
    pilot.save(OUT/f'anchor_audit_{precise_anchor.calls}.json',metadata);precise_anchor.calls+=1
    x.flags.writeable=False
    return Anchor(x,ledger,metadata)
precise_anchor.calls=0

def main():
    c=json.loads((ROOT/'p2_screening/PROTOCOL.json').read_text())
    lock=json.loads((ROOT/'p2_screening/lock.json').read_text())
    assert all(pilot.digest(ROOT/p)==h for p,h in lock['files'].items())
    assert Path('/content/drive/MyDrive').is_dir()
    device=pilot.jax.devices('gpu')[0];assert 'A100' in device.device_kind
    OUT.mkdir(parents=True,exist_ok=False);DRIVE.mkdir(parents=True,exist_ok=True)
    pilot.OUT,pilot.DRIVE,pilot.find_anchor=OUT,DRIVE,precise_anchor
    amendment=json.loads(Path('/content/SYNTHETIC_RECOVERY_PROTOCOL_v1.json').read_text())
    pilot.save(OUT/'PROTOCOL_AMENDMENT.json',amendment)
    shutil.copyfile(__file__,OUT/'recover_pilot_v1.py');pilot.save(OUT/'original_source_lock.json',lock)
    pilot.save(OUT/'manifest.json',dict(status='validating',device=device.device_kind,heldout_consumed=False));pilot.sync()
    start=perf_counter()
    try:
        with threadpool_limits(limits=1):
            pilot.save(OUT/'validation.json',pilot.validate(device));pilot.sync()
            for spec in c['targets']:
                if spec['id'] not in amendment['targets']:continue
                pilot.save(OUT/'manifest.json',dict(status='running',target=spec['id'],heldout_consumed=False));pilot.sync()
                pilot.target(spec,c,min(start+3600,perf_counter()+1200))
            pilot.save(OUT/'manifest.json',dict(status='complete',elapsed_seconds=perf_counter()-start,heldout_consumed=False,
                scope='Recovered pilot only; selection/heldout not run'))
    except Exception:
        pilot.save(OUT/'manifest.json',dict(status='failed',error=traceback.format_exc(),heldout_consumed=False));raise
    finally:
        archive=OUT/'P2_Synthetic_Pilot_Recovery_Results.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in OUT.rglob('*'):
                if p.is_file() and p!=archive:z.write(p,p.relative_to(OUT))
        pilot.sync();shutil.copyfile(archive,'/content/'+archive.name)
    print('SYNTHETIC_RECOVERY_COMPLETE_AND_BACKED_UP',DRIVE,flush=True)
if __name__=='__main__':main()
