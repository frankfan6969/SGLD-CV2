"""Run the independently frozen GPU replication, inference, and export."""
from pathlib import Path
import json,os,subprocess,sys,traceback,zipfile
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/p1_german_gpu_replication_v1'
env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
def export():
    destination=ROOT.parent/'P1_GPU_Full_Results.zip'
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file():z.write(p,str(p.relative_to(ROOT)))
        for p in (ROOT/'gpu_replication').glob('*.json'):z.write(p,str(p.relative_to(ROOT)))
    print('EXPORT',destination,flush=True)
try:
    for script in ['run_full_replication.py','report_full_replication.py']:
        print('START',script,flush=True)
        subprocess.run([sys.executable,str(ROOT/'gpu_replication'/script)],cwd=ROOT,env=env,check=True)
    (OUT/'pipeline_status.json').write_text(json.dumps(dict(status='complete')))
except Exception:
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'pipeline_status.json').write_text(json.dumps(dict(status='failed',error=traceback.format_exc())))
    raise
finally:
    export()
