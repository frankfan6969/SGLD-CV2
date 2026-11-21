"""One-shot local finalization for the detached experiment, no further experiments."""
from pathlib import Path
import sys,json,time,subprocess,hashlib,zipfile,os
R=Path(__file__).resolve().parents[2];H=Path(__file__).parent;O=R/'exports/p3a_feature_sensitivity_v1'
if '--wait' in sys.argv:
 start=time.monotonic()
 while time.monotonic()-start<7500:
  p=O/'stage_b_manifest.json';state=json.loads(p.read_text()) if p.exists() else {}
  if state.get('status') in ['complete','failed_or_budget_censored']:break
  time.sleep(10)
subprocess.run([str(R/'.venv-gpu/bin/python'),str(H/'report.py')],check=True)
subprocess.run(['/Library/Frameworks/Python.framework/Versions/3.10/bin/python3',str(H/'plot.py')],env=dict(os.environ,MPLCONFIGDIR='/private/tmp/cv2_mpl'),check=True)
for p in H.iterdir():
 if p.is_file() and p.suffix in ['.py','.json','.log']:(O/p.name).write_bytes(p.read_bytes())
files=sorted(p for p in O.rglob('*') if p.is_file() and p.suffix not in ['.zip','.tmp'] and not p.name.startswith('.') and p.name!='checksums.json');checks={str(p.relative_to(O)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};(O/'checksums.json').write_text(json.dumps(checks,indent=2)+'\n');archive=O/'P3A_Feature_Sensitivity_v1.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in files+[O/'checksums.json']:z.write(p,str(p.relative_to(O)))
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
s=json.loads((R/'p3a/STATUS.json').read_text());m=json.loads((O/'stage_b_manifest.json').read_text());s.update(stage='Feature sensitivity stageB '+m['status'],feature_sensitivity_report=str((O/'REPORT.md').relative_to(R)),feature_sensitivity_archive=str(archive.relative_to(R)),feature_sensitivity_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),feature_sensitivity_archive_verified=True,feature_sensitivity_persistence='Local only; no active Drive sync',next_step='Review feature sensitivity; P3B not launched.');(R/'p3a/STATUS.json').write_text(json.dumps(s,indent=2)+'\n');print('FEATURE_SENSITIVITY_FINALIZED',m['status'],flush=True)
