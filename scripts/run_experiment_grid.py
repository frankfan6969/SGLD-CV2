"""Sequential resumable configuration suite; completed outputs are preserved."""
import argparse
import hashlib
import json
from pathlib import Path
from threadpoolctl import threadpool_limits
from sgld_cv.workflow import run_study
import sgld_cv.workflow

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('suite',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();suite=json.loads(a.suite.read_text());a.output.mkdir(parents=True,exist_ok=True)
    for name in suite['configs']:
        config=json.loads((a.suite.parent/name).read_text());out=a.output/Path(name).stem
        if (out/'manifest.json').exists():
            old=json.loads((out/'manifest.json').read_text())
            source_root=Path(sgld_cv.workflow.__file__).parent
            same_source=all((source_root/k).exists() and hashlib.sha256((source_root/k).read_bytes()).hexdigest()==v
                            for k,v in old.get('source_sha256',{}).items())
            if old['status']=='complete' and old['config']==config and same_source:
                print(f'Skipping completed {out}',flush=True);continue
            raise RuntimeError(f'{out} is incomplete or uses a different configuration; choose a new output path')
        with threadpool_limits(limits=1):run_study(config,out)
