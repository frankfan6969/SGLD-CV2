"""Write explicit d-by-conditioning configs; no expensive experiments run here."""
import argparse
import json
from pathlib import Path

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dimensions',nargs='+',type=int,default=[2,20,100])
    p.add_argument('--kappas',nargs='+',type=float,default=[10,100,1000])
    a=p.parse_args();base=json.loads(a.base.read_text())
    a.output.mkdir(parents=True,exist_ok=False)
    items=[]
    for d in a.dimensions:
        for k in a.kappas:
            for comparison in ['common','tuned']:
                c=json.loads(json.dumps(base));c['comparison']=comparison;c['problem']['d']=d
                c['problem']['N']=max(c['problem']['N'],10*d)
                key='kappa' if c['problem']['kind']=='gaussian' else 'design_kappa'
                c['problem'][key]=k
                name=f'd{d}_k{k:g}_{comparison}.json'
                (a.output/name).write_text(json.dumps(c,indent=2)+'\n');items.append(name)
    (a.output/'suite.json').write_text(json.dumps({'configs':items},indent=2)+'\n')
