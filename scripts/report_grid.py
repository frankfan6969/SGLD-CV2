"""One figure per kappa and comparison, with dimension panels and six curves."""
import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'cv2_matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sgld_cv.workflow import METHODS


def main(folder):
    folder=Path(folder);groups={}
    for manifest in folder.glob('*/manifest.json'):
        meta=json.loads(manifest.read_text())
        if meta['status']!='complete':continue
        c=meta['config'];problem=c['problem']
        kappa=problem.get('kappa',problem.get('design_kappa'))
        if kappa is None:continue
        groups.setdefault((kappa,c.get('comparison','tuned')),[]).append((problem['d'],manifest.parent))
    if not groups:raise ValueError('no completed Gaussian/synthetic grid results found')
    for (kappa,comparison),entries in sorted(groups.items()):
        entries=sorted(entries)
        for horizontal in ['iteration','total_seconds']:
            fig,axes=plt.subplots(1,len(entries),figsize=(5*len(entries),4.8),squeeze=False)
            for ax,(d,path) in zip(axes[0],entries):
                with (path/'metrics.csv').open() as f:rows=list(csv.DictReader(f))
                metric='accuracy' if 'accuracy' in rows[0] else 'w2'
                with (path/'reference_baselines.csv').open() as f:baseline=[float(r[metric]) for r in csv.DictReader(f)]
                for method in METHODS:
                    sub=[r for r in rows if r['method']==method];points=[]
                    for k in sorted({int(r['iteration']) for r in sub}):
                        group=[r for r in sub if int(r['iteration'])==k]
                        w=[float(r[metric]) for r in group] if all(r['status']=='ok' for r in group) else [np.nan]
                        points.append([np.median([float(r[horizontal]) for r in group]),np.median(w)])
                    points=np.array(points)
                    ax.plot(points[:,0],points[:,1],label=method,linestyle='--' if method.startswith('p') else '-')
                ax.axhline(np.median(baseline),color='gray',linestyle=':',label='reference baseline')
                kind=json.loads((path/'manifest.json').read_text()).get('accuracy_metric',{}).get('kind','w2')
                ax.set(title=f'd = {d}',xlabel='Iteration' if horizontal=='iteration' else 'Total ensemble seconds',
                       ylabel='Empirical W2' if kind=='w2' else 'Posterior-standardized sliced W2')
                ax.grid(alpha=.2)
            fig.suptitle(f'Conditioning parameter = {kappa:g}; {comparison} settings (pilot)',fontsize=12)
            handles,labels=axes[0,0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=8)
            fig.tight_layout(rect=(0,.13,1,.91));fig.savefig(folder/f'k{kappa:g}_{comparison}_{horizontal}.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('results',type=Path)
    main(p.parse_args().results)
