"""Generate figures, separate scientific tables, and paired comparisons from saved runs."""
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
from sgld_cv.workflow import save_csv, METHODS


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def report(folder):
    folder=Path(folder)
    manifest=json.loads((folder/'manifest.json').read_text())
    metrics=read(folder/'metrics.csv');ess=read(folder/'ess.csv')
    baselines=read(folder/'reference_baselines.csv')
    metric_name=manifest.get('accuracy_metric',{}).get('kind','w2')
    metric_column='accuracy' if 'accuracy' in metrics[0] else 'w2'
    ylabel='Empirical W2' if metric_name=='w2' else 'Posterior-standardized sliced W2'
    colors=dict(zip(METHODS,['#d55e00','#0072b2','#009e73','#cc79a7','#56b4e9','#8073ac']))
    label={'sgld':'SGLD','cv1':'CV1','cv2':'CV2','psgld':'pSGLD','pcv1':'pCV1','pcv2':'pCV2'}
    for axis in ['iteration','total_seconds','observation_gradients']:
        fig,ax=plt.subplots(figsize=(7.5,4.8))
        for method in METHODS:
            rows=[r for r in metrics if r['method']==method]
            points=[]
            for k in sorted({int(r['iteration']) for r in rows}):
                group=[r for r in rows if int(r['iteration'])==k]
                w=[float(r[metric_column]) for r in group] if all(r['status']=='ok' and r[metric_column] for r in group) else [np.nan]
                points.append([np.median([float(r[axis]) for r in group]),np.median(w),np.quantile(w,.25),np.quantile(w,.75)])
            x,y,lo,hi=np.array(points).T
            ax.plot(x,y,label=label[method],color=colors[method],linestyle='--' if method.startswith('p') else '-')
            ax.fill_between(x,lo,hi,color=colors[method],alpha=.12)
        base=[float(r[metric_column]) for r in baselines]
        ax.axhline(np.median(base),color='gray',linestyle=':',label='Reference vs reference')
        ax.axhspan(np.quantile(base,.25),np.quantile(base,.75),color='gray',alpha=.1)
        ax.set(xlabel={'iteration':'Iteration','total_seconds':'Total ensemble seconds, including setup',
                       'observation_gradients':'Observation-gradient evaluations, including setup'}[axis],ylabel=ylabel)
        validity='reference diagnostics passed' if manifest['reference_reliable'] else 'REFERENCE NOT VALIDATED'
        ax.set_title(f"{manifest['config']['problem']['kind']} ({manifest['config']['purpose']})\n{validity}; median / IQR across seeds",fontsize=10)
        ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2);fig.tight_layout()
        fig.savefig(folder/f'accuracy_{axis}.png',dpi=180)
        if metric_name=='w2':
            fig.savefig(folder/f'w2_{axis}.png',dpi=180)
        plt.close(fig)
    radius=read(folder/'gradient_radius.csv')
    fig,ax=plt.subplots(figsize=(6.5,4.5))
    for method in ['sgld','cv1','cv2']:
        rows=[r for r in radius if r['method']==method]
        ax.loglog([float(r['radius']) for r in rows],[max(float(r['mean_mse']),1e-30) for r in rows],
                  marker='o',label=label[method],color=colors[method])
    ax.set(xlabel='Distance from anchor',ylabel='Mean squared gradient error',title='Shared states and batches; raw gradient MSE')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(folder/'gradient_radius.png',dpi=180);plt.close(fig)
    summary=[];costs=[]
    for method in sorted({r['method'] for r in ess}):
        group=[r for r in ess if r['method']==method]
        values=lambda key:[float(r[key]) for r in group if r[key] not in ('','None')]
        item={'method':method,'all_chains_reliable':all(r['reliable']=='True' for r in group),
              'median_bulk_ess':np.median(values('ess_bulk')) if values('ess_bulk') else None,
              'median_ess_per_total_second':np.median(values('ess_per_total_second')) if values('ess_per_total_second') else None,
              'minimum_tail_ess':min(values('ess_tail')) if values('ess_tail') else None,
              'max_rhat':max(values('rhat')) if values('rhat') else None,
              'interpretation':'exploratory adaptive ESS' if method.startswith('p') and not method.endswith('_frozen') else 'check mixing flags and W2 before comparing throughput'}
        summary.append(item)
    for file in folder.glob('diagnostics_*_seed*.json'):
        data=json.loads(file.read_text());stem=file.stem.removeprefix('diagnostics_')
        method,seed=stem.rsplit('_seed',1)
        costs.append({'method':method,'seed':int(seed),'total_seconds':data['cost']['total_seconds'],
                      **data['cost']['seconds'],**data['cost']['operations']})
    save_csv(folder/'ess_summary.csv',summary);save_csv(folder/'cost_breakdown.csv',costs)
    paired=[]
    lookup={(r['method'],r['seed'],r['iteration']):r for r in metrics}
    for r in metrics:
        if r['method'] not in ('cv2','pcv2') or r['status']!='ok':continue
        other=lookup[(r['method'].replace('2','1'),r['seed'],r['iteration'])]
        if other['status']=='ok':
            paired.append({'comparison':r['method']+' minus '+other['method'],'seed':r['seed'],
                'iteration':r['iteration'],'accuracy_difference':float(r[metric_column])-float(other[metric_column]),
                'metric':metric_name,
                'note':'matched iteration; separately tuned settings, not matched total cost'})
    save_csv(folder/'paired_accuracy.csv',paired)
    # A conservative resolution gate, not subtraction or a bias correction.
    accuracy=[]
    for epsilon in manifest['config'].get('accuracy_thresholds',[]):
        for method in METHODS:
            for seed in sorted({r['seed'] for r in metrics}):
                rows=sorted([r for r in metrics if r['method']==method and r['seed']==seed],key=lambda r:int(r['iteration']))
                floor=float(np.quantile([float(r[metric_column]) for r in baselines if r['seed']==seed],.75))
                status='unresolved at current reference sample size' if epsilon<=floor else 'not reached within budget'
                reached=None
                if epsilon>floor:
                    for i in range(len(rows)-2):
                        if all(r['status']=='ok' and float(r[metric_column])<=epsilon for r in rows[i:i+3]):
                            reached=rows[i];status='reached for three recorded checkpoints';break
                accuracy.append({'method':method,'seed':seed,'epsilon':epsilon,'reference_baseline_q75':floor,
                    'status':status,'metric':metric_name,'iteration':reached['iteration'] if reached else None,
                    'total_seconds':reached['total_seconds'] if reached else None})
    if accuracy:save_csv(folder/'time_to_accuracy.csv',accuracy)
    lines=['# Experiment results','',f"Purpose: {manifest['config']['purpose']}. Reference validation: {manifest['reference_reliable']}.",'',
           f'The six main methods are SGLD, CV1, CV2 and their adaptive Li-style preconditioned variants. Frozen variants are additional efficiency controls. Adaptive Gamma is omitted. Posterior accuracy metric: {metric_name}.','',
           f"Scoring calibration: {manifest.get('accuracy_metric',{}).get('calibration','not recorded')}. "
           f"Calibration SHA-256: {manifest.get('accuracy_metric',{}).get('calibration_sha256') or 'not applicable'}. "
           "Compare frozen runs only when their scoring calibration fingerprints match.",'',
           '## Sampling efficiency','',
           '| Method | Median bulk ESS | Median ESS per total second | Diagnostics passed |',
           '| --- | ---: | ---: | --- |']
    for row in summary:
        f=lambda x:f'{x:.3g}' if x is not None else 'NA'
        lines.append(f"| {row['method']} | {f(row['median_bulk_ess'])} | {f(row['median_ess_per_total_second'])} | {row['all_chains_reliable']} |")
    lines+=['','ESS uses only post-burn-in samples; total seconds include anchor optimization, CV preparation, metric pilot, burn-in and retained sampling. False flags mean values must not be ranked as validated posterior throughput. A passing flag does not establish posterior accuracy.','',
            '## Gradient estimation','',
            'See gradient_mse.csv for arithmetic mean MSE and its state-level standard error, median and upper-tail squared errors. This is separate from the posterior-sampling and ESS tables.','',
            '## Accuracy and cost','',
            'See accuracy_iteration.png and accuracy_total_seconds.png. The reference baseline is not subtracted and is not a lower bound: a collapsed distribution can score below it. Mean and covariance errors are also saved. At fixed iteration the methods can have different tuned batches and steps; paired_accuracy.csv therefore does not establish a fixed-cost advantage. cost_breakdown.csv preserves each setup component and curvature operation counts.','',
            'These pilot outputs are not publication conclusions. Longer runs, adequate reference resolution and held-out evaluation remain necessary.']
    (folder/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    # Standalone LaTeX table for later manuscript inclusion; avoid interpreting failures as wins.
    latex=['\\begin{tabular}{lrrl}','\\hline','Method & Bulk ESS & ESS/total second & Diagnostics passed \\\\','\\hline']
    for r in summary:
        f=lambda x:f'{x:.3g}' if x is not None else '--'
        method_tex=r['method'].replace('_',r'\_')
        latex.append(f"{method_tex} & {f(r['median_bulk_ess'])} & {f(r['median_ess_per_total_second'])} & {r['all_chains_reliable']} \\\\")
    latex+=['\\hline','\\end{tabular}']
    (folder/'ess_table.tex').write_text('\n'.join(latex)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('results',type=Path)
    report(p.parse_args().results)
