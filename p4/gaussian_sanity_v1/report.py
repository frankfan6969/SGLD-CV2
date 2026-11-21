"""Regenerate the P4 report and figures from frozen saved outputs only."""
from pathlib import Path
import json, hashlib, zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/p4_gaussian_sanity_v1'
EXP=ROOT/'exports/p4_gaussian_sanity_v1'
D=json.loads((EXP/'analysis.json').read_text())
C=json.loads((ROOT/'p4/gaussian_sanity_v1/PROTOCOL.json').read_text())

def selected(rows,**kw):
    return [r for r in rows if all(r[k]==v for k,v in kw.items())]

def mean(rows,key):return float(np.mean([r[key] for r in rows]))

maxgrad=max(r['max_scaled_relative_error'] for r in D['gradient_mse'] if r['method']=='cv2')
maxpath=max(r['paired_max_whitened_path_difference'] for r in D['runs'])
maxlaw=max(r['covariance_error_to_finite_law'] for r in D['scores'] if r['method']=='ula' and r['K'])
lines=['# P4 Gaussian exactness and preconditioning sanity check', '',
'**Complete: 32/32 fixed-budget method runs, zero failures; all declared numerical gates pass.**', '',
'This is verification, not another speedup search. JAX CPU float64; N=256,d=10; target κ=10 and10,000; identity/full-H geometry; CV2 and ULA; four paired seeds;512 independent mode-start replicas; checkpoints0/16/64/256/1024; minibatch16 iid with replacement. The common local stability coefficient is h L_P=.1. No performance-based tuning or budget extension occurred.', '',
'## 1. Exactness', '',
f'- Maximum scaled CV2 gradient discrepancy versus independently assembled full gradient: **{maxgrad:.3e}** (gate1e-10).',
f'- Maximum whitened CV2–ULA path discrepancy over every transition and replica: **{maxpath:.3e}** (gate1e-8).',
f'- Largest whitened empirical covariance discrepancy from the exact finite-step ULA law, Frobenius/sqrt(d): **{maxlaw:.4f}** (gross-check gate.5).',
'- Independent repository NumPy versus JAX CV2 gradients pass. The CV2 kernel explicitly evaluates the minibatch Taylor remainder; it does not call or alias the ULA kernel.', '',
'Gradient MSE below is the arithmetic mean squared error over36 fixed locations (anchor,32 posterior draws,three controlled radial states),128 minibatches per location. It includes floating-point arithmetic error. The CV1/SGLD rows are diagnostic controls only, not additional sampling arms.', '',
'|κ|Method|Raw gradient MSE|Identity preconditioned MSE|Full-H preconditioned MSE|Max scaled gradient error|',
'|---:|---|---:|---:|---:|---:|']
for k in C['kappas']:
    for m in ['sgld','cv1','cv2']:
        a=selected(D['gradient_mse'],kappa=k,method=m,metric='identity')[0]
        b=selected(D['gradient_mse'],kappa=k,method=m,metric='full_hessian')[0]
        lines.append(f"|{k}|{m}|{a['raw_mse']:.3e}|{a['preconditioned_mse']:.3e}|{b['preconditioned_mse']:.3e}|{a['max_scaled_relative_error']:.3e}|")
lines+=['', 'The nonzero CV2 MSE is numerical roundoff, not meaningful residual stochastic variance. Finite nonzero values must not be used to estimate a noise-scaling law.', '',
'## 2. Posterior accuracy and geometry', '',
'The exact finite-K Gaussian law uses A=I−hPQ and noise covariance2hP. Population W₂ is computed analytically against N(Q⁻¹b,Q⁻¹), using commuting precision/metric eigenvalues. Full-H removes anisotropy but leaves constant-step discretization bias; exact-gradient CV2 is not exact posterior sampling at finite h.', '',
'At K=1024 (empirical columns average four paired seed ensembles; CV2 and ULA agree to roundoff):', '',
'|κ|Metric|h|Exact population W₂|Exact whitened W₂|Empirical W₂|Exact-vs-exact empirical floor|Whitened covariance error|',
'|---:|---|---:|---:|---:|---:|---:|---:|']
for k in C['kappas']:
    for metric in C['metrics']:
        a=selected(D['analytic'],kappa=k,metric=metric,K=1024)[0]
        s=selected(D['scores'],kappa=k,metric=metric,method='ula',K=1024)
        lines.append(f"|{k}|{metric}|{a['h']:.6g}|{a['population_w2']:.6f}|{a['population_whitened_w2']:.6f}|{mean(s,'w2_empirical'):.6f}|{mean(s,'iid_floor'):.6f}|{mean(s,'covariance_error_whitened'):.6f}|")
lines+=['',
'Identity and full-H use different h, prescribed by the same hL_P=.1 rule. Thus this is a stable-budget geometry ablation, not a same-absolute-step causal comparison. At high κ, identity makes very slow progress in low-curvature directions. Full-H has the same standardized exact-gradient law across κ by construction.', '',
'At κ=10, identity reaches a smaller final population W₂ than full-H under this rule (.0194 versus .0822): it uses a smaller absolute step and has already relaxed by K1024. Full-H therefore does not uniformly minimize finite-step error. Its mode-start discrepancy can also undershoot its eventual discretization floor before increasing toward stationarity; nonmonotone transient W₂ is expected here.', '',
'Empirical W₂ is exact equal-weight optimal transport (Hungarian assignment) between512 endpoints and512 independent posterior draws. It is not sliced W₂ and not a Gaussian moment fit. The exact-vs-exact finite-cloud baseline is retained, never subtracted; empirical endpoint W₂ need not converge to zero at this sample size. Whitened and original-coordinate W₂ are distinct metrics and both are retained in the data. No empirical confidence interval or population-coverage claim is made from four algorithm seeds.', '',
'![Exact finite-step Gaussian transport and covariance](gaussian_law.png)', '',
'![Empirical transport and independent-reference baseline](empirical_transport.png)', '',
'## 3. Runtime (verification context only)', '',
'Whole1024-step,512-replica ensembles; mean four seeds. Warm includes attributed posterior-oracle setup, metric setup, CV cache if used, shared pre-generated RNG input cost and synchronized kernel execution. Cold adds the actual compilation call/cache attribution. Scoring and data construction are excluded. The pre-generated input transfer to JAX is synchronized before timing but is not separately attributed, so these totals are declared verification timings, not production end-to-end deployment benchmarks.', '',
'|κ|Metric|Method|Kernel s|RNG input s|Warm total s|Compile s|Cold total s|',
'|---:|---|---|---:|---:|---:|---:|---:|']
for k in C['kappas']:
    for metric in C['metrics']:
        for method in C['methods']:
            r=selected(D['runs'],kappa=k,metric=metric,method=method)
            vals=[mean(r,x) for x in ['kernel_s','input_rng_s','warm_total_s','compile_s','cold_total_s']]
            lines.append(f"|{k}|{metric}|{method}|"+'|'.join(f'{v:.6f}' for v in vals)+'|')
lines+=['',
'ULA uses a cached analytic quadratic full-gradient oracle Qθ−b, rather than traversing allN observations each step. CV2 carries its explicit minibatch correction. No CV2 speedup or implementation-efficiency conclusion is drawn from these asymmetric verification workloads. Known Gaussian posterior setup is measured and attributed; there is no hidden optimizer. This runtime cannot be pooled with prior GPU seconds.', '',
'## 4. Provenance, export repair and limits', '',
'The first identical32-run attempt passed numerical checks and saved endpoint ensembles, but final JSON export failed on a NumPy Boolean. That attempt and original runner snapshot remain in `results/p4_gaussian_sanity_v1_export_error_attempt1/`. The serializer was repaired and incremental persistence added, then the unchanged protocol/seeds were rerun. This was an infrastructure repair, not a method/parameter amendment; only the completed second attempt supplies final timings. The report audit compares corresponding endpoint archives across attempts.', '',
'All source hashes, endpoint archives, exact-law rows, runtime components and errors are saved. No failed replicas were dropped. This checks a quadratic finite-sum construction at one dimension and two κ values, from the mode. It does not validate nonquadratic Taylor approximation, worst-case mixing, a broad dimension trend, ESS performance, or optimized computational superiority. P3A/P3B remain the primary computational-efficiency evidence.', '',
'## Files', '',
'- `analysis.json`: all gradient controls, exact laws, seed-level accuracy and runtime records.',
'- `gradient_mse.csv`, `analytic_gaussian.csv`, `endpoint_metrics.csv`, `runtime.csv`: complete numerical tables.',
'- `QUALITY_AUDIT.json`: protocol/source/paired-path/failure/replay checks.',
'- `P4_Gaussian_Sanity_Review_v1.zip`: protocol, runner, report, tables, plots and raw endpoint archives.', '']

# Read-only numerical consistency audit; no re-estimation of sampler paths.
audit={'gates_pass':D['gates_pass'],'run_count':len(D['runs']),'failed_runs':sum(r['failed'] for r in D['runs']),'max_cv2_scaled_gradient_error':maxgrad,'max_whitened_paired_path_error':maxpath,'max_covariance_discrepancy_to_exact_law':maxlaw,'source_hashes_verify':True,'repeated_attempt_checkpoint_comparisons':0,'max_repeat_checkpoint_abs_difference':0.}
manifest=json.loads((OUT/'manifest.json').read_text())
for rel,h in manifest['source_sha256'].items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
for f in OUT.glob('k*_seed*.npz'):
    old=ROOT/'results/p4_gaussian_sanity_v1_export_error_attempt1'/f.name
    if old.exists():
        with np.load(f) as a,np.load(old) as b:
            for method in C['methods']:
                delta=float(np.max(np.abs(a[method]-b[method])))
                audit['max_repeat_checkpoint_abs_difference']=max(audit['max_repeat_checkpoint_abs_difference'],delta)
                audit['repeated_attempt_checkpoint_comparisons']+=1
assert audit['max_repeat_checkpoint_abs_difference']<1e-12
(EXP/'QUALITY_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
(EXP/'REPORT.md').write_text('\n'.join(lines))

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(10,4))
for k in C['kappas']:
    for metric in C['metrics']:
        a=selected(D['analytic'],kappa=k,metric=metric)
        x=[r['K'] for r in a]
        label=f'κ={k:,}, '+('identity' if metric=='identity' else 'full-H')
        axes[0].plot(x,[r['population_whitened_w2'] for r in a],marker='o',label=label)
        axes[1].plot(x,[r['population_whitened_cov_error'] for r in a],marker='o',label=label)
for ax,title,ylabel in zip(axes,['Exact finite-step transport','Exact covariance error'],['Whitened population W₂','Relative covariance Frobenius error']):
    ax.set_xscale('symlog',linthresh=16);ax.set_yscale('log');ax.set_xlabel('Iterations K');ax.set_ylabel(ylabel);ax.set_title(title);ax.grid(alpha=.2)
axes[0].legend(fontsize=8);fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(EXP/f'gaussian_law.{ext}',dpi=180)
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,k in zip(axes,C['kappas']):
    for metric in C['metrics']:
        values=[]
        for K in C['checkpoints']:
            values.append(mean(selected(D['scores'],kappa=k,metric=metric,method='ula',K=K),'w2_empirical_whitened'))
        ax.plot(C['checkpoints'],values,marker='o',label=metric.replace('_hessian','-H'))
    floor=selected(D['scores'],kappa=k,method='ula')[0]['iid_floor_whitened']
    ax.axhline(floor,color='grey',ls='--',label='Exact vs exact (512 vs 512)')
    ax.set_xscale('symlog',linthresh=16);ax.set_xlabel('Iterations K');ax.set_ylabel('Empirical whitened full W₂');ax.set_title(f'κ={k:,}; CV2 overlaps ULA');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(EXP/f'empirical_transport.{ext}',dpi=180)
plt.close(fig)

files=[x for x in EXP.iterdir() if x.is_file() and x.suffix!='.zip' and x.name!='ARTIFACT_INDEX.json']+list(OUT.glob('*.npz'))+[OUT/'manifest.json',ROOT/'p4/gaussian_sanity_v1/PROTOCOL.json',ROOT/'p4/gaussian_sanity_v1/run.py',Path(__file__).resolve()]
index={'files':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(files)]}
(EXP/'ARTIFACT_INDEX.json').write_text(json.dumps(index,indent=2)+'\n')
archive=EXP/'P4_Gaussian_Sanity_Review_v1.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in files+[EXP/'ARTIFACT_INDEX.json']:z.write(p,str(p.relative_to(ROOT)))
print(json.dumps({'report':str(EXP/'REPORT.md'),'audit':audit,'archive_bytes':archive.stat().st_size},indent=2))
