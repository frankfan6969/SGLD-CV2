"""Presentation-only one-stop report; never modifies frozen experimental metrics."""
from pathlib import Path
import os,json,csv,hashlib
import numpy as np
R=Path(__file__).resolve().parents[2];O=Path(os.environ.get('P3B_REPORT_INPUT',str(R/'exports/p3b_n_scaling_v1')));D=Path(os.environ.get('P3B_REPORT_RAW',str(R/'results/p3b_n_scaling_v1')));a=json.loads((O/'analysis.json').read_text());c=a['protocol'];Ns=c['data']['N'];byN={q['N']:q for q in a['statuses']};quality=[];index=[];evidence=[];threshold_rows=[]
fmt=lambda v,precision=3:'—' if v is None else f'{v:.{precision}f}'
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(p.read_text()) if p.exists() else None
def sweep(N,threshold,cost):return next((q for q in a['optimized_thresholds'] if q['N']==N and q['variance_threshold']==threshold and q['cost']==cost),None)
def get_slope(method,metric,group='reference',scale=None):return next((q for q in a['descriptive_scaling_slopes'] if q['method']==method and q['metric']==metric and q['group']==group and q['scale']==scale),None)
for N in Ns:
 ND=D/f'N{N}';review=O/f'N{N}';va=load(review/'analysis.json');raw=ND/'optimized';ref=load(ND/'target/reference_pair.json');diags=[load(ND/'target'/f'reference_{j}.json') for j in [0,1]];seed=load(raw/'seed_audit.json');preflight=load(raw/'PREFLIGHT.json');manifest=load(raw/'manifest.json');archive=Path(manifest['archive']) if manifest and 'archive' in manifest else None;archive_match=bool(archive and archive.exists() and digest(archive)==manifest['archive_sha256']);individual=bool(all(q and q.get('reliable') for q in diags));pair=bool(ref and ref.get('reliable'));passed=byN.get(N,{}).get('status')=='complete' and va is not None and individual and pair and archive_match and bool(seed and seed.get('passed')) and bool(preflight and preflight.get('status')=='passed');v=va['validation'] if va else {};quality.append(dict(N=N,status=byN.get(N,{}).get('status','not_completed'),report_ready=passed,reference_individual_passed=individual,reference_pair_passed=pair,max_rhat=max((q['max_rhat'] for q in diags if q),default=None),min_bulk_ess=min((q['min_bulk_ess'] for q in diags if q),default=None),min_tail_ess=min((q['min_tail_ess'] for q in diags if q),default=None),max_mean_mcse_z=ref.get('max_mean_mcse_z') if ref else None,covariance_relative_difference=ref.get('covariance_relative_difference') if ref else None,seed_audit_passed=bool(seed and seed.get('passed')),preflight_passed=bool(preflight and preflight.get('status')=='passed'),archive_sha256_verified=archive_match,**{k:v.get(k) for k in ['runs','pilot_runs','candidate_count','eligible_configs','failed_runs','failed_replicas','budget_censored_runs','nrmses_reconstructed','simultaneous_maxstat_q95']}));index.append(dict(N=N,report=str(review/'REPORT.md'),analysis=str(review/'analysis.json'),frontier=str(review/'accuracy_cost_frontiers.png'),thresholds=str(review/'optimized_thresholds.png'),raw_directory=str(ND),archive=str(archive) if archive else None,archive_sha256=manifest.get('archive_sha256') if manifest else None))
 for cost in ['warm','cold']:
  for threshold in c['optimized']['threshold_sweep']['variance']:
   q=sweep(N,threshold,cost)
   if q is None:continue
   row=dict(N=N,cost=cost,variance_threshold=float(threshold),point_speedup=q['point_speedup_cv1_over_cv2'],point_speedup_ci95=q['conditional_speedup_ci95'],supported_speedup=q['supported_point_speedup'],bootstrap_both_reached_fraction=q['bootstrap_both_reached_fraction'])
   for method in ['cv1','cv2']:
    v=q['methods'][method]
    for typ,prefix in [('point_choice','point'),('simultaneous_supported_choice','supported')]:
     choice=v[typ]
     for key in ['batch','h','K',cost+'_seconds']:row[f'{method}_{prefix}_{key}']=choice.get(key) if choice else None
    row[method+'_bootstrap_not_reached_fraction']=v['bootstrap_not_reached_fraction']
   threshold_rows.append(row)
mechanism_complete=all(any(q['N']==N and q['method']==m and q['group']=='reference' for q in a['mechanism']) for N in Ns for m in ['sgld','cv1','cv2']);threshold_sweep_complete=all(sum(q['N']==N for q in a['optimized_thresholds'])==82 for N in Ns);ready=all(q['report_ready'] for q in quality) and len(quality)==4 and mechanism_complete and threshold_sweep_complete;summary=['# P3B 一页结论摘要','', '**状态：'+('四个N的独立验证、reference质量和各N归档校验均已完成。' if ready else '结果不完整或存在未通过的质量检查；不作完整四点一般化结论。')+'**','']
summary+=['## 1. 梯度噪声随N怎样变化？','', '|Method|Raw MSE log-log slope|Preconditioned slope|Reference N points|','|---|---:|---:|---:|']
for m in ['sgld','cv1','cv2']:
 p=get_slope(m,'raw_mse_mean');w=get_slope(m,'preconditioned_mse_mean');summary.append(f"|{m.upper()}|{fmt(p['descriptive_loglog_slope']) if p else '—'}|{fmt(w['descriptive_loglog_slope']) if w else '—'}|{p['N_count'] if p else 0}|");evidence.append(dict(question='Gradient variance scaling',method=m,metric='raw/preconditioned descriptive slope',finding=dict(raw=p,preconditioned=w),scope='Fixedn16,one nested family,fourN; raw theoretical upper orders are benchmarks, not exact equalities',source='analysis.json:descriptive_scaling_slopes'))
summary+=['', '这是固定n=16、各N自己的32个reference位置的描述性趋势。不是渐近证明；raw与preconditioned幂率不能混用。','', '## 2. 优化后的效率优势是否随N持续？','', '固定白化SW2≤.3；下表warm speedup=CV1/CV2 time，>1表示CV2更快，—表示无法计算有限比值。','', '|NRMSE target|N=2000|N=8000|N=32000|N=128000|','|---:|---:|---:|---:|---:|']
for threshold in [.5,.4,.3,.2]:
 row=[];records=[]
 for N in Ns:
  q=sweep(N,threshold,'warm');row.append(fmt(q['point_speedup_cv1_over_cv2']) if q else '—');records.append(q)
 summary.append('|'+f'{threshold:.2f}'+'|'+'|'.join(row)+'|');valid=[q for q in records if q and q['point_speedup_cv1_over_cv2'] is not None];directions='四个N点估计方向均有利于CV2' if len(valid)==4 and all(q['point_speedup_cv1_over_cv2']>1 for q in valid) else '四个N点估计方向均有利于CV1' if len(valid)==4 and all(q['point_speedup_cv1_over_cv2']<1 for q in valid) else '方向混合或存在未达标N，不能概括为全程优势';evidence.append(dict(question='Time-to-accuracy persistence',variance_target=threshold,finding=directions,scope='Frozen validated candidate subgrid,whitenedSW2≤.3,CPUwarm time; not continuous globaloptimum',source='analysis.json:optimized_thresholds'))
summary+=['', '完整41档warm/cold结果与同时accuracy上界支持的比较见主报告。不要只挑选有利阈值；Not Reached不是无限speedup。','', '## 3. 极端observations是否抵消concentration？','']
ref=[q for q in a['mechanism'] if q['method']=='cv2' and q['group']=='reference'];shell=[q for q in a['mechanism'] if q['method']=='cv2' and q['group']=='controlled_shell' and q['scale']==1.];md=sorted(a['target_metadata'],key=lambda q:q['N'])
if len(ref)>=2:
 ref=sorted(ref,key=lambda q:q['N']);ratio=ref[-1]['max_raw_remainder_across_locations']/ref[0]['max_raw_remainder_across_locations'];summary.append(f"从N={ref[0]['N']}到{ref[-1]['N']}，reference位置最大raw CV2 residual变为原来的{ratio:.4f}倍。")
if len(shell)>=2:
 shell=sorted(shell,key=lambda q:q['N']);ratio=shell[-1]['max_raw_remainder_across_locations']/shell[0]['max_raw_remainder_across_locations'];summary.append(f"共同白化半径sqrt(d)的最大raw CV2 residual变为{ratio:.4f}倍；这一对照不依赖不同N的reference位置抽样。")
if len(md)>=2:summary.append(f"同时，max||x_i||从{md[0]['max_x_norm']:.4f}变为{md[-1]['max_x_norm']:.4f}。应结合两条曲线和失败记录解释，不能只看典型MSE。")
summary+=['', '## 4. 稳定性与证据边界','', '|N|Validation failed runs|Failed replicas|Censored runs|Quality ready|','|---:|---:|---:|---:|---|']
for q in quality:summary.append(f"|{q['N']}|{q['failed_runs']}|{q['failed_replicas']}|{q['budget_censored_runs']}|{q['report_ready']}|")
summary+=['', '只有一个AR(1)rho=.5、一个beta和一个nested数据实现。Nested N、41thresholds和配对seeds不能被当成独立数据生成重复。Bootstrap条件于固定reference，且每N只有8seed clusters；同时带是近似校准，不是跨四N的联合95%覆盖保证。','', 'P3A已关闭，不重跑P2 Bank追求有利结果；本报告不自动授权继续扩展实验。']
(O/'ONE_PAGE_SUMMARY.md').write_text('\n'.join(summary)+'\n');body=(O/'REPORT.md').read_text();body=(O/'METHODS_AND_RESULTS.md').read_text() if body.startswith('# P3B 最终一站式报告') and (O/'METHODS_AND_RESULTS.md').exists() else body;(O/'METHODS_AND_RESULTS.md').write_text(body);text=['# P3B 最终一站式报告','', '\n'.join(summary[2:]),'', '## 阅读导航','',f"- [一页摘要]({O/'ONE_PAGE_SUMMARY.md'})",f"- [方差scaling图]({O/'gradient_noise_scaling.png'})",f"- [最大remainder与极端设计图]({O/'remainder_scaling.png'})",f"- [N–time-to-accuracy图]({O/'time_to_accuracy_scaling.png'})",f"- [全部41档threshold热图]({O/'N_threshold_heatmap.png'})",f"- [完整阈值及最优配置CSV]({O/'optimized_thresholds_all.csv'})",f"- [质量审计JSON]({O/'QUALITY_AUDIT.json'})",'', '## Reference与实现质量审计','', '|N|max Rhat|min bulk ESS|min tail ESS|Pair passed|Seed audit|Archive hash|','|---:|---:|---:|---:|---|---|---|']
for q in quality:text.append(f"|{q['N']}|{fmt(q['max_rhat'],5)}|{fmt(q['min_bulk_ess'],0)}|{fmt(q['min_tail_ess'],0)}|{q['reference_pair_passed']}|{q['seed_audit_passed']}|{q['archive_sha256_verified']}|")
text+=['', '## 同时accuracy上界支持的warm比较','', '这张表的达标条件比点估计更保守，最优配置可能不同；并不保证连续参数空间成本最优。每N同时带分别校准，未作跨四N联合校准。','', '|NRMSE target|N=2000|N=8000|N=32000|N=128000|','|---:|---:|---:|---:|---:|']
for threshold in [.5,.4,.3,.2]:text.append('|'+f'{threshold:.2f}'+'|'+'|'.join(fmt(q['supported_point_speedup']) if (q:=sweep(N,threshold,'warm')) else '—' for N in Ns)+'|')
text+=['', '## 完整分析、方法与限制','',body,'', '## 每N可核查产物','', '每个链接包含独立N的完整候选frontier、cold/warm扫描、失败记录和bootstrap选择频率。','']
for q in index:
 text.append(f"- N={q['N']}：[报告]({q['report']}) · [frontier]({q['frontier']}) · [分析JSON]({q['analysis']})"+(f" · [raw结果包]({q['archive']})" if q['archive'] else ''))
text+=['', '## 交付状态','', '最终总review archive由master在报告与图全部生成后自动校验。上面的archive检查针对已经完成的各N raw archive；当前报告阶段不提前宣称总archive已完成。所有结果本地保存，无active Drive sync。']
(O/'REPORT.md').write_text('\n'.join(text)+'\n')
for name,rows in [('optimized_thresholds_all.csv',threshold_rows),('QUALITY_AUDIT.csv',quality)]:
 with (O/name).open('w') as f:
  fields=list(dict.fromkeys(k for q in rows for k in q));w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
(O/'QUALITY_AUDIT.json').write_text(json.dumps(dict(all_N_ready=ready,mechanism_complete=mechanism_complete,threshold_sweep_complete=threshold_sweep_complete,per_N=quality,scope='Presentation audit only; no metric recomputation or experimental protocol change'),indent=2)+'\n');(O/'EVIDENCE_MATRIX.json').write_text(json.dumps(evidence,indent=2)+'\n');(O/'ARTIFACT_INDEX.json').write_text(json.dumps(dict(top_report=str(O/'REPORT.md'),one_page=str(O/'ONE_PAGE_SUMMARY.md'),per_N=index),indent=2)+'\n');print('ONE_STOP_REPORT_GENERATED','complete' if ready else 'partial',flush=True)
