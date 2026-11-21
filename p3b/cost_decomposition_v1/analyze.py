"""Existing-log accounting only: no sampler runs or accuracy redefinition."""
from pathlib import Path
import json,hashlib,csv
import numpy as np
R=Path(__file__).resolve().parents[2];H=Path(__file__).parent;O=R/'exports/p3b_cost_decomposition_v1';D=R/'results/p3b_n_scaling_v1';S=R/'exports/p3b_n_scaling_v1';a=json.loads((S/'analysis.json').read_text());Ns=a['protocol']['data']['N'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();seeds_by_N={};runs_by_N={};inputs=[S/'analysis.json']
for N in Ns:
 p=D/f'N{N}/optimized/validation_summary.json';runs_by_N[N]=json.loads(p.read_text());inputs.extend([p,D/f'N{N}/optimized/PROTOCOL.json']);seeds_by_N[N]=json.loads((D/f'N{N}/optimized/PROTOCOL.json').read_text())['validation']['seeds']
protocol=dict(stage='Post-result descriptive cost decomposition of existing P3B',datasets=Ns,representative_thresholds=[.5,.4,.3,.2],full_sweep='All41 thresholds, warm/cold, point/simultaneous-supported frozen-grid optima',prefix_sampling='At selected checkpoint K, sampling_seconds=recorded_warm_total-anchor_seconds_attributed-actual_setup_seconds. Never divide full4096-horizon time by a shorter selected K.',unit='Per-step cost is wall-clock for one batched update of512 replicas; not a serial single-replica cost.',identity='S=(K1/K2)*(c1/c2)*D, c_m=prefix_sampling_m/K_m, D=(1+A1/Q1)/(1+A2/Q2), A=anchor+setup,Q=prefix_sampling. Ratios of endpoint factors multiply exactly to S(Nmax)/S(Nmin).',counterfactuals='Sampling-only and removing only anchor attribution are bookkeeping views, not production rankings or rerun experiments.',matched_controls='Compare raw sampling_seconds/K at common validated n,h,horizon and same seed within eachN; canonical h rounding to12 decimals only resolves numerical1e-14 differences. Retain failures explicitly; no success-lane selection.',uncertainty='10000 paired whole-seed bootstrap, fixed selected(n,h,K) for decomposition; existing reselected-optimum frontier CIs remain separate. Confidence conditional on chosen config/reference and measured implementation. This is an accounting decomposition, not a randomized causal identification of all runtime differences.',bootstrap_seed=211900,draws=10000,input_hashes={str(p.relative_to(R)):sha(p) for p in inputs})
(H/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n');(O/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n');boot=np.random.default_rng(protocol['bootstrap_seed']).integers(0,8,(10000,8));ci=lambda x:np.quantile(x,[.025,.975]).tolist();extracts={};source_records=[]
def extract(N,choice,cost):
 key=(N,choice['method'],choice['batch'],round(choice['h'],12),choice['K'])
 if key in extracts:return extracts[key]
 rs=sorted([r for r in runs_by_N[N] if r['method']==choice['method'] and r['batch']==choice['batch'] and abs(r['h']-choice['h'])<1e-10],key=lambda r:r['seed']);assert [r['seed'] for r in rs]==seeds_by_N[N];assert all(r['failed_replicas']==0 and not r['budget_censored'] for r in rs);k=choice['K'];values=[]
 for r in rs:
  p=next(p for p in r['checkpoints'] if p['iteration']==k);anchor=r['anchor_seconds_attributed'];setup=r['setup_seconds'];warm=p['warm_total_seconds'];sampling=warm-anchor-setup;assert sampling>=0;np.testing.assert_allclose(p['cold_total_seconds'],warm+r['compile_seconds'],rtol=1e-12,atol=1e-12)
  if k==r['completed_iterations']:np.testing.assert_allclose(sampling,r['sampling_seconds'],rtol=1e-12,atol=1e-12)
  values.append([anchor,setup,sampling,r['compile_seconds']]);source_records.append(dict(N=N,method=r['method'],batch=r['batch'],h=r['h'],K=k,seed=r['seed'],anchor_seconds=anchor,setup_seconds=setup,prefix_sampling_seconds=sampling,compile_seconds=r['compile_seconds'],warm_seconds=warm,cold_seconds=p['cold_total_seconds']))
 v=np.array(values);mean=v.mean(0);A=mean[0]+mean[1];Q=mean[2];q=dict(N=N,method=choice['method'],batch=choice['batch'],h=choice['h'],K=k,replicas=512,anchor_seconds=float(mean[0]),setup_seconds=float(mean[1]),A_seconds=float(A),prefix_sampling_seconds=float(Q),per_batched_step_ms=float(Q/k*1000),compile_seconds=float(mean[3]),warm_seconds=float(A+Q),cold_seconds=float(A+Q+mean[3]),warm_setup_share=float(A/(A+Q)),warm_anchor_share=float(mean[0]/(A+Q)),sampling_step_ci95_ms=ci(v[:,2][boot].mean(1)/k*1000),seed_component_values=v.tolist())
 np.testing.assert_allclose(q[cost+'_seconds'],choice[cost+'_seconds'],rtol=1e-12,atol=1e-12);extracts[key]=q;return q
rows=[];pairs=[]
for sweep in a['optimized_thresholds']:
 N=sweep['N'];threshold=sweep['variance_threshold'];cost=sweep['cost']
 for typ in ['point_choice','simultaneous_supported_choice']:
  selected={}
  for method in ['cv1','cv2']:
   choice=sweep['methods'][method][typ]
   if choice is not None:
    q=extract(N,choice,cost);selected[method]=q;rows.append(dict(cost=cost,choice_type=typ,threshold=threshold,**{k:v for k,v in q.items() if k!='seed_component_values'}))
  if len(selected)<2:continue
  q1,q2=selected['cv1'],selected['cv2'];iteration=q1['K']/q2['K'];step=q1['per_batched_step_ms']/q2['per_batched_step_ms'];sampling=q1['prefix_sampling_seconds']/q2['prefix_sampling_seconds'];S=q1['warm_seconds']/q2['warm_seconds'];dilution=(1+q1['A_seconds']/q1['prefix_sampling_seconds'])/(1+q2['A_seconds']/q2['prefix_sampling_seconds']);noanchor=(q1['setup_seconds']+q1['prefix_sampling_seconds'])/(q2['setup_seconds']+q2['prefix_sampling_seconds']);np.testing.assert_allclose(S,iteration*step*dilution,rtol=1e-12);v1=np.array(q1['seed_component_values']);v2=np.array(q2['seed_component_values']);b1=v1[boot].mean(1);b2=v2[boot].mean(1);ratios=(b1[:,:3].sum(1)/b2[:,:3].sum(1));p=dict(N=N,threshold=threshold,cost=cost,choice_type=typ,cv1_choice={k:q1[k] for k in ['batch','h','K']},cv2_choice={k:q2[k] for k in ['batch','h','K']},iteration_factor=float(iteration),per_step_factor=float(step),sampling_only_speedup=float(sampling),setup_dilution_factor=float(dilution),warm_speedup=float(S),cold_speedup=float(q1['cold_seconds']/q2['cold_seconds']),without_anchor_attribution_speedup=float(noanchor),fixed_selected_choice_warm_speedup_ci95=ci(ratios),cv1_setup_share=q1['warm_setup_share'],cv2_setup_share=q2['warm_setup_share']);pairs.append(p)
# Endpoint multiplicative decomposition: exact factors, not causal percentages.
endpoints=[]
for cost in ['warm','cold']:
 for typ in ['point_choice','simultaneous_supported_choice']:
  for threshold in sorted(set(q['threshold'] for q in pairs)):
   lo=next((q for q in pairs if q['N']==Ns[0] and q['threshold']==threshold and q['cost']==cost and q['choice_type']==typ),None);hi=next((q for q in pairs if q['N']==Ns[-1] and q['threshold']==threshold and q['cost']==cost and q['choice_type']==typ),None)
   if lo is None or hi is None:continue
   rK=hi['iteration_factor']/lo['iteration_factor'];rc=hi['per_step_factor']/lo['per_step_factor'];rD=hi['setup_dilution_factor']/lo['setup_dilution_factor'];ratio=hi['warm_speedup']/lo['warm_speedup'];np.testing.assert_allclose(rK*rc*rD,ratio,rtol=1e-12);endpoints.append(dict(cost_selected=cost,choice_type=typ,threshold=threshold,N0=Ns[0],N1=Ns[-1],iteration_factor_change=float(rK),per_step_factor_change=float(rc),setup_dilution_change=float(rD),warm_speedup_ratio_Nmax_over_Nmin=float(ratio),warm_speedup_percent_decline=float((1-ratio)*100),log_changes=dict(iteration=float(np.log(rK)),per_step=float(np.log(rc)),setup_dilution=float(np.log(rD)),total=float(np.log(ratio)))))
# Matched validated configurations distinguish batch-choice effects from algorithm overhead.
matched=[]
for N in Ns:
 groups={}
 for r in runs_by_N[N]:groups.setdefault((r['batch'],round(r['h'],12)),{}).setdefault(r['method'],[]).append(r)
 for (n,h),methods in sorted(groups.items()):
  if set(methods)!=set(['cv1','cv2']):continue
  rr={m:sorted(rs,key=lambda r:r['seed']) for m,rs in methods.items()};eligible=all(len(rs)==8 and all(not r['budget_censored'] and not r['failed_replicas'] for r in rs) for rs in rr.values());q=dict(N=N,batch=n,h_canonical=h,eligible=eligible,scope='Same n,h,horizon,R512,seeds; synchronized full-horizon sampler includes RNG; no setup orcompile')
  if eligible:
   assert [r['seed'] for r in rr['cv1']]==[r['seed'] for r in rr['cv2']];k={r['completed_iterations'] for rs in rr.values() for r in rs};assert len(k)==1;steps={m:np.array([r['sampling_seconds']/r['completed_iterations'] for r in rs]) for m,rs in rr.items()};ratio=steps['cv2'].mean()/steps['cv1'].mean();q.update(K=next(iter(k)),cv1_step_ms=float(steps['cv1'].mean()*1000),cv2_step_ms=float(steps['cv2'].mean()*1000),cv2_over_cv1_step_ratio=float(ratio),ratio_ci95=ci(steps['cv2'][boot].mean(1)/steps['cv1'][boot].mean(1)))
  matched.append(q)
analysis=dict(protocol=protocol,optimal_configurations=rows,speedup_decomposition=pairs,endpoint_factor_changes=endpoints,matched_configuration_controls=matched,source_prefix_components=source_records,checks=dict(all_input_hashes_passed=all(sha(R/p)==v for p,v in protocol['input_hashes'].items()),unique_selected_prefixes=len(extracts),prefix_seed_records=len(source_records),decomposition_identity_checks=len(pairs),endpoint_identity_checks=len(endpoints),new_sampling_runs=0));(O/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
for name,rr in [('optimal_configurations_all.csv',rows),('speedup_decomposition_all.csv',pairs),('endpoint_factors_all.csv',endpoints),('matched_per_step_controls.csv',matched),('source_prefix_components.csv',source_records)]:
 with (O/name).open('w') as f:
  fields=list(dict.fromkeys(k for q in rr for k in q));w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rr)
text=['# P3B：speedup下降的现有日志成本分解','', '**结论：不能将下降简单归因于CV2 correction随N变贵。** 在固定d=20、缓存likelihood Hessian的实现中，CV2 correction每步不需要遍历N；full-H CV1也需要构造Hessian。现有记录首先支持共同mode/setup成本对快速CV2的加性稀释，同时部分门槛下CV1重新调参缩小了采样工作量差距。','', '本分析没有新采样。最优是既有冻结validated候选子网格中的point-estimate或simultaneous-supported选择。四档代表门槛外，全部41档、warm/cold选择完整保留。','', '## 1. 最优配置与实际prefix成本','', '每步ms是512 replicas一起更新一次的batched wall-clock，不是单条链的串行耗时。Setup A包含mode fit attribution和实际prepare/state setup；采样时间Q只到所选K，非4096全程时间。','', '|NRMSE|N|Method|n|h|K|Mode s|Prepare s|Sampling Q s|Batched step ms|Warm total s|A/total|','|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for t in [.5,.4,.3,.2]:
 for q in sorted([q for q in rows if q['threshold']==t and q['cost']=='warm' and q['choice_type']=='point_choice'],key=lambda q:(q['N'],q['method'])):text.append(f"|{t:.2f}|{q['N']}|{q['method']}|{q['batch']}|{q['h']:.5g}|{q['K']}|{q['anchor_seconds']:.6f}|{q['setup_seconds']:.6f}|{q['prefix_sampling_seconds']:.6f}|{q['per_batched_step_ms']:.4f}|{q['warm_seconds']:.6f}|{q['warm_setup_share']:.1%}|")
text+=['', '## 2. Exact accounting identity','', r'$S=\frac{K_1}{K_2}\frac{c_1}{c_2}\frac{1+A_1/Q_1}{1+A_2/Q_2}$，其中$Q_m=K_m c_m$。前两项是sampling-only speedup，最后一项是additive setup dilution。所有等式从原日志逐seed重建并数值校验。','', '|NRMSE|N|K1/K2|c1/c2|Sampling-only speedup|Setup dilution|Total warm speedup|','|---:|---:|---:|---:|---:|---:|---:|']
for t in [.5,.4,.3,.2]:
 for q in sorted([q for q in pairs if q['threshold']==t and q['cost']=='warm' and q['choice_type']=='point_choice'],key=lambda q:q['N']):text.append(f"|{t:.2f}|{q['N']}|{q['iteration_factor']:.3f}|{q['per_step_factor']:.3f}|{q['sampling_only_speedup']:.3f}|{q['setup_dilution_factor']:.3f}|{q['warm_speedup']:.3f}|")
text+=['', '## 3. N=2000→128000：下降因子','', '三项变化因子相乘=最终speedup/最初speedup。小于1表示压低speedup，大于1表示抵消下降。这是精确bookkeeping，不是随机化因果贡献百分比。','', '|NRMSE|Iteration factor change|Per-step ratio change|Setup dilution change|Product|Speedup decline|','|---:|---:|---:|---:|---:|---:|']
for q in endpoints:
 if q['cost_selected']=='warm' and q['choice_type']=='point_choice' and q['threshold'] in [.5,.4,.3,.2]:text.append(f"|{q['threshold']:.2f}|{q['iteration_factor_change']:.3f}|{q['per_step_factor_change']:.3f}|{q['setup_dilution_change']:.3f}|{q['warm_speedup_ratio_Nmax_over_Nmin']:.3f}|{q['warm_speedup_percent_decline']:.1f}%|")
text+=['', '## 4. 同n,h的每步控制比较','', '不同最优n对应的每步时间不能被叫做纯算法overhead。因此另外列出所有common validated(n,h)；若有失败则保留ineligible标记，不只选成功replicas。','', '|N|n|h|CV1 batched ms/step|CV2 batched ms/step|CV2/CV1 [95%CI]|','|---:|---:|---:|---:|---:|---|']
for q in matched:
 if q['eligible']:text.append(f"|{q['N']}|{q['batch']}|{q['h_canonical']:.5g}|{q['cv1_step_ms']:.4f}|{q['cv2_step_ms']:.4f}|{q['cv2_over_cv1_step_ratio']:.3f} [{q['ratio_ci95'][0]:.3f},{q['ratio_ci95'][1]:.3f}]|")
 else:text.append(f"|{q['N']}|{q['batch']}|{q['h_canonical']:.5g}|Ineligible|Ineligible|—|")
text+=['', '## 5. 机制解释与边界','', '1. **共同固定成本稀释**：mode和full-H准备随N增长，双方都承担；CV2采样本来更快，相同额外setup对其总时间占比更大，能将总比值拉向1。这不意味着应把setup从正式排名删掉。','2. **CV1调参变化**：不同精度下可能换更大h、更小batch或更少K；把这些实际选择连同c一起看，不能只看raw梯度MSE。','3. **原始与白化噪声不同**：P3B的CV1 raw MSE约N增长，但preconditioned slope约0.013，近乎平坦；CV2 preconditioned噪声约N^-1.018下降。实际full-H dynamics面临的不是CV1噪声按N无限增长。','4. **网格边界**：本次CV2在代表门槛的最优配置均采用n=1；宽松目标已在最早K=128达标。更多方差降低未必能在现有网格中继续减少batch或iterations，不应解释为连续参数空间已无收益。','5. **未完全隔离运行机制**：历史wall-clock受cache、thermal/load及随机调度影响；相同n,h控制帮助判断，但不是专门的component microbenchmark。缓存Hessian乘法的独立计时没有保存，不能事后给出精确correction-only成本。','', '## 不确定性与交付','', '本分解的bootstrap固定已选配置；不是重新挑最优候选的frontier CI。原P3B的reselected bootstrap/simultaneous bands保持原样。去除mode或setup的速度比只是成本口径敏感性，不是新的正式效率结果。无新实验、无数据或threshold修改。','', 'analysis.json、全部41档optimal_configurations_all.csv、speedup_decomposition_all.csv、endpoint_factors_all.csv、matched_per_step_controls.csv、source_prefix_components.csv、图表和校验archive均本地保存。']
(O/'REPORT.md').write_text('\n'.join(text)+'\n')
print(json.dumps([q for q in endpoints if q['cost_selected']=='warm' and q['choice_type']=='point_choice' and q['threshold'] in [.5,.4,.3,.2]],indent=2))
