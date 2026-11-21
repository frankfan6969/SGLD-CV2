from pathlib import Path
import json,csv
import numpy as np
R=Path(__file__).resolve().parents[2];D=R/'results/p3b_n_scaling_v1';O=R/'exports/p3b_n_scaling_v1';c=json.loads((D/'PROTOCOL.json').read_text());metadata=[];mechanism=[];efficiency=[];statuses=[];reference_gates=[]
for N in c['data']['N']:
 ND=D/f'N{N}';p=ND/'STATUS.json';statuses.append(json.loads(p.read_text()) if p.exists() else dict(N=N,status='not_completed'))
 for filename,storage in [('target/metadata.json',metadata),('mechanism_summary.json',mechanism)]:
  p=ND/filename
  if p.exists():
   v=json.loads(p.read_text());storage.extend(v if isinstance(v,list) else [v])
 p=ND/'target/reference_pair.json'
 if p.exists():reference_gates.append(dict(N=N,**json.loads(p.read_text())))
 p=O/f'N{N}/analysis.json'
 if p.exists():efficiency.extend(dict(N=N,**q) for q in json.loads(p.read_text())['threshold_sweep'])
slopes=[]
for method in ['sgld','cv1','cv2']:
 for group,scale in [('reference',None),('controlled_shell',.5),('controlled_shell',1.),('controlled_shell',2.)]:
  qs=sorted([q for q in mechanism if q['method']==method and q['group']==group and q['scale']==scale],key=lambda q:q['N'])
  for metric in ['raw_mse_mean','preconditioned_mse_mean','max_raw_remainder_across_locations','max_preconditioned_remainder_across_locations','mean_raw_anchor_distance']:
   valid=[q for q in qs if q[metric]>0];slope=float(np.polyfit(np.log([q['N'] for q in valid]),np.log([q[metric] for q in valid]),1)[0]) if len(valid)>=2 else None;slopes.append(dict(method=method,group=group,scale=scale,metric=metric,N_count=len(valid),descriptive_loglog_slope=slope,adjacent_slopes=[dict(N0=x['N'],N1=y['N'],slope=float(np.log(y[metric]/x[metric])/np.log(y['N']/x['N']))) for x,y in zip(valid[:-1],valid[1:])],scope='Descriptive four-point scaling on one nested DGP realization; no fitted asymptotic equality or independent-DGP CI'))
a=dict(protocol=c,statuses=statuses,target_metadata=metadata,reference_gates=reference_gates,mechanism=mechanism,descriptive_scaling_slopes=slopes,optimized_thresholds=efficiency);(O/'analysis.json').write_text(json.dumps(a,indent=2)+'\n')
for name,rows in [('mechanism_scaling.csv',mechanism),('descriptive_slopes.csv',slopes)]:
 with (O/name).open('w') as f:
  fields=list(dict.fromkeys(k for q in rows for k in q));w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
text=['# P3B：synthetic Bayesian logistic 的N-scaling与优化后accuracy–cost','', '**数据与问题：** d=20、AR(1) rho=.5、固定beta*使population logit SD=2、Gaussian N(0,I) prior。N={2000,8000,32000,128000}为同一最大样本的精确prefix，不重新standardize。分别检验raw/preconditioned gradient-noise scaling及每N各自优化后的实际wall-clock frontier。','', '## 完成状态','', '|N|Status|','|---:|---|']
for q in statuses:text.append(f"|{q['N']}|{q['status']}|")
text+=['', '任何reference失败的N不产生可信posterior accuracy排名，也不降低门槛重试。其余N按同一冻结规则继续。只有全部四N完成，才可以把四点趋势作为完整实验展示。','', '## Target与极端observation诊断','', '|N|Mode gradient norm|max||x_i|||max whitened design norm|Hessian condition|Mode error totruth|','|---:|---:|---:|---:|---:|---:|']
for q in metadata:text.append(f"|{q['N']}|{q['mode']['gradient_norm']:.3g}|{q['max_x_norm']:.4f}|{q['max_whitened_design_norm']:.4f}|{q['hessian_condition']:.3f}|{q['raw_mode_error_against_truth']:.4f}|")
text+=['', '每N独立计算mode、Hessian metric和高质量reference。白化使用z=(theta-mode) W_N,W_N²=H_N+1e-10I,P_N=W_N^-2；不同N使用同一128单位投影方向与同样reference大小，variance NRMSE采用统一relative定义。局部白化提高可比性，但不意味着不同N posterior完全相同。','', '## A：梯度方差scaling，固定n=16','', 'Exact finite-population variance枚举所有observations计算；SGLD/CV1/CV2随机梯度的deterministic prior/correction在噪声中心化后抵消。主要位置来自每N自己的两个reference scoring clouds，共32个固定位置；另有共同白化方向的48个shell位置及anchor。','', '|N|Method|Raw MSE|Preconditioned MSE|Max raw residual across ref locations|Median largest-observation share|','|---:|---|---:|---:|---:|---:|']
for q in sorted([q for q in mechanism if q['group']=='reference'],key=lambda q:(q['N'],q['method'])):text.append(f"|{q['N']}|{q['method']}|{q['raw_mse_mean']:.5g}|{q['preconditioned_mse_mean']:.5g}|{q['max_raw_remainder_across_locations']:.5g}|{q['median_largest_noise_share']:.3%}|")
text+=['', '### 描述性log-log斜率','', '|Method|Raw MSE slope|Preconditioned MSE slope|N points|','|---|---:|---:|---:|']
for method in ['sgld','cv1','cv2']:
 q=[q for q in slopes if q['method']==method and q['group']=='reference'];r=next(q for q in q if q['metric']=='raw_mse_mean');p=next(q for q in q if q['metric']=='preconditioned_mse_mean');fmt=lambda x:'NA' if x is None else f'{x:.3f}';text.append(f"|{method}|{fmt(r['descriptive_loglog_slope'])}|{fmt(p['descriptive_loglog_slope'])}|{r['N_count']}|")
text+=['', '理论raw upper-order基准SGLD~N²/n,CV1~N/n,CV2~1/n依赖posterior concentration与正则性；不要求实际MSE严格等于这些幂律。P_N随N改变，preconditioned斜率必须单独解释。只有四N、一个nested数据实现，拟合斜率不是渐近证明。Gaussian features无界，max||x_i||及worst-observation smoothness不保证随N保持常数。','', '## B：每N优化后的time-to-accuracy','', '每N使用共同LP-based初始h系数、六档n、有界单轮步长细化、相同候选筛选规则及独立validation。初始参数不是Bank最佳配置。每method最多12个validated候选、8新seeds×512replicas，统一phase horizon由pilot预设规则决定，不在validation后延长。','', '完整41档NRMSE(0.20–0.60)，固定whitened SW2≤0.3。下面仅列预定代表门槛；完整warm/cold、point/simultaneous-supported曲线和所有失败在各N报告与总JSON。','', '|N|NRMSE|Warm CV1 seconds|Warm CV2 seconds|Point speedup|Supported speedup|Cold point speedup|','|---:|---:|---:|---:|---:|---:|---:|']
for N in c['data']['N']:
 for threshold in [.5,.4,.3,.2]:
  warm=next((q for q in efficiency if q['N']==N and q['cost']=='warm' and q['variance_threshold']==threshold),None);cold=next((q for q in efficiency if q['N']==N and q['cost']=='cold' and q['variance_threshold']==threshold),None)
  if warm is None:continue
  vals=[]
  for m in ['cv1','cv2']:
   q=warm['methods'][m]['point_choice'];vals.append('Not Reached' if q is None else f"{q['warm_seconds']:.4f}")
  fmt=lambda x:'—' if x is None else f'{x:.3f}';text.append(f"|{N}|{threshold:.2f}|{vals[0]}|{vals[1]}|{fmt(warm['point_speedup_cv1_over_cv2'])}|{fmt(warm['supported_point_speedup'])}|{fmt(cold['point_speedup_cv1_over_cv2']) if cold else '—'}|")
text+=['', '### 如何解读效率结果','', 'Speedup=CV1/CV2达到同一joint accuracy所需time，>1表示CV2较快；只限定每N冻结验证候选子网格，不是连续参数全局最优。只一方达标不计算有限或无限speedup；各N都可能有不同的最优n,h,K。','', '取validation最小值存在选择乐观偏差。Point与simultaneous上界支持达标并列；10,000次paired whole-seed bootstrap在每draw重新计算最小值和frontier，保存Not Reached mass。Reference clouds固定、只有8独立seed clusters，同时带覆盖是近似而非严格有限样本保证。跨N的reference draws独立生成，不将41阈值或nested datasets视为独立重复。','', '## 机制与一般化限制','', '应联合查看posterior raw距离收缩、max||x_i||增长、raw/preconditioned最大Taylor residual与noise share，再解释效率是否随N持续、改善或饱和。即使典型噪声降低，也不保证所有轨迹无极端事件。', '','本轮只有一种AR(1)rho=.5、一个beta与一个nested生成实现；结果检验该受控family，不能外推为所有几何、信号或heavy-tail designs。P3A Bank实证与机制结果保留，未重跑P2 Bank以追求有利结果。','', '## 成本与产物','', '统一本地JAX CPU float64，warm计入各N本地mode fit attribution、actual setup及同步sampler/RNG；cold再加shape编译，非fresh-process timing。数据生成、reference、指标计算和归档成本单独记录，不混入sampler效率；各N数据大小不同导致的setup增长保留。','', '完整每N的protocol、reference gates、mechanism locations、pilot/validation JSON/NPZ、frontiers、failure records和校验archive均自动保存。总review archive较小；大体积raw archives分N保留。全部本地，未同步Google Drive。']
(O/'REPORT.md').write_text('\n'.join(text)+'\n');print('P3B_CROSS_ANALYSIS_COMPLETE',len(mechanism),len(efficiency),flush=True)
