"""Automatic validated-grid frontiers with paired and simultaneous uncertainty."""
from pathlib import Path
import sys,json,csv,collections,os
import numpy as np
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'src'))
from sgld_cv.transport import AccuracyMetric
N=int(os.environ['P3B_N']);D=R/'results/p3b_n_scaling_v1'/f'N{N}'/'optimized';O=R/'exports/p3b_n_scaling_v1'/f'N{N}';A=R/'results/p3b_n_scaling_v1'/f'N{N}'/'target';c=json.loads((D/'PROTOCOL.json').read_text());freeze=json.loads((D/'FROZEN_CANDIDATES.json').read_text());runs=json.loads((D/'validation_summary.json').read_text());pilot=json.loads((D/'pilot_summary.json').read_text());assert len(runs)==len(freeze['candidates'])*8;assert json.loads((D/'PREFLIGHT.json').read_text())['status']=='passed';assert json.loads((D/'seed_audit.json').read_text())['passed'];a=np.load(A/'target.npz');s=np.load(A/'scoring.npz');anchor,W=a['anchor'],a['W'];d=len(anchor);refs=[s['first'],s['second']];metric=AccuracyMetric('standardized_sliced_w2',anchor,W,s['projections']);ks=[k for k in c['validation']['checkpoints'] if k<=freeze['max_iterations']];groups=collections.defaultdict(list)
for r in runs:groups[r['method'],r['batch'],r['h']].append(r)
rows=[];means=[];times=[];coldtimes=[];failures=[];reconstructed=0
with threadpool_limits(limits=1):
 for (m,n,h),rr in sorted(groups.items()):
  rr=sorted(rr,key=lambda r:r['seed']);assert [r['seed'] for r in rr]==c['validation']['seeds'];eligible=not any(r['failed_replicas'] or r['budget_censored'] or r['completed_iterations']!=freeze['max_iterations'] for r in rr)
  for r in rr:
   if r['failed_replicas'] or r['budget_censored']:failures.append({k:r[k] for k in ['id','method','batch','h','seed','failed_replicas','failed_at','budget_censored','completed_iterations']})
  for k in ks:
   ps=[next((p for p in r['checkpoints'] if p['iteration']==k),None) for r in rr];q=dict(method=m,batch=n,h=h,K=k,eligible=eligible,failed_runs=sum(r['failed_replicas']>0 for r in rr),failed_replicas=sum(r['failed_replicas'] for r in rr),total_replicas=4096,budget_censored_runs=sum(r['budget_censored'] for r in rr))
   if eligible:
    v=np.array([[[z[t] for t in ['sw2','variance_nrmse']] for z in p['references']] for p in ps]);means.append(v);times.append([p['warm_total_seconds'] for p in ps]);coldtimes.append([p['cold_total_seconds'] for p in ps]);q['eligible_index']=len(means)-1
    for r,p in zip(rr,ps):
     z=np.load(D/f"{r['id']}_K{k}.npz");assert np.all(z['failed_at']==0);avg=z['functional_sums']/k;var=avg[:,d:]-avg[:,:d]**2
     for j,ref in enumerate(refs):
      vref=((ref-anchor)@W).var(0);got=float(np.sqrt(np.mean(((var-vref)/vref)**2,axis=0)).mean());np.testing.assert_allclose(got,p['references'][j]['variance_nrmse'],rtol=1e-11,atol=1e-12)
      # One full SW2 reconstruction per config/seed, plus every NRMSE checkpoint.
      if k==ks[-1]:np.testing.assert_allclose(metric.distance(z['endpoints'],ref),p['references'][j]['sw2'],rtol=1e-11,atol=1e-12)
     np.testing.assert_allclose(p['cold_total_seconds']-p['warm_total_seconds'],r['compile_seconds'],rtol=1e-11,atol=1e-11)
     if k==ks[-1]:np.testing.assert_allclose(p['warm_total_seconds'],r['anchor_seconds_attributed']+r['setup_seconds']+r['sampling_seconds'],rtol=1e-12)
     reconstructed+=1
   else:q.update(sw2=None,variance_nrmse=None,warm_seconds=None,cold_seconds=None,reason='Complete configuration excluded, no survivor scoring')
   rows.append(q)
ci=lambda x:np.quantile(x,[.025,.975],axis=0);boot=np.random.default_rng(c['bootstrap']['seed']).integers(0,8,(c['bootstrap']['draws'],8));valid=[q for q in rows if q['eligible']];frontiers=[];sweeps=[];selection_freq=[];bootstrap_frontier=[];band_quantile=None
if means:
 V=np.stack(means,axis=1);T=np.array(times).T;C=np.array(coldtimes).T;point=V.mean(0);bs=np.concatenate([V[b].mean(1) for b in np.array_split(boot,20)],axis=0);bt=np.concatenate([T[b].mean(1) for b in np.array_split(boot,20)],axis=0);bc=np.concatenate([C[b].mean(1) for b in np.array_split(boot,20)],axis=0);se=bs.std(0,ddof=1);scale=np.maximum(se,1e-12);maxstat=np.max(np.abs((bs-point)/scale),axis=(1,2,3));band_quantile=float(np.quantile(maxstat,.95));upper=(point+band_quantile*se).max(1);lower=np.maximum(0,point-band_quantile*se).max(1);acc=point.max(1);bacc=bs.max(2)
 for i,q in enumerate(valid):
  q.update(sw2=float(acc[i,0]),variance_nrmse=float(acc[i,1]),warm_seconds=float(T[:,i].mean()),cold_seconds=float(C[:,i].mean()),sw2_ci95=ci(bacc[:,i,0]).tolist(),variance_nrmse_ci95=ci(bacc[:,i,1]).tolist(),warm_ci95=ci(bt[:,i]).tolist(),cold_ci95=ci(bc[:,i]).tolist(),sw2_simultaneous_upper=float(upper[i,0]),variance_simultaneous_upper=float(upper[i,1]),sw2_simultaneous_lower=float(lower[i,0]),variance_simultaneous_lower=float(lower[i,1]))
 def nondominated(indices,costkey):
  p=np.array([[valid[i][costkey],valid[i]['sw2'],valid[i]['variance_nrmse']] for i in indices]);dom=np.all(p[:,None,:]<=p[None,:,:],axis=2)&np.any(p[:,None,:]<p[None,:,:],axis=2);return [indices[j] for j in np.flatnonzero(~dom.any(0))]
 for cost,tt in [('warm',bt),('cold',bc)]:
  for m in c['methods']:
   ii=[i for i,q in enumerate(valid) if q['method']==m]
   if not ii:continue
   f=nondominated(ii,cost+'_seconds');frontiers.append(dict(cost=cost,method=m,eligible_indices=f,scope='3D descriptive nondominated (time,SW2,variance), not a single-threshold ranking'))
  for threshold in c['threshold_sweep']['variance']:
   entry=dict(cost=cost,variance_threshold=float(threshold),methods={});bmin={};selected={}
   for m in c['methods']:
    ii=np.array([i for i,q in enumerate(valid) if q['method']==m],dtype=int);pointmask=(acc[ii,0]<=.3)&(acc[ii,1]<=threshold) if len(ii) else np.array([],bool);bandmask=(upper[ii,0]<=.3)&(upper[ii,1]<=threshold) if len(ii) else np.array([],bool)
    def choose(mask):
     jj=ii[mask]
     if not len(jj):return None
     i=int(jj[np.argmin([valid[j][cost+'_seconds'] for j in jj])]);return {k:valid[i][k] for k in ['eligible_index','method','batch','h','K','sw2','variance_nrmse',cost+'_seconds']}
    choice=choose(pointmask);supported=choose(bandmask)
    if len(ii):
     mask=(bacc[:,ii,0]<=.3)&(bacc[:,ii,1]<=threshold);values=np.where(mask,tt[:,ii],np.inf);arg=np.argmin(values,1);bm=values[np.arange(len(boot)),arg];ids=ii[arg];hit=np.isfinite(bm);selection_freq.extend(dict(cost=cost,threshold=float(threshold),method=m,eligible_index=int(i),bootstrap_selection_fraction=float(np.mean(hit&(ids==i)))) for i in ii if np.any(hit&(ids==i)))
    else:bm=np.full(len(boot),np.inf);hit=np.zeros(len(boot),bool)
    bmin[m]=bm;entry['methods'][m]=dict(point_choice=choice,simultaneous_supported_choice=supported,bootstrap_not_reached_fraction=float(np.mean(~hit)),conditional_time_ci95=ci(bm[hit]).tolist() if hit.any() else None)
   both=np.isfinite(bmin['cv1'])&np.isfinite(bmin['cv2']);ch=[entry['methods'][m]['point_choice'] for m in c['methods']];bandch=[entry['methods'][m]['simultaneous_supported_choice'] for m in c['methods']];entry.update(point_speedup_cv1_over_cv2=ch[0][cost+'_seconds']/ch[1][cost+'_seconds'] if all(ch) else None,supported_point_speedup=bandch[0][cost+'_seconds']/bandch[1][cost+'_seconds'] if all(bandch) else None,bootstrap_both_reached_fraction=float(both.mean()),conditional_speedup_ci95=ci(bmin['cv1'][both]/bmin['cv2'][both]).tolist() if both.any() else None);sweeps.append(entry)

 # Recompute each method's3D nondominated set per paired bootstrap draw.
 for cost,tt in [('warm',bt),('cold',bc)]:
  for m in c['methods']:
   ii=np.array([i for i,q in enumerate(valid) if q['method']==m],dtype=int)
   if not len(ii):continue
   count=np.zeros(len(ii),dtype=int)
   for start in range(0,len(boot),100):
    pp=np.concatenate([tt[start:start+100,ii,None],bacc[start:start+100,ii]],axis=2)
    le=np.all(pp[:,:,None,:]<=pp[:,None,:,:],axis=3);lt=np.any(pp[:,:,None,:]<pp[:,None,:,:],axis=3);count+=np.sum(~np.any(le & lt,axis=1),axis=0)
   bootstrap_frontier.extend(dict(cost=cost,method=m,eligible_index=int(i),bootstrap_nondominated_frequency=float(v/len(boot))) for i,v in zip(ii,count))

# Pilot minima not interchangeable with frozen validated-candidate minima.
validation=dict(runs=len(runs),pilot_runs=len(pilot),candidate_count=len(groups),eligible_configs=sum(all(q['eligible'] for q in rows if (q['method'],q['batch'],q['h'])==key) for key in groups),failed_runs=len([r for r in runs if r['failed_replicas']]),failed_replicas=sum(r['failed_replicas'] for r in runs),budget_censored_runs=sum(r['budget_censored'] for r in runs),nrmses_reconstructed=reconstructed,simultaneous_maxstat_q95=band_quantile,band_family='All eligible candidates x all checkpoints x2 references x2 accuracy means; maxabs standardized bootstrap; conditional on fixed refs; approximate8-seed calibration')
analysis=dict(validation=validation,configuration_checkpoints=rows,frontiers=frontiers,threshold_sweep=sweeps,bootstrap_selection_frequencies=selection_freq,bootstrap_frontier_frequencies=bootstrap_frontier,failures=failures);(O/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
with (O/'configuration_checkpoints.csv').open('w') as f:
 fields=list(dict.fromkeys(k for q in rows for k in q));w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
text=['# Synthetic logistic P3B: per-N optimized accuracy–cost comparison','', '**范围：预先冻结的有界搜索，独立验证后的候选网格。不是连续参数空间的全局最优。**','',f"Pilot完成{len(pilot)} runs；冻结{len(groups)}个(n,h)候选；独立验证{len(runs)} runs，每run512 replicas，8个新seeds。统一验证horizon={freeze['max_iterations']}。验证中{validation['failed_runs']} runs含{validation['failed_replicas']}个失败replicas，{validation['budget_censored_runs']}个预算截断runs。",'', '## 搜索与公平性','', '- 同一N-specific synthetic target、Gaussian prior、full-H metric、reference clouds与投影。CV1/CV2各有6档batch、6档初始h，同一最多2档补充步长规则；只补一轮。','- 初始h由实际LP构造，不沿用Bank最优步长。非支配筛选同时考虑warm/cold time、SW2、variance NRMSE；每method/batch最多2个稳定候选，保留batch覆盖。','- 新validation seeds从未参与任何pilot或此前实验。候选与统一horizon先冻结，之后不补调参。','- 所有失败保留；任一seed失败或不完整就排除该完整(n,h)配置全部checkpoint，绝不只评分成功replicas。','', '## 主要结果：warm time联合精度扫描','', 'SW2≤0.3固定，NRMSE门槛从0.60到0.20全部报告。T是冻结且通过验证的候选网格内最短setup-inclusive wall-clock。Point列按误差点估计达标；Supported列按整个accuracy family的95%同时上界达标，不能混为一谈。','', '|NRMSE|CV1 point seconds(n,h,K)|CV2 point seconds(n,h,K)|Point speedup [pointwise95%CI]|Supported CV1 seconds|Supported CV2 seconds|','|---|---|---|---|---:|---:|']
def choicefmt(q,cost='warm'):
 return 'Not Reached' if q is None else f"{q[cost+'_seconds']:.4f} ({q['batch']},{q['h']:.5g},{q['K']})"
for q in reversed([q for q in sweeps if q['cost']=='warm']):
 p=q['methods'];speed='—' if q['point_speedup_cv1_over_cv2'] is None else f"{q['point_speedup_cv1_over_cv2']:.3f} [{q['conditional_speedup_ci95'][0]:.3f},{q['conditional_speedup_ci95'][1]:.3f}]";support=lambda z:'Not Reached' if z is None else f"{z['warm_seconds']:.4f}";text.append(f"|{q['variance_threshold']:.2f}|{choicefmt(p['cv1']['point_choice'])}|{choicefmt(p['cv2']['point_choice'])}|{speed}|{support(p['cv1']['simultaneous_supported_choice'])}|{support(p['cv2']['simultaneous_supported_choice'])}|")
warm_sweep=[q for q in sweeps if q['cost']=='warm']
faster=sum(q['point_speedup_cv1_over_cv2'] is not None and q['point_speedup_cv1_over_cv2']>1 for q in warm_sweep);slower=sum(q['point_speedup_cv1_over_cv2'] is not None and q['point_speedup_cv1_over_cv2']<1 for q in warm_sweep);only_cv2=sum(q['methods']['cv2']['point_choice'] is not None and q['methods']['cv1']['point_choice'] is None for q in warm_sweep);only_cv1=sum(q['methods']['cv1']['point_choice'] is not None and q['methods']['cv2']['point_choice'] is None for q in warm_sweep)
text+=['', '### 自动结果摘要', '', f'完整41档warm point-estimate扫描中：CV2较快{faster}档，CV1较快{slower}档，仅CV2达标{only_cv2}档，仅CV1达标{only_cv1}档；其余为两者未达标或相同时间。相邻阈值高度相关，不能把这些数量当作独立重复试验。', '']
text+=['', '完整warm/cold扫描、每候选checkpoint及bootstrap最优配置选择频率见analysis.json/CSV。Frontier图显示所有有效点和逐方法非支配边界，不只展示有利阈值。','', '## 如何解读','', '不同阈值的胜负可受每步开销、误差改善与离散checkpoint跳变共同影响；不要求优势单调。仅一个方法在预算内达标时，只能报告可达精度差异，不计算有限或无限speedup。即使同checkpoint首次达标，也不能排除中间iteration差异。','', '验证后从多个候选中取最小值仍有选择偏差，描述性frontier可能偏乐观。同时accuracy band覆盖所有有效候选、checkpoint、两项指标及两个references，提供较强但近似的达标支持，不证明全局最优成本。Bootstrap每次重新计算最小值；CI若排除Not Reached draws属于conditional interval，未达标比例完整保留在JSON，不能解读为未来成功概率。','', '## 不确定性与验证','', f"10,000次paired whole-seed bootstrap；同时带maxstat临界值={band_quantile}。置信区间条件于固定reference，未计reference Monte Carlo误差、anchor计时波动或全部未来硬件波动。只有8个独立seed clusters，同时带校准是近似，不是有限样本严格覆盖保证。",f"重建{reconstructed}个checkpoint的NRMSE，逐配置/seed重建末端SW2并核对time identities。源hash、reference gates、seed隔离及NumPy float64/JAX fixed-input preflight均通过。",'', '## 成本口径与限制','', '- Warm包括此前本地测得的mode fit attribution、本次prepare/state setup和同步JAX sampler/RNG；cold再加实际shape-specific compile attribution。缓存顺序影响cold，它不是fresh-process冷启动。','- Full-H下CV1也构造Hessian，CV2共享已有Hessian；setup记录聚合Hessian/metric分解，不能事后拆出准确独立组件。','- scoring/functionals/reference/storage成本单独记录，不纳入sampler ranking；绝对CPU时间不能与原A100/T4时间合并排名。','- 各算法比较只针对该N-specific target；不同N的posterior不同，白化改善精度可比性但不使target相同。','- P3A结果保留；本轮是新nested synthetic family上的前瞻冻结实验，但只有一个数据生成实现。Point及supported前沿均只在pilot冻结的validated候选子网格上定义。','', '## 失败记录','']
if failures:
 text+=['|Method|n|h|seed|failed replicas|first failure|censored|','|---|---:|---:|---:|---:|---:|---|']
 for r in failures:text.append(f"|{r['method']}|{r['batch']}|{r['h']:.6g}|{r['seed']}|{r['failed_replicas']}|{min([t for t in r['failed_at'] if t],default=None)}|{r['budget_censored']}|")
else:text.append('验证期间未观察到失败。零失败不能证明普遍长期稳定。')
text+=['', '## 输出与停止规则','', 'raw JSON/NPZ、协议、候选冻结、seeds、source code、完整曲线、分析JSON/CSV及校验归档自动生成。全部本地保存，未同步Google Drive。每N完成后master按冻结顺序继续；不追加参数搜索。']
(O/'REPORT.md').write_text('\n'.join(text)+'\n');print('ANALYSIS_COMPLETE',json.dumps(validation),flush=True)
