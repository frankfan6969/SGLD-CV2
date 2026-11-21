"""Adapt verified P3A search/analysis unchanged in algebra to explicit P3B per-N paths."""
from pathlib import Path
import ast
R=Path(__file__).resolve().parents[2];H=Path(__file__).parent;S=R/'p3a/log1p_optimized_v1'
p=(S/'pipeline.py').read_text().replace("H=Path(__file__).parent;O=R/'results/p3a_log1p_optimized_v1';E=R/'exports/p3a_log1p_optimized_v1';A=R/'exports/p3a_feature_sensitivity_v1/log1p_previous';c=json.loads((H/'PROTOCOL.json').read_text())", "H=Path(__file__).parent;N=int(os.environ['P3B_N']);O=R/'results/p3b_n_scaling_v1'/f'N{N}'/'optimized';E=R/'exports/p3b_n_scaling_v1'/f'N{N}';A=R/'results/p3b_n_scaling_v1'/f'N{N}'/'target';c=json.loads((O/'PROTOCOL.json').read_text())")
p=p.replace("H/'analyze.py'","H/'optimized_analyze.py'").replace("H/'plot.py'","H/'optimized_plot.py'").replace("E/'P3A_Log1p_Optimized_v1.zip'","E/f'P3B_N{N}_Optimized_v1.zip'").replace("R/'p3a/STATUS.json'","R/'p3b/STATUS.json'").replace('Log1p optimized','Synthetic per-N optimized').replace('Log1p Bank optimized','Synthetic per-N optimized').replace('No further Bank tuning or P3B launched.','Per-N result complete; master continues frozen P3B schedule.').replace('no further Bank tuning or P3B launched.','Per-N result complete; master continues frozen P3B schedule.')
(H/'optimized_pipeline.py').write_text(p)
s=(S/'analyze.py').read_text().replace('import sys,json,csv,collections','import sys,json,csv,collections,os').replace("D=R/'results/p3a_log1p_optimized_v1';O=R/'exports/p3a_log1p_optimized_v1';A=R/'exports/p3a_feature_sensitivity_v1/log1p_previous'", "N=int(os.environ['P3B_N']);D=R/'results/p3b_n_scaling_v1'/f'N{N}'/'optimized';O=R/'exports/p3b_n_scaling_v1'/f'N{N}';A=R/'results/p3b_n_scaling_v1'/f'N{N}'/'target'")
s=s.replace('Log1p Bank: optimized accuracy–cost comparison','Synthetic logistic P3B: per-N optimized accuracy–cost comparison').replace('同一log1p target','同一N-specific synthetic target').replace('不沿用原Bank最优步长','不沿用Bank最优步长').replace('各算法比较只针对本log1p target；原Bank/Winsorization posterior不同，本轮不能单独证明变换是全部效率变化的唯一原因。','各算法比较只针对该N-specific target；不同N的posterior不同，白化改善精度可比性但不使target相同。').replace('早先固定n,h、threshold sweep与原P3A结果保留，本轮属于探索性扩展。','P3A结果保留；本轮是新nested synthetic family上的前瞻冻结实验，但只有一个数据生成实现。').replace('到本轮结束即停止Bank扩展，不自动进入P3B。','每N完成后master按冻结顺序继续；不追加参数搜索。')
(H/'optimized_analyze.py').write_text(s)
s=(S/'plot.py').read_text().replace('import json','import json,os',1).replace("O=R/'exports/p3a_log1p_optimized_v1'", "N=int(os.environ['P3B_N']);O=R/'exports/p3b_n_scaling_v1'/f'N{N}'").replace('Log1p Bank: independent validation','Synthetic logistic: independent validation')
(H/'optimized_plot.py').write_text(s)
for p in H.glob('*.py'):ast.parse(p.read_text())
print('Verified search/analysis adapted for per-N P3B')
