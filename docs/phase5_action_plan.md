# Phase 5 行动计划与服务器最小验证

依据：[当前代码与真实日志诊断](phase5_diagnostic_analysis.md)。保护Phase 4基线bc4d642b44b408419b489e084cb12d8c7a48c8e9；工程分支feat/neurofdan-phase5-diagnosis。没有更改模型、目标、优化器、采样数、数据或选模规则。无validation，seed42，DEAP正类1/HCI正类0，测试选模偏差仍需报告。

## A. 日志与结果整理：本轮实施

原要求同时提到“必要打印两组矩阵”和“不要每轮打印完整矩阵”。采用每轮一行中的紧凑CM=[[a,b],[c,d]]满足前者，不再输出原多行长块；矩阵行真实、列预测，始终[0,1]。一行包含Loss、训练/测试ACC/F1、两组CM、测试预测计数、单类连续轮数、best刷新及完整性。

每10轮输出简短事实/前门对照、CE/KL、概率变化与full的关键中介/梯度量级；不输出全梯度字典。--log_detail_interval默认10，0只关闭周期终端摘要，不关闭每轮保存。单类/失衡/精确全零梯度的首次出现或类别切换立即提醒，持续情况在每轮计数及周期摘要体现；NaN/Inf仍立即报错。没有擅自以小梯度阈值停止训练。

basic仍每轮计算完整优化训练子集和测试集的分类/概率统计；full仍逐轮额外计算CNS/ANS分布、采样、类别及裁剪前梯度。两者控制计算/保存，不通过full倾倒终端字典。四类JSONL/NPZ能力保留，TensorBoard继续保存原标量。

新增run_reporting.py只读产物，正常结束生成run_summary.json和run_report.txt；run_loso异常退出时也尝试写部分摘要，不掩盖原异常。它不加载权重、不重新选best，不是完整LOSO验收。新fold.json记录本次启动源码身份，新epoch记录各自source_version；续训可能跨代码版本，不能把最近启动SHA当作全部历史的SHA。

## B. 服务器结果自动上传：本轮实施

实现scripts/upload_run_diagnostics.sh及同名Python工具。独立临时Git工作区向server-diagnostics/runs/<北京时间_数据集_任务_fold_唯一ID>追加，不切换实验工作区、不强推、不覆盖历史，不使用git add .。只上传轻量白名单，不上传模型、信号或虚拟环境；缺失、超限、被拒文件写入manifest。

旧Phase 4结果无需重新训练或修改文件即可上传；缺少摘要时在临时快照生成upload_summary.json/upload_report.txt。工具代码SHA、实验checkout在上传时的SHA、产物记录的启动SHA分开保留。自动脱敏有局限，首次可用dry-run检查。详见[上传指南](phase5_server_upload_guide.md)。本轮不运行该工具，不从云端创建或上传实验快照。

## C. 模型训练问题修复：本轮不实施

目前没有独立证据证明计算图断开、S/K公式错误或标签实现错误。若服务器探针证实例如同一不同输入进入骨干后全部相同、评估图全去自环、错误缓存、明确梯度断路，先记录输入/输出、参数、函数及最小重现，再提出具体修复。不能以“分数更高”代替bug证据。

已有梯度统计只读总loss、裁剪前，未分离CE/KL/发放率，也未记录裁剪后参数变化；这是一项测量边界，不是证明裁剪造成退化。本轮不更改训练反向或新增正则。

## D. 模型结构或目标调整：本轮不实施

若需改变beta、weight_decay、BN、骨干、参考维数或训练目标，需独立说明理论依据、参数/函数位置和潜在代价，并先明确实验方案。不能默认观察标签是真实do标签，不能删除CNS→ANS、添加Mc→Y或改成普通融合。原论文对潜在中介识别的限制仍成立。

## 最小验证顺序：先已有产物，再固定checkpoint，不先重跑LOSO

| 步骤 | 服务器操作及固定条件 | 支持或排除的解释 | 需要回传 |
|---|---|---|---|
| E0 | 上传本次Phase 4 run的JSONL/NPZ、fold.json、来源索引和日志；先确认任务、参数及实际源码 | 直接检查全轮预测、早期best、完整性；不再从ACC反推矩阵 | 白名单快照和manifest链接 |
| E1 | 同一checkpoint，选16个已处理窗口，eval且不step；分别看原输入/h_c/h_a跨样本std、有效图边数、mu/logvar变化 | 输入变而h不变→骨干/图；h变而参数不变→posterior参数化或权重；只有eval退化→模式差异候选 | 仅数值摘要，不传信号或特征张量 |
| E2 | 同批同权重，用autograd.grad分别读取事实CE、加权KL、加权fire的模块梯度，retain_graph，不写p.grad、不step；记录参数范数、Adam内部L2量级、裁剪前/后总范数 | 区分任务信号弱、正则主导、缺失梯度、裁剪缩放；不是根据总梯度认定CE无梯度 | 每项每模块范数及是否None，另列当前权重配置 |
| E3 | 固定ANS输入及MC噪声，改变m_c数值，比较附加电流与原电流RMS、mu_a/logvar_a差异；固定m_c再改变x_a | 区分CNS调节失效、ANS自身输入失效及尺度问题 | 电流比、参数变化范数，不上传完整张量 |
| E4 | 同一组m_a样本、同一参考库，先对当前g(x)沿S平均，再以相同S对独立参考沿K/S平均；参考库固定，不比较两次独立forward的随机结果 | 将参考边缘化与“单次事实采样 vs 多次中介采样”分开；可看OutcomeHead对m_a/g的依赖 | 两条概率均值/std、逐类矩阵、L1、固定样本/参考来源数量 |
| E5 | 固定checkpoint/库/干预点，配对公共随机数比较K=1/16/64及S=1/4/16；不改变训练 | 若仅减少MC噪声而常数不变，积分噪声不是输入信息丢失的修复 | 各配置概率误差/变化，不宣称更高ACC即识别正确 |

E1–E5是推荐方案，未由Codex执行，也没有被本轮自动加入训练。只有E0可以立即用现有结果完成。以下给出E1的最小服务器命令；一次只读16窗口和已有CPU checkpoint，不训练，不构建完整参考库：

```bash
cd /home/fss/NeuroFDAN
python - <<'PY'
import json
from pathlib import Path
import torch
from Datasets import PhysiologicalWindows
from NeuroFDAN import NeuroFDAN
from frontdoor_scm import FrontDoorConfig
folder=Path('outputs/phase4/DEAP/arousal/seed42/deap_arousal_fold01_full/fold01')
cfg=json.loads((folder/'fold.json').read_text())
args=cfg['arguments']
state=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
torch.set_num_threads(1)
model=NeuroFDAN(ecg_channels=2 if args['dataset']=='DEAP' else 3,
    frontdoor_config=FrontDoorConfig(**state['frontdoor_config']))
model.load_state_dict(state['model']); model.eval()
data=PhysiologicalWindows('/home/fss',args['dataset'],[cfg['held_out_subject']],args['label'])
# 均匀选16个窗口，仅做输入依赖检查，不作为完整测试成绩。
indices=torch.linspace(0,len(data)-1,16).long()
def spread(t):
    return t.reshape(len(t),-1).double().std(0,unbiased=False).mean().item()
with torch.no_grad(),torch.random.fork_rng(devices=[]):
    torch.manual_seed(42)
    xc,xa=data.x_c[indices],data.x_a[indices]
    hc,ha=model.backbone(xc,xa)
    out=model.frontdoor(hc,ha,fd_mode='none',S=16)
    graph=(model.backbone.dag_adj+model.backbone.eval_gumbel).argmax(-1)
    result=dict(checkpoint_epoch=state['epoch'],dataset=args['dataset'],task=args['label'],
        samples=len(indices),input_std_c=spread(xc),input_std_a=spread(xa),
        feature_std_c=spread(hc),feature_std_a=spread(ha),
        graph_nonself_rows=int((graph!=torch.arange(32)).sum()),
        mu_c_std=spread(out['mu_c']),logvar_c_std=spread(out['logvar_c']),
        mu_a_mean_std=spread(out['mu_a'].mean(1)),
        conditional_mu_a_std=out['mu_a'].double().std(1,unbiased=False).mean().item(),
        factual_probability_std=out['probs_obs'].double().std(0,unbiased=False).tolist(),
        note='16窗口只读诊断；none仅为获取事实路径，不是重训消融或物理干预')
destination=Path('outputs/phase5_probes'); destination.mkdir(parents=True,exist_ok=True)
(destination/'probe_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
print(json.dumps(result,ensure_ascii=False,indent=2))
PY
```

将folder改为实际run和任务位置；脚本读取fold.json的任务，不预设粘贴日志是哪项任务。只读取自己训练的可信checkpoint；没有last.pt时先上传E0，不补造权重。把outputs/phase5_probes作为run目录传给上传工具即可上传probe_summary.json（metadata可能未知，文件内明确记录任务）。Codex没有运行上述命令。

E2–E5需要结合E0/E1结果确定具体checkpoint、窗口与参考来源后再进行；不建议先训练七套模型。尤其fd_mode=none在训练时跳过参考采样，会改变后续RNG消耗，不能把两次独立重训的差异全部归于边缘化。no_cns2ans是估计器结构消融，不是真实生理干预。

## 本轮检查与未完成事项

按本轮要求不运行单元测试、import、CLI、上传演练、模型前向/反向或实验；历史Phase 4的71项结果不能写成本轮通过。仅进行文件审阅、静态语法解析及Git差异检查。旧测试中多行日志断言随新终端契约更新，但未执行。

已实际静态核验：6个修改/新增Python文件可由ast解析；上传Shell包装器通过bash -n（仅语法，不执行）；git diff --check无空白错误。train_one_epoch、evaluate、evaluate_training_set、BestCheckpoint、build_reference_bank、cache_reference_features、save_checkpoint、load_checkpoint、seed_everything的AST与Phase 4逐项一致。NeuroFDAN.py、frontdoor_scm.py、Datasets.py、loss.py、CalculateResult.py、result_audit.py、requirements.txt无差异。这些不等于运行测试通过。

尚未确定：真实训练的精确label/启动参数、早期best及完整100轮轨迹、实际权重/激活敏感度、CE与正则独立梯度、数据支持/参考分布偏差、主要根因。A/B工程工具也待服务器dry-run与首次真实上传验证；不能据此声称ACC或模型已改善。
