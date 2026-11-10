# Phase 4：服务器训练塌缩诊断指南

仅在自己的真实数据服务器执行以下命令。工作目录均为 /home/fss/NeuroFDAN；沿用 Phase 3 的数据、Python 环境、GPU 配置及训练协议。Codex没有执行真实数据或GPU实验。分析依据见 [诊断报告](phase4_diagnostic_analysis.md)。

## 1. 保护正在运行的 Phase 3 与已有结果

先记录分支、SHA、运行进程；不要在现有训练尚未结束时切换代码。等待它正常结束，或按自己的实验计划中断并保留 last.pt、history 与终端日志。不要删除输出、reset、覆盖 checkpoint 或把 Phase 3 产物移动成 Phase 4 成绩。

```bash
cd /home/fss/NeuroFDAN
git status --short
git branch --show-current
git rev-parse HEAD
pgrep -af mainNeuroFDAN.py
ls -lah outputs/phase3
```

Phase 3 稳定提交为 cd844c85d28f2335acf4d41083184d8ad98e5bbd。工作区干净、训练进程已结束后再切换：

```bash
cd /home/fss/NeuroFDAN
git fetch origin
if git show-ref --verify --quiet refs/heads/feat/neurofdan-phase4; then
    git switch feat/neurofdan-phase4
else
    git switch --track -c feat/neurofdan-phase4 origin/feat/neurofdan-phase4
fi
git pull --ff-only origin feat/neurofdan-phase4
git rev-parse HEAD
git ls-remote origin refs/heads/feat/neurofdan-phase4
```

两处SHA应一致。已有本地修改时先保留，不用强制重置。Phase 4不会写入下文命令之外的Phase 3 run目录，也不会自动读取或接着训练Phase 3权重。

## 2. 复用原环境与数据

激活Phase 3实际使用的环境。下面仅为 .venv 的示例，若使用conda或其他venv，改为其原激活方式；不要为了本轮重装PyTorch/CUDA。

```bash
cd /home/fss/NeuroFDAN
source .venv/bin/activate
python --version
python -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python mainNeuroFDAN.py --help
nvidia-smi
ls -lah /home/fss/Dataset/DEAP-python
mkdir -p logs/phase4
```

没有新增第三方依赖，沿用 requirements.txt。数据格式、标签列、阈值、平衡、归一化和窗口处理未变，不需要重新转换数据。仍需DEAP 1.npy至32.npy的既有pickle字典。

## 3. 先直接核验 Phase 3 已有测试矩阵

Phase 3 的 history.jsonl 已有每轮测试 confusion_matrix。下面只读取文件，不重跑模型；将run名改为服务器实际值即可。该例路径是Phase 3指南中的正式arousal任务，不假定你的命名相同。

```bash
cd /home/fss/NeuroFDAN
python - <<'PY'
import json
from pathlib import Path
import numpy as np
path = Path('outputs/phase3/DEAP/arousal/seed42/deap_arousal_seed42/fold01/history.jsonl')
for line in path.read_text().splitlines()[-10:]:
    row = json.loads(line)
    cm = np.asarray(row['confusion_matrix'])
    print('epoch', row['epoch'], '完整测试', row['test']['complete_test'])
    print('真实[0,1]', cm.sum(1).tolist(), '预测[0,1]', cm.sum(0).tolist())
    print(cm)
    print('单类别预测', bool((cm.sum(0) == 0).any()))
PY
```

Valence使用相应valence run的history，不用arousal数据推断valence。真实计数来自矩阵行和，预测计数来自列和；全预测1必须第一列为零，全预测0必须第二列为零。若history未保存或文件路径不符，保留原始日志反馈，不能用ACC/F1补造矩阵。best_predictions.json只对应最佳轮，不能代表全部epoch。

## 4. DEAP第1折基本诊断

默认basic每轮都完整评估实际优化训练子集及完整测试集，打印两组主路径矩阵、Precision/Recall/Specificity/F1、真实/预测计数、概率均值/标准差、熵及factual对照。额外评估不更新模型，不参与选模。

以下是100轮、第1折、独立目录的新运行，未改原有科学参数，不设置有限batch上限：

```bash
cd /home/fss/NeuroFDAN
bash scripts/phase3_run_logged.sh logs/phase4/deap_arousal_fold01_basic.log \
  python mainNeuroFDAN.py \
  --dataset DEAP --label arousal --root_dir /home/fss \
  --output_dir outputs/phase4 --run_name deap_arousal_fold01_basic \
  --seed 42 --device cuda:0 --held_out_subjects 1 \
  --epochs 100 --batch_size 40 --workers 0 \
  --diagnostic_level basic
```

如果只需要短程接口检查，可以另用全新run_name并显式加 --epochs 1 --max_train_batches 1 --max_test_batches 1；此时训练/测试评估均仅有限batch，JSON与日志标记不完整，不能作为完整矩阵或正式结果。没有GPU时显式 --device cpu；不承诺默认正式采样在CPU上的运行时长。

完整训练集指原平衡选择后实际用于优化的窗口子集，评估不丢弃最后不足batch的窗口。它不同于未平衡的参考总体，数量写入epoch_metrics.jsonl；参考库仍从未平衡总体抽取，不含held-out。

## 5. 开启full，诊断两项任务

full在同样评估基础上增加中介与梯度诊断，默认latent、S/K/N_ref、优化器、损失权重及warmup均不变。优先用full回答当前塌缩问题：

```bash
cd /home/fss/NeuroFDAN
bash scripts/phase3_run_logged.sh logs/phase4/deap_arousal_fold01_full.log \
  python mainNeuroFDAN.py \
  --dataset DEAP --label arousal --root_dir /home/fss \
  --output_dir outputs/phase4 --run_name deap_arousal_fold01_full \
  --seed 42 --device cuda:0 --held_out_subjects 1 \
  --epochs 100 --batch_size 40 --workers 0 \
  --diagnostic_level full
```

```bash
cd /home/fss/NeuroFDAN
bash scripts/phase3_run_logged.sh logs/phase4/deap_valence_fold01_full.log \
  python mainNeuroFDAN.py \
  --dataset DEAP --label valence --root_dir /home/fss \
  --output_dir outputs/phase4 --run_name deap_valence_fold01_full \
  --seed 42 --device cuda:0 --held_out_subjects 1 \
  --epochs 100 --batch_size 40 --workers 0 \
  --diagnostic_level full
```

逐项运行，不同时向同一run/fold写入。若任务已存在，用新run_name或同run的 --resume，不删除已有产物；basic/full不是新的模型或消融。

## 6. 终端与文件检查

```bash
cd /home/fss/NeuroFDAN
tail -f logs/phase4/deap_arousal_fold01_full.log
```

日志打印顺序为优化Loss及CE/KL/发放率损失、[训练集]和[测试集]多行矩阵、full中介与梯度摘要、Checkpoint决定。内部为保持原测试随机路径，先算测试再算额外训练集评估，打印仍按训练/测试排列。矩阵始终行真实、列预测、顺序[0,1]；DEAP正类1，HCI正类0，HCI矩阵不能直接照搬DEAP的TN/FP/FN/TP位置含义。

全预测一类会有中文WARNING及single-class prediction detected标记，同时记录连续轮数；非零但最少预测类比例<1%时报告极度失衡，1%仅是诊断阈值。概率行合法也可能全为同一类别，不代表已经学会分类。

```bash
cd /home/fss/NeuroFDAN
cat logs/phase4/deap_arousal_fold01_full.log.exit
tail -n 2 outputs/phase4/DEAP/arousal/seed42/deap_arousal_fold01_full/fold01/epoch_metrics.jsonl
```

原日志包装脚本保留pipefail和Python/tee两个退出码。缺少.exit时先看进程，不把未完成写入当成成功。

每fold新增：

| 文件 | 内容 |
|---|---|
| epoch_metrics.jsonl | 每epoch一条，完整训练/测试指标与两组矩阵、优化过程统计、中介摘要及详细文件引用、完整性及smoke标记。 |
| epoch_confusion_matrices.npz | epochs、train、test、complete数组；train/test形状[E,2,2]，complete为每轮两集是否都完整。 |
| mediator_diagnostics.jsonl | full的两集CNS/ANS中介完整统计；basic的train/test为null，表示未采集，不填零占位。 |
| collapse_diagnostics.jsonl | 主路径/factual计数、概率、熵、两路径差异、跨轮平均概率变化、单类连续轮数和full零梯度连续轮数。 |

既有history.jsonl的train仍是优化过程统计，test仍是原测试选模路径；新epoch_metrics.train才是epoch末完整训练集评估。原best/last、预测与来源文件继续保存，不用训练集表现选best。summary与history保持一致。诊断文件按epoch更新，续训回退时去除回退点之后的旧诊断，不补造缺失轮次；多文件保存不构成单一事务。

读取矩阵及指标：

```bash
cd /home/fss/NeuroFDAN
python - <<'PY'
import json
from pathlib import Path
import numpy as np
folder = Path('outputs/phase4/DEAP/arousal/seed42/deap_arousal_fold01_full/fold01')
row = json.loads((folder/'epoch_metrics.jsonl').read_text().splitlines()[-1])
print('epoch', row['epoch'], '完整', row['complete'], '主路径', row['prediction_path'])
for stage in ('train','test'):
    m = row[stage]
    print(stage, {k:m[k] for k in ('acc','precision','recall','specificity','selection_f1')})
    print(m['prediction_diagnostics']['primary'])
print('优化统计', row['optimization'])
with np.load(folder/'epoch_confusion_matrices.npz', allow_pickle=False) as data:
    print('epochs',data['epochs'].tolist(),'完整标记',data['complete'].tolist())
    print('最后训练矩阵\n',data['train'][-1])
    print('最后测试矩阵\n',data['test'][-1])
PY
```

读取full中介与梯度：

```bash
cd /home/fss/NeuroFDAN
python - <<'PY'
import json
from pathlib import Path
folder = Path('outputs/phase4/DEAP/arousal/seed42/deap_arousal_fold01_full/fold01')
row = json.loads((folder/'mediator_diagnostics.jsonl').read_text().splitlines()[-1])
for stage in ('train','test'):
    if row[stage] is None:
        print(stage,'basic未采集中介统计'); continue
    for branch in ('cns','ans'):
        m=row[stage][branch]
        print(stage,branch,'均值跨样本std',m['posterior_mean_between_sample_std'])
        print('posterior std',m['posterior_std'],'variance',m['posterior_variance'])
        print('采样内变化',m['sampled_within_sample_std'],'类均值距离',m['class_mean_distance_l2'])
metrics=json.loads((folder/'epoch_metrics.jsonl').read_text().splitlines()[-1])
print('裁剪前梯度',metrics['optimization'].get('gradients','basic未采集'))
PY
```

中介统计沿完整流合并Welford矩，ddof=0。CNS mu为[N,D]；ANS先对每样本S个条件均值求平均，再计算std_N，另记录std_S。ANS该值有有限MC噪声。posterior std/variance的均值/极值精确；中位数采用4096个配置范围内等宽箱的近似，保存median_bin_width，常量分布精确。小型潜变量维度摘要的中位数为精确值。没有缓存全部[N,S,D]或[B,S,K,C]诊断副本；模型原有积分内部张量不在本轮改造。

缺少某一真实类别时类均值距离为null，类内方差为零时Fisher比为null，不产生NaN/Inf或虚构无限可分性。S=1无法检验同一样本不同MC的变化。低均值变化与较大posterior方差可同时存在；大量随机采样变化也不代表编码了输入信息。

variable_dimension_fraction_at_1e_3仅描述跨样本标准差达到历史阈值的维度比例；between_mean_to_posterior_std_rms_ratio比较均值跨样本变化与posterior噪声尺度。二者都不是经过标定的可识别性或塌缩检验。

梯度统计在backward之后、原max_norm=1裁剪和step之前，记录范数、精确零batch数、缺失梯度数；路径消融或Identity投影无参数时active=false，不作“训练失败”警告。整模块整轮梯度为零时警告并记连续轮数。出现非有限梯度时full在step前报错停止，不把NaN写成正常统计。没有复制全模型逐step计算参数更新范数，以控制开销；CPU测试核验实际参数可更新。

## 7. 如何判断下一步

- 主训练矩阵也单类且factual也单类：先看事实CE、类别计数、梯度及输入是否合理。
- factual学会分类而do仍单类：比较两路径矩阵、不一致率与概率L1，考虑事实参考线索和边缘化目标差异；不能把两者分类指标混用。
- 均值跨样本变化低但posterior std正常：说明均值信息弱，不等于采样值常量或方差塌缩。
- 采样变化大但类均值/样本均值无差异：可能主要为噪声；需要进一步分析，不声称有真实中介信息。
- 模块零梯度：查看active、missing_grad_tensors_mean、cns2ans消融与多轮状态，区别缺失梯度与接近零梯度。

以上是排查顺序，不是自动根因结论。请反馈两项任务同一段epoch的矩阵、事实/主路径对照、中介统计、梯度、SHA及日志，勿仅反馈ACC/F1。不要为了分数变化调整阈值、loss或移除KL。

## 8. 恢复正式LOSO与原Phase 3运行

诊断结果保留在独立目录。若继续Phase 4代码跑正式完整LOSO，另用全新run_name，默认basic仍逐轮完整训练/测试评估：

```bash
cd /home/fss/NeuroFDAN
bash scripts/phase3_run_logged.sh logs/phase4/deap_arousal_loso_basic.log \
  python mainNeuroFDAN.py \
  --dataset DEAP --label arousal --root_dir /home/fss \
  --output_dir outputs/phase4 --run_name deap_arousal_loso_basic \
  --seed 42 --device cuda:0 --start_subject 1 --end_subject 32 \
  --epochs 100 --batch_size 40 --workers 0 \
  --diagnostic_level basic
```

若需回到原Phase 3继续其既有run，等Phase 4任务结束、工作区干净后切回稳定分支：

```bash
cd /home/fss/NeuroFDAN
git switch feat/neurofdan-phase3
git rev-parse HEAD
```

然后严格按原任务的output_dir、run_name和原配置，使用 [Phase 3续训指南](phase3_server_guide.md) 中单fold --resume，再从下一个未运行subject继续。不要向Phase 3代码传 --diagnostic_level，也不要把Phase 3 checkpoint直接指向Phase 4新run（既有续训接口会拒绝不同父目录）。没有承诺逐位等同未中断训练。

Phase 4独立run的续训也只支持同run单fold --resume，诊断级别可换，但原科学参数应相同，100轮目标不变。正式统计仍用CalculateResult.py --run_dir和--require_complete，正类DEAP=1/HCI=0；单fold诊断不能描述成完整LOSO，smoke不能混入正式表。

## 9. 科学限制与开销

没有validation，固定seed42，test-selected checkpoint规则保持不变，因此测试选模偏差仍存在。真实中介不可观测、没有oracle监督；描述性中介/类别分离或梯度非零不证明真实生理机制或因果识别。no_cns2ans仍是估计器消融。

额外完整训练集评估有实际推理成本，full还增加CPU统计与梯度读取。每epoch共享同一权重的参考缓存，避免为两集重复编码；额外评估在eval/no_grad下批量执行，独立加载器及随机流避免影响训练。没有云端真实显存/耗时测试，不给出未经实测的性能承诺。
