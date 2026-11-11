# Phase6：固定自然训练协议与多数类退化后的下一步

## 1. 用户已确定的训练协议

后续正式CLI保留全部自然训练窗口，不做类别下采样、过采样、重采样或类别平衡；不因本次成绩差而撤回。窗口shuffle只是无放回排列，不改变入选窗口集合。中介S次采样和参考K次Monte Carlo仍是前门积分的一部分，不能因“不采样”而删除。

与对比方法统一自然训练总体消除一项设置差异，但完整公平性还需核实预处理、划分、选模和统计口径。本项目仍保持原train/test二分和test-selected checkpoint规则，并披露选择偏差。

代码落地：mainNeuroFDAN.py的--no_balance默认True，旧显式参数继续兼容；ModelTrain.run_fold在读取数据前拒绝no_balance=False，优化索引直接为全部arange。Datasets.balanced_indices保留作旧版本资料和历史测试，不再被当前训练入口调用；旧平衡实验用受保护旧版本回放。模型结构、损失、阈值、学习率、BatchNorm冻结等均不变。

本次推荐的DEAP400、HCI100恰好整除自然训练窗口，没有尾窗被丢弃。其他batch仍沿用原drop_last规则并在fold.json明确报告，不把它隐瞒为全覆盖；本次不改尾批或BN策略。

## 2. 新真实数据结果已核验

- DEAP：f325c6b61f749184313aa198a5cf846525f8363c，runs/2026-11-11_220240_DEAP_arousal_fold01_e096018e。
- HCI：fa2deee20fde307068e691035e539136402f6a58，runs/2026-11-11_220447_HCI_arousal_fold01_4eabfdb8。

逐文件长度和SHA256一致，无跳过项。两组均是源码446b0b591a216a5718baaca69a76f9af63bcb08d的完整非smoke20轮，实际优化窗口等于训练总体12400/2300。每轮训练/测试评估完整；参考来源JSON与旧基线逐字节相同，仍为2048个非held-out训练窗口，来源检查通过。

缺少项是通用白名单的config/probe及fold级重复摘要，必要fold.json、history、epoch_metrics、summary和reference_sources均存在；不是运行失败。上传程序退出0只证明上传，以上完成性来自原始训练产物和日志。

## 3. 结果并未解决退化

矩阵行真值、列预测，顺序固定[0,1]。以下是第20轮；两条路径在全部20轮均为同一个单类判决。

| 数据集/集合 | probs_do与probs_obs的矩阵 | ACC | Balanced Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| DEAP训练 | [[0,5270],[0,7130]] | 0.575000 | 0.500000 | 0.365079 |
| DEAP测试 | [[0,160],[0,240]] | 0.600000 | 0.500000 | 0.375000 |
| HCI训练 | [[1450,0],[850,0]] | 0.630435 | 0.500000 | 0.386667 |
| HCI测试 | [[20,0],[80,0]] | 0.200000 | 0.500000 | 0.166667 |

DEAP训练事实CE=0.681549，训练类别先验熵=0.681855；HCI对应0.655220与0.658724。说明事实监督当前主要停留在先验附近，不能把DEAP60%解释为有效分类改善。概率仍有跨窗口变化，不能断言所有中介/输入信息为零，也不通过改阈值或标签改变这些结论。

DEAP第20轮测试CNS posterior均值跨样本std约0.000613，平均posterior std约0.9944；HCI约0.020604和0.9764。梯度存在，但不代表监督已学出有效条件关系。两条路径都单类，不能把失败完全归因于前门平均。

DEAP最佳为epoch1、ACC0.6/F1正类1为0.75，仍是全预测1的先验基线。HCI完整结束但从未ACC>0.5，has_eligible_best=False，只有last.pt；这不是未完成，也不能伪造best.pt或悄悄排除该fold。

此刻不重跑相同20轮、不追加100轮、不扩展LOSO、不重新引入平衡/权重，不切换正式主输出。先补齐基础拟合能力证据。

## 4. 补齐曾经未执行的固定小批检查

旧固定小批记录a6e5bb2e5f77dfbe400af632e8a55ea0e5c0e339仅在torch导入时失败，详见capacity_import_failure.md；从未得到记忆能力结论。后续真实训练已成功，无需再重复环境诊断。现在用新的自然训练last.pt补齐这一项，不新加合成真值监督或教师模型。

沿用scripts/check_fixed_batch_capacity.py：两个私有临时模型分别从seed42初始化、自然训练epoch20权重开始；各最多300个CPU小批更新。使用原事实CE+KL、固定loss_epoch20、lr1e-4/wd0/fire0，重置私有Adam moment，不将其称为原训练续训或完整warmup。

修正旧探针按类别各取一半的方式：改为训练被试2的来源索引等间距16个窗口，不读取标签做选择，也不自动重选；选择后若只有一类就停止。这是限时诊断窗口，不是正式训练集的新抽样规则，不能把结果当泛化ACC。返回真实类别计数和来源；正式训练总体始终全部窗口。

仅事实路径参与本次原监督目标；fd_mode=none用于省去不参与损失的do计算，正式probs_do和SCM完全不变。原Mc→Ma、OutcomeHead、KL/发放率项和BN冻结均保留。临时模型实际backward/step，但不保存权重；原last.pt只读，结束复核其SHA256。

每50步进行train/eval两模式只读检查，各4个固定MC流（不是4个训练种子），分别记录CE/ACC、矩阵、梯度和更新。原预声明记忆阈值仍为两模式平均CE≤0.1、四次单次ACC均≥0.95；它只作小批诊断，不是因果识别或正式选模门槛。未达阈值需结合曲线判断，不自动证明网络不可能训练；成功也可能依赖自身参考分支，不证明中介识别。

本次自然默认值与来源选择新增6项回归测试；云端未运行。已有模型测试和本次服务器两组完整训练是不同证据，不混为一谈。

## 5. 服务器：获取新工具，仅运行CPU限时诊断

继续使用dslenv、保留所有旧目录。不运行mainNeuroFDAN.py，不再训练新的20轮。新分支同时保存自然训练默认值和修改后的探针，旧工具及checkpoint不覆盖。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
git status --short
git worktree list
if [ -n "${GIT_SSH_COMMAND:-}" ]; then echo '已有自定义SSH命令，停止，不覆盖认证设置'; exit 1; fi
export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
timeout 600 git fetch origin fix/neurofdan-natural-training-and-capacity
NATURAL_SHA="$(git rev-parse FETCH_HEAD)"
NATURAL_TOOLS=/home/fss/NeuroFDAN-natural-capacity-tools
if [ ! -e "$NATURAL_TOOLS" ]; then
    git worktree add --detach "$NATURAL_TOOLS" "$NATURAL_SHA"
fi
test "$(git -C "$NATURAL_TOOLS" rev-parse HEAD)" = "$NATURAL_SHA"
test -z "$(git -C "$NATURAL_TOOLS" status --porcelain)"
git -C "$NATURAL_TOOLS" log -1 --oneline
NATURAL_FOLD=/home/fss/NeuroFDAN/outputs/phase6/population_alignment/DEAP/arousal/seed42/deap_arousal_fold01_nobalance20_001/fold01
test -f "$NATURAL_FOLD/last.pt"
NATURAL_RESULT=/home/fss/NeuroFDAN/outputs/phase6/capacity/deap_arousal_fold01_natural_capacity_001
test ! -e "$NATURAL_RESULT"
mkdir -p "$NATURAL_RESULT"
(
cd "$NATURAL_TOOLS"
bash scripts/phase3_run_logged.sh "$NATURAL_RESULT/policy_tests.log" \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m unittest discover -s tests -p test_natural_training_policy.py -v
)
bash "$NATURAL_TOOLS/scripts/phase3_run_logged.sh" "$NATURAL_RESULT/train.log" \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -u "$NATURAL_TOOLS/scripts/check_fixed_batch_capacity.py" \
  --root_dir /home/fss --fold_dir "$NATURAL_FOLD" --output_dir "$NATURAL_RESULT" \
  --probe_subject 2 --samples 16 --steps 300 --check_every 50 --seed 42 --device cpu
cat "$NATURAL_RESULT/policy_tests.log.exit"
cat "$NATURAL_RESULT/train.log.exit"
)
```

本脚本实际限定DEAP/arousal/fold1/epoch20，从checkpoint和fold.json严格核验；不虚构dataset、label、epochs等不存在的参数。诊断batch16不改变正式DEAP400/HCI100设置，两个私有模型总计最多600步，不自动延长。需要服务器本地last.pt，GitHub快照不含权重。失败也保留日志；无需重跑原真实训练。

## 6. 上传预览

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
UPLOAD_TOOLS=/home/fss/NeuroFDAN-upload-timeout-tools
NATURAL_RESULT=/home/fss/NeuroFDAN/outputs/phase6/capacity/deap_arousal_fold01_natural_capacity_001
test "$(git -C "$UPLOAD_TOOLS" rev-parse HEAD)" = 6681381298b74e9b69dc0df0594d5d9c7274acbc
test -z "$(git -C "$UPLOAD_TOOLS" status --porcelain)"
if [ -n "${GIT_SSH_COMMAND:-}" ]; then echo '已有自定义SSH命令，停止，不覆盖认证设置'; exit 1; fi
export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" "$NATURAL_RESULT" \
  --log "$NATURAL_RESULT/train.log" --log "$NATURAL_RESULT/policy_tests.log" \
  --git_timeout 600 --dry-run
)
```

## 7. 正式上传与收据

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
UPLOAD_TOOLS=/home/fss/NeuroFDAN-upload-timeout-tools
NATURAL_RESULT=/home/fss/NeuroFDAN/outputs/phase6/capacity/deap_arousal_fold01_natural_capacity_001
test "$(git -C "$UPLOAD_TOOLS" rev-parse HEAD)" = 6681381298b74e9b69dc0df0594d5d9c7274acbc
test -z "$(git -C "$UPLOAD_TOOLS" status --porcelain)"
if [ -n "${GIT_SSH_COMMAND:-}" ]; then echo '已有自定义SSH命令，停止，不覆盖认证设置'; exit 1; fi
export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
NATURAL_UPLOAD_ID="$(date +%Y%m%d_%H%M%S)_$$"
NATURAL_LOG="/home/fss/NeuroFDAN/outputs/phase6/upload_logs/natural_capacity_${NATURAL_UPLOAD_ID}.log"
NATURAL_BUNDLE="/home/fss/NeuroFDAN/outputs/phase6/upload_bundles/natural_capacity_${NATURAL_UPLOAD_ID}"
mkdir -p outputs/phase6/upload_logs outputs/phase6/upload_bundles
bash "$UPLOAD_TOOLS/scripts/phase3_run_logged.sh" "$NATURAL_LOG" \
  bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" "$NATURAL_RESULT" \
  --log "$NATURAL_RESULT/train.log" --log "$NATURAL_RESULT/policy_tests.log" \
  --git_timeout 600 --bundle_dir "$NATURAL_BUNDLE"
cat "$NATURAL_LOG.exit"
cat "$NATURAL_BUNDLE/upload_state.json"
)
```

回传新的SHA及链接，成功/未达记忆阈值/失败都如实回传。上传退出0不等于阈值通过。probe_summary.json保存两种起点的完整记录、来源、真实计数、原文件未变及不保存权重标志；不上传原权重或原始信号。

## 8. 后续收敛判读

- 两起点都能拟合：基础学习路径有能力，再分析全训练总体的异质性、优化和中介约束；不据此宣称前门识别。
- train能拟合而eval不能：优先检查图/模式与采样差异，不改BN或假定唯一根因。
- 新初始化能拟合而epoch20不能：进一步定位当前参数状态，不能直接怪原Adam moment，因为本探针已重置。
- 两起点都不能形成拟合：结合损失、分组梯度、更新和输入统计定位基本执行路径，不继续扩大真实LOSO或堆辅助损失。

本轮云端仅审查源码、核验Git载荷、计算已保存矩阵的派生指标及静态AST/Shell检查，没有运行CLI、模型、单元测试、真实数据或GPU实验。新6项测试和容量探针尚待服务器执行。独立分支交付，不修改main、不创建PR或合并。自然训练协议是已确定要求，后续不因ACC变动恢复类别平衡。
