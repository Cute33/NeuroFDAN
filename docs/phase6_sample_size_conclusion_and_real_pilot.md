# Phase6：收束合成诊断，回到真实观察总体对照

## 1. 已核验的最终合成学习曲线

- n8192：1577dad1bee355ccdc73428aeb6a062caf51a0f7，runs/2026-11-11_213440_unknown_unknown_run_eef1a71e。
- n32768：8091ab2b94128e09b948ee4ce5132db860910eee，runs/2026-11-11_213705_unknown_unknown_run_11ed8d39。

两份manifest全部载荷长度与SHA256一致，无跳过项。通用run_summary.json缺少不影响本程序结果。两组completed=True，拟合收敛，源文件未变，冻结中介和公共MC流核验通过。源代码为dd1f70b16588029eff712ab737faf36828560cae。7项测试成功；两份快照附带同一测试日志，不计成14项独立测试。

| 条件结果拟合样本量 | gamma0条件测试MSE | gamma1条件测试MSE | gamma0完整前门LL MSE | gamma1完整前门LL MSE |
|---|---:|---:|---:|---:|
| 2048，既有结果 | 0.012028 | 0.011672 | 0.004684 | 0.003863 |
| 8192 | 0.002601 | 0.002192 | 0.000582 | 0.000715 |
| 32768 | 0.000603 | 0.000533 | 0.000470 | 0.000388 |

32768样本时，真实中介+拟合Head的TL MSE为0.000108/0.000143。已学中介+已知Head的LT仍为0.000400/0.000281，与各规模逐次结果完全一致。gamma0参考系数范数从0.39369降至0.15948、0.08161，方向符合该设置真实参考系数为零的约束，但不将范数当显著性检验。

这条固定机制、单个训练数据流、16点、两个gamma的曲线支持有限样本条件估计是重要因素。不能声称单凭此已经解释真实模型全部塌缩，也不能把单种子/嵌套样本当独立重复或报告统计显著。gamma1仍为近似logit模型；MLP的容量、Dropout和优化因素没有被逐一分离。

已知公式积分正确、有限数据条件模型学习困难、无中介标签不可识别是三个层次的问题。合成中介真值监督是诊断参考组，没有教师—学生结构。现在结束该支线：不增加样本量、不重跑20轮合成模型、不将线性Head或真实中介监督移植到正式训练。

真实同trial窗口及同被试记录存在相关性，不能把窗口数直接当独立样本量，更不能将这条合成曲线换算成DEAP/HCI所需窗口数或可达到的ACC。关闭下采样也不意味着自动补足潜在识别信息。

## 2. 真实训练里已确认的总体差异

实际代码路径：

- Datasets.py/PhysiologicalWindows.balanced_indices：依据当前任务标签，trial级下采样并按整组截断。
- ModelTrain.py/run_fold：默认用selected构建WindowSelection进行优化；参考库始终由完整population构建。
- ModelTrain.py/train_one_epoch → frontdoor_scm.py/frontdoor_loss：事实logits的普通CE，加原KL和发放率项；没有选择概率校正，也没有观察标签的do监督。

设原自然训练总体为P，S表示trial是否被优化集保留。在固定表示且保留机制只依赖标签时，若r_y=P(S=1|Y=y)，有

\[
P(Y=y\mid m,x,S=1)=\frac{r_y P(Y=y\mid m,x)}{\sum_z r_z P(Y=z\mid m,x)}.
\]

均匀标签下采样通常使r_0≠r_1，因此事实CE拟合的条件概率目标不同，而参考分布仍为未下采样的经验P(X)。实际选择为无放回的trial级有限样本设计，上式是对应总体选择机制的解释，不把各窗口视为独立Bernoulli选择。潜在中介端到端学习使问题更复杂，公式也不提供潜在识别保证。

这不是标签映射错误的证据，也不能断言它是唯一根因；它是已确认的训练目标与参考总体差异，需要在最终估计目标中明确处理。若研究有意采用平衡目标，应明确相应总体和校准，不能自动称其与自然总体等价。

最小对照使用已经存在的--no_balance：保留全部自然训练窗口，事实CE与固定参考总体在来源上对齐，不增加损失项、不改默认配置或标签定义。不使用中介真值、U或干预真值。它只排查总体目标问题，不解决潜在中介的一般非识别。

## 3. 与既有真实基线严格对应

已重算核验以下旧上传载荷、20轮完整记录和配置：

| 数据集 | 平衡基线提交 | 优化窗口 | 完整训练总体 | batch | 每轮更新 |
|---|---|---:|---:|---:|---:|
| DEAP | f93d7a0cf3d032f92c122cba5fb623bf172f3571 | 10400 | 12400 | 400 | 26 |
| HCI | 8b8f4a69081cae2b254b37c86f037279241714d0 | 1600 | 2300 | 100 | 16 |

两基线源码446b0b591a216a5718baaca69a76f9af63bcb08d。与当前文档分支相比，NeuroFDAN.py、frontdoor_scm.py、Datasets.py、ModelTrain.py、mainNeuroFDAN.py、loss.py逐文件无差异。因此直接复用服务器已有NeuroFDAN-large-batch-tools，不需要获取新运行代码或重跑平衡基线。

保持相同：arousal、held-out1、seed42、20轮、全局窗口shuffle、评估batch40、Adam lr1e-4/weight_decay0、DEAP fire0/HCI fire1、原S/K/N_ref、模型维度、KL/warmup、预处理、test-selected选模规则。weight_decay0是沿用已授权基线设置，不是修改源码默认0.01。

唯一显式实验设置变化为关闭trial标签平衡。它必然同时改变类别比例、保留窗口数和每epoch更新次数：DEAP为31步/epoch、20轮620步（旧520）；HCI为23步/epoch、20轮460步（旧320）。这是epoch匹配的选择策略对照，不能将效果单独归因为类别先验或声称更新预算完全匹配；本轮不再增加步数匹配组。

训练集评估总体也从平衡子集变为完整训练总体，两个训练ACC不能不加说明直接比较。测试被试和窗口不变，可比较其主/事实两路径完整矩阵、ACC、Macro-F1、Balanced Accuracy、概率变化、事实CE、KL及梯度；Macro-F1/Balanced Accuracy可由已保存2×2矩阵计算，无需修改训练器。不要仅凭测试ACC上涨认定因果机制正确。

## 4. 服务器执行：原模型、两个固定单折20轮

继续使用dslenv及原数据/GPU。仅在服务器执行真实训练，云端不运行。本轮不访问合成脚本，不用--resume旧平衡模型。主项目分支不切换；下列工具版本已经在服务器存在且经过基线实验。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
REAL_TOOLS=/home/fss/NeuroFDAN-large-batch-tools
git status --short
git worktree list
test "$(git -C "$REAL_TOOLS" rev-parse HEAD)" = 446b0b591a216a5718baaca69a76f9af63bcb08d
test -z "$(git -C "$REAL_TOOLS" status --porcelain)"
REAL_RUN=/home/fss/NeuroFDAN/outputs/phase6/population_alignment/DEAP/arousal/seed42/deap_arousal_fold01_nobalance20_001
test ! -e "$REAL_RUN"
mkdir -p "$REAL_RUN"
bash "$REAL_TOOLS/scripts/phase3_run_logged.sh" "$REAL_RUN/train.log" \
  python -u "$REAL_TOOLS/mainNeuroFDAN.py" \
  --dataset DEAP --label arousal --root_dir /home/fss \
  --output_dir /home/fss/NeuroFDAN/outputs/phase6/population_alignment \
  --run_name deap_arousal_fold01_nobalance20_001 \
  --seed 42 --device cuda:0 --held_out_subjects 1 --epochs 20 \
  --batch_size 400 --eval_batch_size 40 --workers 0 --sampling_mode window \
  --lr 0.0001 --weight_decay 0 --fire_weight 0 --no_balance --fd_mode full \
  --mediator_dim 64 --hidden_dim 128 --dropout 0.3 --ref_dim 0 \
  --S_train 4 --K_train 16 --S_eval 16 --K_eval 64 --N_ref 2048 --reference_chunk_size 4 \
  --beta 0.001 --beta_warmup_epochs 10 --lambda_obs 1 --time_steps 4 \
  --logvar_min -6 --logvar_max 2 --diagnostic_level full --log_detail_interval 1 \
  --max_train_batches 0 --max_test_batches 0
cat "$REAL_RUN/train.log.exit"
)
```

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
REAL_TOOLS=/home/fss/NeuroFDAN-large-batch-tools
test "$(git -C "$REAL_TOOLS" rev-parse HEAD)" = 446b0b591a216a5718baaca69a76f9af63bcb08d
test -z "$(git -C "$REAL_TOOLS" status --porcelain)"
REAL_RUN=/home/fss/NeuroFDAN/outputs/phase6/population_alignment/HCI/arousal/seed42/hci_arousal_fold01_nobalance20_001
test ! -e "$REAL_RUN"
mkdir -p "$REAL_RUN"
bash "$REAL_TOOLS/scripts/phase3_run_logged.sh" "$REAL_RUN/train.log" \
  python -u "$REAL_TOOLS/mainNeuroFDAN.py" \
  --dataset HCI --label arousal --root_dir /home/fss \
  --output_dir /home/fss/NeuroFDAN/outputs/phase6/population_alignment \
  --run_name hci_arousal_fold01_nobalance20_001 \
  --seed 42 --device cuda:0 --held_out_subjects 1 --epochs 20 \
  --batch_size 100 --eval_batch_size 40 --workers 0 --sampling_mode window \
  --lr 0.0001 --weight_decay 0 --fire_weight 1 --no_balance --fd_mode full \
  --mediator_dim 64 --hidden_dim 128 --dropout 0.3 --ref_dim 0 \
  --S_train 4 --K_train 16 --S_eval 16 --K_eval 64 --N_ref 2048 --reference_chunk_size 4 \
  --beta 0.001 --beta_warmup_epochs 10 --lambda_obs 1 --time_steps 4 \
  --logvar_min -6 --logvar_max 2 --diagnostic_level full --log_detail_interval 1 \
  --max_train_batches 0 --max_test_batches 0
cat "$REAL_RUN/train.log.exit"
)
```

两组串行运行，避免争用同一GPU；可以先运行DEAP再运行HCI。没有新增validation，主输出probs_do、事实输出诊断用途不变。ACC>0.5及ACC/F1选模规则不变；无合格best时只有last.pt，不伪造best。test-selected偏差和被试内无标签归一化限制继续保留。

## 5. 完成后核验参考库和完整记录

```bash
cd /home/fss/NeuroFDAN
python - <<'PY'
import hashlib,json
from pathlib import Path
root=Path('/home/fss/NeuroFDAN/outputs/phase6/population_alignment')
for ds,prefix,n,steps,ref_sha in [
    ('DEAP','deap',12400,31,'5f775ae397291792223b0d38e2c6c8072395b6d4709bfed8cbd7a04be862e9f6'),
    ('HCI','hci',2300,23,'c19916880450654c9459720a40d03839b0043f339e97f745ea503a4048753564')]:
    fold=root/ds/'arousal/seed42'/f'{prefix}_arousal_fold01_nobalance20_001/fold01'
    c=json.loads((fold/'fold.json').read_text())
    assert c['arguments']['no_balance'] and c['optimization_windows']==c['population_windows']==n
    assert c['full_epoch_optimizer_steps']==steps and not c['smoke']
    raw=(fold/'reference_sources.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==ref_sha, '参考库与旧基线不一致，请停止解释比较'
    refs=json.loads(raw)
    assert len(refs)==2048 and all(x['subject_id']!=1 for x in refs)
    rows=[json.loads(line) for line in (fold/'epoch_metrics.jsonl').read_text().splitlines() if line.strip()]
    assert [r['epoch'] for r in rows]==list(range(1,21)) and all(r['complete'] for r in rows)
    summary=json.loads((fold/'summary.json').read_text())
    print(ds,'完整20轮、参考库与基线一致、无测试被试；有合格best=',summary['has_eligible_best'])
PY
```

如果任何检查或训练失败，保留该组日志和部分产物并上传，不通过改变标签或阈值修饰结果。没有GPU时可以使用cpu，但此次是原GPU基线的短程对照，不需要为此构造替代云端训练。

## 6. 上传预览

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
UPLOAD_TOOLS=/home/fss/NeuroFDAN-upload-timeout-tools
test "$(git -C "$UPLOAD_TOOLS" rev-parse HEAD)" = 6681381298b74e9b69dc0df0594d5d9c7274acbc
test -z "$(git -C "$UPLOAD_TOOLS" status --porcelain)"
if [ -n "${GIT_SSH_COMMAND:-}" ]; then echo '已有自定义SSH命令，停止，不覆盖认证设置'; exit 1; fi
export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
for REAL_DATASET in DEAP HCI
do
    if [ "$REAL_DATASET" = DEAP ]; then REAL_PREFIX=deap; else REAL_PREFIX=hci; fi
    REAL_RUN="/home/fss/NeuroFDAN/outputs/phase6/population_alignment/$REAL_DATASET/arousal/seed42/${REAL_PREFIX}_arousal_fold01_nobalance20_001"
    if [ ! -d "$REAL_RUN" ]; then echo "尚未运行：$REAL_RUN"; continue; fi
    bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" "$REAL_RUN" \
      --log "$REAL_RUN/train.log" --git_timeout 600 --dry-run
done
)
```

## 7. 正式上传与持久收据

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
UPLOAD_TOOLS=/home/fss/NeuroFDAN-upload-timeout-tools
test "$(git -C "$UPLOAD_TOOLS" rev-parse HEAD)" = 6681381298b74e9b69dc0df0594d5d9c7274acbc
test -z "$(git -C "$UPLOAD_TOOLS" status --porcelain)"
if [ -n "${GIT_SSH_COMMAND:-}" ]; then echo '已有自定义SSH命令，停止，不覆盖认证设置'; exit 1; fi
export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
POP_UPLOAD_ID="$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p outputs/phase6/upload_logs outputs/phase6/upload_bundles
for REAL_DATASET in DEAP HCI
do
    if [ "$REAL_DATASET" = DEAP ]; then REAL_PREFIX=deap; else REAL_PREFIX=hci; fi
    REAL_RUN="/home/fss/NeuroFDAN/outputs/phase6/population_alignment/$REAL_DATASET/arousal/seed42/${REAL_PREFIX}_arousal_fold01_nobalance20_001"
    if [ ! -d "$REAL_RUN" ]; then echo "尚未运行：$REAL_RUN"; continue; fi
    POP_LOG="/home/fss/NeuroFDAN/outputs/phase6/upload_logs/population_${REAL_PREFIX}_${POP_UPLOAD_ID}.log"
    POP_BUNDLE="/home/fss/NeuroFDAN/outputs/phase6/upload_bundles/population_${REAL_PREFIX}_${POP_UPLOAD_ID}"
    bash "$UPLOAD_TOOLS/scripts/phase3_run_logged.sh" "$POP_LOG" \
      bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" "$REAL_RUN" \
      --log "$REAL_RUN/train.log" --git_timeout 600 --bundle_dir "$POP_BUNDLE"
    cat "$POP_LOG.exit"
    cat "$POP_BUNDLE/upload_state.json"
done
)
```

回传两份SHA和链接；若只先运行一组，可先上传该组。新run目录不覆盖原phase4/5、合成结果或平衡基线，不上传原始数据、模型权重和TensorBoard大文件。

## 8. 交付与验收边界

本轮只新增本报告，没有修改任何运行代码，不需要新增运行测试。云端仅核验上传载荷、静态追踪、比较受保护源文件、检查命令Shell与内嵌Python语法；没有运行模型、CLI、单元测试、真实数据或GPU训练。文档分支独立提交推送，不修改main、不创建PR或合并。

现在可以进入原模型的真实数据短程受控试验，尚不是最终LOSO成绩。若自然总体对照没有改善，保留负结果，转入明确的潜在中介约束讨论，而非调高ACC。若有改善，也需区分样本量/更新预算/类别比例作用，并在无真值权限下补足机制证据，冻结方案后再安排正式LOSO。潜在非识别不会因这次总体对齐而消失。

如果最终采用自然总体训练，需要如实记录与其他方法训练采样/类别权重设置的差异，继续核实预处理、划分和选模口径，不能只凭同为LOSO就宣称比较完全公平。
