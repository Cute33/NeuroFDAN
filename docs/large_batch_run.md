# 全局窗口shuffle与大batch：DEAP400、HCI100

## 已核验采样顺序对照

grouped快照e0f35ed62f5712b9c04647beb03a91c85abb0704，runs/2026-11-11_105918_DEAP_arousal_fold01_82c60af8；window快照20e4886e5aa44bd97d8699b0ace37bb1b0bfe871，runs/2026-11-11_105945_DEAP_arousal_fold01_9d576600。两组各13载荷大小/SHA256一致、无Traceback、epoch1–20完整，summary与history末条一致；参考来源和起始RMS相同，CLI仅名称和sampling_mode不同。RMS不代替完整权重哈希。

测试主路径精确单类轮数grouped16/20、window9/20，最长连续分别7、4；最后5轮事实训练ACC均值0.51288、0.52917。window有改善，但尚不稳定。其best为epoch12，ACC0.6025，矩阵[[7,153],[6,234]]，平衡准确率仅0.509375；grouped best是全预测1的0.6。不能把多数类比例附近的best当作研究成功。

window epoch20前门测试矩阵[[160,0],[239,1]]，仍只预测一个正类；事实测试ACC0.43，训练ACC0.52510。不能以“某些epoch出现两类预测”推断已解决中介学习。SCM、潜在中介不可观测、真实干预缺乏oracle监督与测试选模偏差等边界保持不变。

## 已确认的batch组成与shuffle

用户明确选择全局窗口shuffle＋大batch。训练用sampling_mode=window，每epoch以推进的独立seed42生成器重新排列所有优化窗口，无放回；DataLoader通过sampler打乱，因此不再同时设置shuffle=True。grouped也有shuffle，打乱单位是trial的窗口组。测试及epoch末训练集评估保持顺序读取，不参与优化。

DEAP原每人40trial×10窗=400，HCI每人20trial×5窗=100；但全局标签平衡可能使每个被试保留窗口数不同，再全局shuffle后每批混合多个训练被试。batch400/100不等于每批一名被试。本轮不新增按被试BatchSampler、不关闭既有标签平衡或更改参考总体。

## 更新预算与实现边界

现有DEAP优化窗口10400，batch40每epoch260次更新，batch400为26次；20轮分别5200和520次。相同epoch对应相同样本曝光次数，不是相同更新预算。不能据新20轮仍欠拟合就单独断言大batch或SCM失败，也不自动将epoch加至200。HCI的实际窗口数和更新次数由fold.json记录，不假设其等于原始2300训练窗口。

batch大小变化本身会改变梯度噪声、步数及随机流消费，本轮不自动按比例提高lr或改变beta warmup。wd=0、lr1e-4及其他原实验配置保持，正式默认wd仍0.01。原CLI batch默认40保留供旧命令兼容；本次命令显式设置DEAP400/HCI100、sampling_mode=window。

大batch的B×S×K组合会增加单次显存需求。新增可选eval_batch_size，默认None时仍沿用batch_size；本次显式保持评估40，与旧baseline一致，避免评估也随训练batch扩大。测试、epoch末训练集评估和参考特征缓存均按40分批，样本全覆盖，不改概率公式、S/K或选模输出。真实大batch训练显存仍需smoke验证，不能声称已通过。

fold.json新增训练/评估batch、完整epoch更新次数、从头完整run的计划更新次数；smoke实际更新另由上限和记录判断。禁止续训时静默改变训练/评估batch，旧checkpoint评估batch按旧训练batch解释。模型、SCM、loss.py和train_one_epoch/evaluate/BestCheckpoint不变。

## 服务器步骤1：工具与CPU小测试

继续使用dslenv。仅测试子进程切换工具目录以导入新版本；主操作仍从/home/fss/NeuroFDAN执行。

```bash
cd /home/fss/NeuroFDAN
export LARGE_TOOL_DIR=/home/fss/NeuroFDAN-large-batch-tools
(
set -euo pipefail
git status --short
git worktree list
git fetch origin refs/heads/feat/neurofdan-large-batch-run:refs/remotes/origin/feat/neurofdan-large-batch-run
if [ ! -e "$LARGE_TOOL_DIR" ]; then
    git worktree add --detach "$LARGE_TOOL_DIR" origin/feat/neurofdan-large-batch-run
fi
test "$(git -C "$LARGE_TOOL_DIR" rev-parse HEAD)" = "$(git rev-parse origin/feat/neurofdan-large-batch-run)" || {
    echo '版本不符，停止，不覆盖。'; exit 1;
}
test -z "$(git -C "$LARGE_TOOL_DIR" status --porcelain)" || {
    echo '工具有本地修改，停止，不覆盖。'; exit 1;
}
(cd "$LARGE_TOOL_DIR" && python -m unittest discover -s tests -p test_large_batch_interface.py -v)
(cd "$LARGE_TOOL_DIR" && python -m unittest discover -s tests -p test_sampling_order_control.py -v)
)
```

这9项测试仅用小型TensorDataset，不训练模型。失败先回传输出，不继续。它们不能替代GPU smoke。

## 服务器步骤2：两个数据集各一个batch的smoke

以下是完整可执行循环，数据集、batch、fire权重和名称均在循环内明确指定，不需要补参数。先DEAP后HCI，使用独立smoke目录。HCI是否可读仍以服务器实际文件为准，不自动转换或下载数据。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
for LARGE_DATASET in DEAP HCI
do
    if [ "$LARGE_DATASET" = DEAP ]; then
        LARGE_PREFIX=deap; LARGE_BATCH=400; LARGE_FIRE=0
    else
        LARGE_PREFIX=hci; LARGE_BATCH=100; LARGE_FIRE=1
    fi
    LARGE_RUN_NAME="${LARGE_PREFIX}_arousal_fold01_b${LARGE_BATCH}_smoke_001"
    LARGE_RUN_DIR="/home/fss/NeuroFDAN/outputs/phase5_large_batch/$LARGE_DATASET/arousal/seed42/$LARGE_RUN_NAME"
    mkdir -p "$(dirname "$LARGE_RUN_DIR")"
    mkdir "$LARGE_RUN_DIR"
    bash /home/fss/NeuroFDAN-large-batch-tools/scripts/phase3_run_logged.sh "$LARGE_RUN_DIR/train.log" \
      python -u /home/fss/NeuroFDAN-large-batch-tools/mainNeuroFDAN.py \
      --dataset "$LARGE_DATASET" --label arousal --root_dir /home/fss \
      --output_dir /home/fss/NeuroFDAN/outputs/phase5_large_batch --run_name "$LARGE_RUN_NAME" \
      --held_out_subjects 1 --seed 42 --device cuda:0 \
      --epochs 1 --batch_size "$LARGE_BATCH" --eval_batch_size 40 --workers 0 \
      --sampling_mode window --lr 0.0001 --weight_decay 0 \
      --diagnostic_level full --log_detail_interval 1 --fd_mode full \
      --mediator_dim 64 --hidden_dim 128 --dropout 0.3 --ref_dim 0 \
      --S_train 4 --K_train 16 --S_eval 16 --K_eval 64 --N_ref 2048 \
      --reference_chunk_size 4 --beta 0.001 --beta_warmup_epochs 10 \
      --lambda_obs 1 --fire_weight "$LARGE_FIRE" --time_steps 4 --logvar_min=-6 --logvar_max 2 \
      --max_train_batches 1 --max_test_batches 1
    cat "$LARGE_RUN_DIR/train.log.exit"
done
)
```

任一smoke失败即停止循环；若出现OOM不要静默减少S/K、改损失或将其描述为已支持400。上传已有失败日志，再针对内存作最小处理。两个退出码应均0，检查loss非有限问题、参考来源和训练batch大小；不能用smoke矩阵作为正式成绩，不从smoke checkpoint续训。

## 服务器步骤3：独立20轮配置观察

仅在对应smoke通过后运行。下面串行执行DEAP和HCI的arousal折1，从头初始化；不执行完整LOSO，暂不扩展valence。HCI作为新数据接口与配置观察，不与DEAP构成配对因果对照。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
for LARGE_DATASET in DEAP HCI
do
    if [ "$LARGE_DATASET" = DEAP ]; then
        LARGE_PREFIX=deap; LARGE_BATCH=400; LARGE_FIRE=0
    else
        LARGE_PREFIX=hci; LARGE_BATCH=100; LARGE_FIRE=1
    fi
    LARGE_RUN_NAME="${LARGE_PREFIX}_arousal_fold01_b${LARGE_BATCH}_20ep_001"
    LARGE_RUN_DIR="/home/fss/NeuroFDAN/outputs/phase5_large_batch/$LARGE_DATASET/arousal/seed42/$LARGE_RUN_NAME"
    LARGE_SMOKE_DIR="/home/fss/NeuroFDAN/outputs/phase5_large_batch/$LARGE_DATASET/arousal/seed42/${LARGE_PREFIX}_arousal_fold01_b${LARGE_BATCH}_smoke_001"
    python - "$LARGE_SMOKE_DIR" "$LARGE_BATCH" <<'PY'
import json, sys
from pathlib import Path
folder = Path(sys.argv[1])
codes = dict(line.split('=', 1) for line in (folder/'train.log.exit').read_text().splitlines())
assert codes == {'python_exit': '0', 'tee_exit': '0'}, '对应smoke没有成功，停止'
cfg = json.loads((folder/'fold01/fold.json').read_text())
assert cfg['smoke'] and cfg['arguments']['batch_size'] == int(sys.argv[2])
assert cfg['arguments']['sampling_mode'] == 'window' and cfg['evaluation_batch_size'] == 40
PY
    mkdir -p "$(dirname "$LARGE_RUN_DIR")"
    mkdir "$LARGE_RUN_DIR"
    bash /home/fss/NeuroFDAN-large-batch-tools/scripts/phase3_run_logged.sh "$LARGE_RUN_DIR/train.log" \
      python -u /home/fss/NeuroFDAN-large-batch-tools/mainNeuroFDAN.py \
      --dataset "$LARGE_DATASET" --label arousal --root_dir /home/fss \
      --output_dir /home/fss/NeuroFDAN/outputs/phase5_large_batch --run_name "$LARGE_RUN_NAME" \
      --held_out_subjects 1 --seed 42 --device cuda:0 \
      --epochs 20 --batch_size "$LARGE_BATCH" --eval_batch_size 40 --workers 0 \
      --sampling_mode window --lr 0.0001 --weight_decay 0 \
      --diagnostic_level full --log_detail_interval 1 --fd_mode full \
      --mediator_dim 64 --hidden_dim 128 --dropout 0.3 --ref_dim 0 \
      --S_train 4 --K_train 16 --S_eval 16 --K_eval 64 --N_ref 2048 \
      --reference_chunk_size 4 --beta 0.001 --beta_warmup_epochs 10 \
      --lambda_obs 1 --fire_weight "$LARGE_FIRE" --time_steps 4 --logvar_min=-6 --logvar_max 2 \
      --max_train_batches 0 --max_test_batches 0
    cat "$LARGE_RUN_DIR/train.log.exit"
done
)
```

不要重复执行已完成的相同命令；目录已存在会停止。重跑换新编号，断点续训必须同一batch/模式/配置且目标轮次更大，不能用旧40的checkpoint接上400然后称为新对照。

## 服务器步骤4：两类输出的dry-run和上传

共四个独立目录，smoke与20轮分别标识。失败时上传已创建目录即可；以下循环跳过尚未创建的目录并明确打印，不伪造结果。

```bash
cd /home/fss/NeuroFDAN
for LARGE_DATASET in DEAP HCI
do
    if [ "$LARGE_DATASET" = DEAP ]; then LARGE_PREFIX=deap; LARGE_BATCH=400; else LARGE_PREFIX=hci; LARGE_BATCH=100; fi
    for LARGE_STAGE in smoke_001 20ep_001
    do
        LARGE_UPLOAD_DIR="/home/fss/NeuroFDAN/outputs/phase5_large_batch/$LARGE_DATASET/arousal/seed42/${LARGE_PREFIX}_arousal_fold01_b${LARGE_BATCH}_${LARGE_STAGE}"
        if [ ! -d "$LARGE_UPLOAD_DIR" ]; then echo "尚未创建：$LARGE_UPLOAD_DIR"; continue; fi
        bash /home/fss/NeuroFDAN-large-batch-tools/scripts/upload_run_diagnostics.sh \
          "$LARGE_UPLOAD_DIR" --log "$LARGE_UPLOAD_DIR/train.log" --dry-run
    done
done
```

检查manifest后正式上传：

```bash
cd /home/fss/NeuroFDAN
for LARGE_DATASET in DEAP HCI
do
    if [ "$LARGE_DATASET" = DEAP ]; then LARGE_PREFIX=deap; LARGE_BATCH=400; else LARGE_PREFIX=hci; LARGE_BATCH=100; fi
    for LARGE_STAGE in smoke_001 20ep_001
    do
        LARGE_UPLOAD_DIR="/home/fss/NeuroFDAN/outputs/phase5_large_batch/$LARGE_DATASET/arousal/seed42/${LARGE_PREFIX}_arousal_fold01_b${LARGE_BATCH}_${LARGE_STAGE}"
        if [ ! -d "$LARGE_UPLOAD_DIR" ]; then echo "尚未创建：$LARGE_UPLOAD_DIR"; continue; fi
        bash /home/fss/NeuroFDAN-large-batch-tools/scripts/upload_run_diagnostics.sh \
          "$LARGE_UPLOAD_DIR" --log "$LARGE_UPLOAD_DIR/train.log"
    done
done
```

返回各次SHA/链接，不能将未运行的目录称为完成。所有输出位于outputs/phase5_large_batch，原采样对照和历史结果不覆盖，不上传权重。正式研究统计排除smoke；20轮仍是配置观察，不等于完整LOSO。

## 实际执行与后续门槛

本轮只有GitHub结果核验、静态源码审查和提交，没有云端模型、CLI、单元测试、GPU或训练。新增测试与大batch显存/数据读取尚待服务器验证，不声称已修复模型或提高ACC。

实际通过：四个Python文件AST解析，五个文档bash块及内嵌Python语法检查，训练参数名与真实CLI静态匹配，git diff --check。AST核对原train_one_epoch、evaluate、选模、参考库、缓存、优化器和损失配置不变；核心模型/SCM/loss.py无文件差异。这些不是服务器运行或单元测试通过记录。

先判断大batch是否稳定、实际更新数和事实学习趋势，再决定合理更新预算或潜在中介学习方案；不把“每轮更新十倍减少”忽略后继续盲目比较相同20轮。不能仅看最高测试ACC，继续记录两路径、矩阵和中介尺度，保留test-selected checkpoint偏差。即使分类改善也不自动证明真实中介或因果效应已识别。
