# Phase6服务器操作与结果回传

本轮无需运行mainNeuroFDAN.py。不训练DEAP/HCI、不跑GPU或LOSO。使用现有dslenv、原数据和原epoch20 checkpoint；新结果统一在outputs/phase6。错误时保存日志，不清空原目录或重跑旧训练。

## 1. 获取独立分支与六项针对性测试

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
git status --short
git worktree list
timeout 600 git fetch origin \
  refs/heads/feat/neurofdan-phase6-mediator-learning:refs/remotes/origin/feat/neurofdan-phase6-mediator-learning
PHASE6_TOOLS=/home/fss/NeuroFDAN-phase6-tools
if [ ! -e "$PHASE6_TOOLS" ]; then
    git worktree add --detach "$PHASE6_TOOLS" origin/feat/neurofdan-phase6-mediator-learning
fi
test "$(git -C "$PHASE6_TOOLS" rev-parse HEAD)" = "$(git rev-parse origin/feat/neurofdan-phase6-mediator-learning)"
test -z "$(git -C "$PHASE6_TOOLS" status --porcelain)"
mkdir -p outputs/phase6
bash "$PHASE6_TOOLS/scripts/phase3_run_logged.sh" \
  /home/fss/NeuroFDAN/outputs/phase6/phase6_interface_tests_001.log \
  bash -c 'cd "$1" && python -m unittest discover -s tests -p test_phase6_learning.py -v' _ "$PHASE6_TOOLS"
)
```

测试失败先反馈phase6_interface_tests_001.log和.exit，不执行后续步骤。云端未运行这些测试。后续合成不读真实数据；随机张量测试不是生理数据实验。

## 2. 两个数据集原权重分项梯度（只读）

复用原支持审计16个查询，原参考库内64成员；epochs=0，不创建优化器。batch_size=16是探针规模，不改变DEAP400/HCI100。GitHub不含权重，下面只读服务器原last.pt及本地支持摘要。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
PHASE6_TOOLS=/home/fss/NeuroFDAN-phase6-tools
for PHASE6_DATASET in DEAP HCI
do
    if [ "$PHASE6_DATASET" = DEAP ]; then PHASE6_PREFIX=deap; PHASE6_BATCH=400; else PHASE6_PREFIX=hci; PHASE6_BATCH=100; fi
    PHASE6_RUN="${PHASE6_PREFIX}_arousal_fold01_gradient_001"
    PHASE6_DEST="/home/fss/NeuroFDAN/outputs/phase6/gradients/$PHASE6_DATASET/arousal/seed42/$PHASE6_RUN"
    mkdir -p "$(dirname "$PHASE6_DEST")"
    mkdir "$PHASE6_DEST"
    bash "$PHASE6_TOOLS/scripts/phase3_run_logged.sh" "$PHASE6_DEST/train.log" \
      python -u "$PHASE6_TOOLS/scripts/phase6_gradient_audit.py" \
      --dataset "$PHASE6_DATASET" --label arousal --root_dir /home/fss \
      --fold_dir "/home/fss/NeuroFDAN/outputs/phase5_large_batch/$PHASE6_DATASET/arousal/seed42/${PHASE6_PREFIX}_arousal_fold01_b${PHASE6_BATCH}_20ep_001/fold01" \
      --support_summary "/home/fss/NeuroFDAN/outputs/phase5_probes/${PHASE6_PREFIX}_arousal_fold01_support_001/probe_summary.json" \
      --output_dir /home/fss/NeuroFDAN/outputs/phase6/gradients \
      --run_name "$PHASE6_RUN" --seed 42 --device cpu \
      --held_out_subjects 1 --epochs 0 --batch_size 16 --workers 0 --diagnostic_level full
    cat "$PHASE6_DEST/train.log.exit"
done
)
```

输出probe_summary.json、train.log及.exit。加权CE/KLc/KLa/fire梯度与CE夹角单列，do_P1_local_derivative只作敏感度，不作为损失。比较train（冻结BN）和eval；不step、不写.grad，校验原模型及文件哈希。

## 3. 三种监督条件的1轮合成smoke

SCM不读取DEAP/HCI，因此没有真实dataset/label/root_dir/held_out参数，不能伪装成真实fold。下面是实际存在的独立主函数。smoke仅接口检查，不能判断模型改善。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
PHASE6_TOOLS=/home/fss/NeuroFDAN-phase6-tools
PHASE6_DEST=/home/fss/NeuroFDAN/outputs/phase6/synthetic/paired_normal_smoke_001
mkdir -p "$(dirname "$PHASE6_DEST")"
mkdir "$PHASE6_DEST"
bash "$PHASE6_TOOLS/scripts/phase3_run_logged.sh" "$PHASE6_DEST/train.log" \
  python -u "$PHASE6_TOOLS/phase6_synthetic_experiment.py" \
  --output_dir /home/fss/NeuroFDAN/outputs/phase6/synthetic --run_name paired_normal_smoke_001 \
  --seed 42 --device cpu --epochs 1 --batch_size 128 --workers 0 --diagnostic_level full \
  --train_size 256 --test_size 64 --points 4 --gammas 1 \
  --lr 0.001 --weight_decay 0 --fidelity_weight 1 --fire_weight 0 \
  --mediator_family normal --mc_repeats 2 --direct_n 4096 --oracle_s 128 --oracle_k 128 --smoke
cat "$PHASE6_DEST/train.log.exit"
)
```

第一批回传两个gradient目录和paired_normal_smoke_001三个记录。先核对这些记录，再执行下一节；不启动真实单fold训练。

## 4. 预定义配对合成（第一批检查通过后执行）

两个学习族各运行三种监督，不选高ACC的族冒充真实新方案。normal相对tanh生成器错设，tanh_normal是匹配控制。lr0.001/batch128为独立合成设置，不修改真实学习率。三组均最后轮评价，无测试或oracle选模。MC重复不是多种子实验。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
PHASE6_TOOLS=/home/fss/NeuroFDAN-phase6-tools
python - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('/home/fss/NeuroFDAN/outputs/phase6/synthetic/paired_normal_smoke_001/probe_summary.json').read_text())
if not r['completed'] or not r['smoke']:
    raise RuntimeError('先完成并核对三组接口检查，不跳过失败')
PY
for PHASE6_FAMILY in normal tanh_normal
do
    PHASE6_RUN="paired_${PHASE6_FAMILY}_20ep_001"
    PHASE6_DEST="/home/fss/NeuroFDAN/outputs/phase6/synthetic/$PHASE6_RUN"
    mkdir "$PHASE6_DEST"
    bash "$PHASE6_TOOLS/scripts/phase3_run_logged.sh" "$PHASE6_DEST/train.log" \
      python -u "$PHASE6_TOOLS/phase6_synthetic_experiment.py" \
      --output_dir /home/fss/NeuroFDAN/outputs/phase6/synthetic --run_name "$PHASE6_RUN" \
      --seed 42 --device cpu --epochs 20 --batch_size 128 --workers 0 --diagnostic_level full \
      --train_size 2048 --test_size 512 --points 16 --gammas 0 1 \
      --lr 0.001 --weight_decay 0 --fidelity_weight 1 --fire_weight 0 \
      --mediator_family "$PHASE6_FAMILY" --mc_repeats 3 \
      --direct_n 32768 --oracle_s 1024 --oracle_k 512
    cat "$PHASE6_DEST/train.log.exit"
done
)
```

每轮打印进度，真值/拟合/评价分别计时，耗时待服务器实测。直接/前门真值差异超容差则保存部分结果并停止；不自动加轮数、放宽阈值或开启真实训练。即便成功，仍须评估识别假设及真实单fold准入条件。

## 5. 上传预览

使用已经服务器验证的6681381工具。未生成的目录会明确跳过；默认第一批仅三个目录。config/history/probe/log在白名单，models/*.pt和scm_parameters.npz不上传。前置测试日志作为额外日志附上。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
UPLOAD_TOOLS=/home/fss/NeuroFDAN-upload-timeout-tools
test "$(git -C "$UPLOAD_TOOLS" rev-parse HEAD)" = 6681381298b74e9b69dc0df0594d5d9c7274acbc
test -z "$(git -C "$UPLOAD_TOOLS" status --porcelain)"
for PHASE6_REL in \
  gradients/DEAP/arousal/seed42/deap_arousal_fold01_gradient_001 \
  gradients/HCI/arousal/seed42/hci_arousal_fold01_gradient_001 \
  synthetic/paired_normal_smoke_001 \
  synthetic/paired_normal_20ep_001 \
  synthetic/paired_tanh_normal_20ep_001
do
    PHASE6_RESULT="/home/fss/NeuroFDAN/outputs/phase6/$PHASE6_REL"
    if [ ! -d "$PHASE6_RESULT" ]; then echo "尚未运行：$PHASE6_RESULT"; continue; fi
    bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" \
      "$PHASE6_RESULT" --log "$PHASE6_RESULT/train.log" \
      --log /home/fss/NeuroFDAN/outputs/phase6/phase6_interface_tests_001.log \
      --git_timeout 600 --dry-run
done
)
```

## 6. 正式上传、日志与持久收据

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
UPLOAD_TOOLS=/home/fss/NeuroFDAN-upload-timeout-tools
test "$(git -C "$UPLOAD_TOOLS" rev-parse HEAD)" = 6681381298b74e9b69dc0df0594d5d9c7274acbc
test -z "$(git -C "$UPLOAD_TOOLS" status --porcelain)"
PHASE6_UPLOAD_ID="$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p outputs/phase6/upload_logs outputs/phase6/upload_bundles
for PHASE6_REL in \
  gradients/DEAP/arousal/seed42/deap_arousal_fold01_gradient_001 \
  gradients/HCI/arousal/seed42/hci_arousal_fold01_gradient_001 \
  synthetic/paired_normal_smoke_001 \
  synthetic/paired_normal_20ep_001 \
  synthetic/paired_tanh_normal_20ep_001
do
    PHASE6_RESULT="/home/fss/NeuroFDAN/outputs/phase6/$PHASE6_REL"
    if [ ! -d "$PHASE6_RESULT" ]; then echo "尚未运行：$PHASE6_RESULT"; continue; fi
    PHASE6_NAME="$(basename "$PHASE6_RESULT")"
    PHASE6_UPLOAD_LOG="/home/fss/NeuroFDAN/outputs/phase6/upload_logs/${PHASE6_NAME}_${PHASE6_UPLOAD_ID}.log"
    PHASE6_BUNDLE="/home/fss/NeuroFDAN/outputs/phase6/upload_bundles/${PHASE6_NAME}_${PHASE6_UPLOAD_ID}"
    bash "$UPLOAD_TOOLS/scripts/phase3_run_logged.sh" "$PHASE6_UPLOAD_LOG" \
      bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" \
      "$PHASE6_RESULT" --log "$PHASE6_RESULT/train.log" \
      --log /home/fss/NeuroFDAN/outputs/phase6/phase6_interface_tests_001.log \
      --git_timeout 600 --bundle_dir "$PHASE6_BUNDLE"
    cat "$PHASE6_UPLOAD_LOG.exit"
    cat "$PHASE6_BUNDLE/upload_state.json"
done
)
```

每次生成唯一bundle/log；若push或核验超时先读收据并按upload_timeout_recovery核验，不盲目重复上传。正常回传每个已运行目录的SHA和链接，第一批预期三组。测试失败时仅发送测试日志与.exit，不继续后续步骤。结果不完整时也可上传已产生日志，云端据completed与退出码判断，不把部分结果当完整实验。不需要凭证或模型权重。
