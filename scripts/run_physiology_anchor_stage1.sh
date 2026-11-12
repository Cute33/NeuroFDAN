#!/usr/bin/env bash
# 第一阶段仅执行小测试、训练来源锚点审查及1轮配对合成，不训练真实模型。
set -euo pipefail
if [ "$#" -ne 2 ]; then
    echo '用法：bash scripts/run_physiology_anchor_stage1.sh 数据根目录 全新阶段结果目录' >&2
    exit 2
fi
ANCHOR_DATA_ROOT=$1
ANCHOR_STAGE=$2
case "$ANCHOR_DATA_ROOT:$ANCHOR_STAGE" in
    /*:/*) ;;
    *) echo '数据根目录和输出目录必须是绝对路径，避免切换工具工作区后改变指向' >&2; exit 2 ;;
esac
ANCHOR_SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ANCHOR_PROJECT=$(dirname -- "$ANCHOR_SCRIPT_DIR")
if [ -e "$ANCHOR_STAGE" ]; then
    echo '阶段结果目录已存在，不覆盖；请使用新的阶段名' >&2
    exit 1
fi
mkdir -p "$ANCHOR_STAGE"
cd "$ANCHOR_PROJECT"
bash scripts/phase3_run_logged.sh "$ANCHOR_STAGE/interface_tests.log" \
    env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    python -m unittest discover -s tests -p test_physiological_anchors.py -v
for ANCHOR_DATASET in DEAP HCI; do
    if [ "$ANCHOR_DATASET" = DEAP ]; then ANCHOR_PREFIX=deap; else ANCHOR_PREFIX=hci; fi
    ANCHOR_AUDIT_NAME="${ANCHOR_PREFIX}_training_anchor_audit_001"
    bash scripts/phase3_run_logged.sh "$ANCHOR_STAGE/$ANCHOR_AUDIT_NAME/train.log" \
        env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        python -u scripts/audit_physiological_anchors.py \
        --dataset "$ANCHOR_DATASET" --label arousal --root_dir "$ANCHOR_DATA_ROOT" \
        --output_dir "$ANCHOR_STAGE" --run_name "$ANCHOR_AUDIT_NAME" \
        --held_out_subjects 1 --seed 42 --device cpu --batch_size 40 --workers 0 --diagnostic_level full
done
bash scripts/phase3_run_logged.sh "$ANCHOR_STAGE/synthetic_normal_smoke_001/train.log" \
    env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    python -u physiology_anchor_experiment.py \
    --output_dir "$ANCHOR_STAGE" --run_name synthetic_normal_smoke_001 \
    --seed 42 --device cpu --workers 0 --diagnostic_level full \
    --epochs 1 --batch_size 64 --train_size 128 --test_size 32 --points 4 \
    --gammas 1 --mechanisms aligned misaligned --lr 0.001 \
    --alpha_c 0.1 --alpha_a 0.1 --mediator_family normal \
    --mc_repeats 2 --direct_n 4096 --oracle_s 128 --oracle_k 128 --smoke --plots
echo '阶段1执行结束；检查各probe_summary的completed、质量统计及smoke标记。未执行真实模型训练。'
