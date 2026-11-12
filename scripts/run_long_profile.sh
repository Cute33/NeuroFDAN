#!/usr/bin/env bash
# 明确联合配置：小学习率/长程/匹配batch/LIF256/fire2.8；不改SCM或主输出。
set -euo pipefail
if [ "$#" -ne 6 ]; then
    echo '用法：bash scripts/run_long_profile.sh smoke|train|resume500 DEAP|HCI arousal|valence 数据根目录 输出根目录 运行名' >&2
    echo '固定lr=2e-5，train总300轮，resume500总500轮；NEURO_DEVICE可选，默认cuda:0。' >&2
    exit 2
fi
LONG_MODE=$1; LONG_DATASET=$2; LONG_LABEL=$3
LONG_ROOT=$4; LONG_OUTPUT=$5; LONG_NAME=$6
LONG_SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
LONG_PROJECT=$(dirname -- "$LONG_SCRIPT_DIR")
case "$LONG_DATASET" in
    DEAP) LONG_BATCH=400; LONG_SUBJECT=2 ;;
    HCI) LONG_BATCH=100; LONG_SUBJECT=1 ;;
    *) echo '数据集仅允许DEAP/HCI' >&2; exit 2 ;;
esac
case "$LONG_LABEL" in arousal|valence) ;; *) echo '任务仅允许arousal/valence' >&2; exit 2 ;; esac
case "$LONG_ROOT:$LONG_OUTPUT" in /*:/*) ;; *) echo '数据和输出根目录必须为绝对路径' >&2; exit 2 ;; esac
case "$LONG_NAME" in ''|*/*|.|..) echo '运行名不能为空或包含路径' >&2; exit 2 ;; esac
LONG_RESULT="$LONG_OUTPUT/$LONG_DATASET/$LONG_LABEL/seed42/$LONG_NAME"
LONG_EXTRA=()
case "$LONG_MODE" in
    smoke) LONG_EPOCHS=1; LONG_LIMIT=1; test ! -e "$LONG_RESULT" ;;
    train) LONG_EPOCHS=300; LONG_LIMIT=0; test ! -e "$LONG_RESULT" ;;
    resume500)
        LONG_EPOCHS=500; LONG_LIMIT=0
        LONG_CHECKPOINT="$LONG_RESULT/$(printf 'fold%02d' "$LONG_SUBJECT")/last.pt"
        test -f "$LONG_CHECKPOINT"
        LONG_EXTRA=(--resume "$LONG_CHECKPOINT") ;;
    *) echo '阶段仅允许smoke/train/resume500' >&2; exit 2 ;;
esac
bash "$LONG_SCRIPT_DIR/phase3_run_logged.sh" "$LONG_RESULT/train.log" \
  python -u "$LONG_PROJECT/mainNeuroFDAN.py" \
  --dataset "$LONG_DATASET" --label "$LONG_LABEL" --root_dir "$LONG_ROOT" \
  --output_dir "$LONG_OUTPUT" --run_name "$LONG_NAME" \
  --seed 42 --device "${NEURO_DEVICE:-cuda:0}" --held_out_subjects "$LONG_SUBJECT" \
  --epochs "$LONG_EPOCHS" --lr 0.00002 --weight_decay 0 \
  --batch_size "$LONG_BATCH" --eval_batch_size "$LONG_BATCH" --workers 0 --sampling_mode window \
  --fd_mode full --no_balance --mediator_dim 64 --hidden_dim 128 --lif_neurons 256 \
  --dropout 0.3 --ref_dim 0 --time_steps 4 --logvar_min -6 --logvar_max 2 \
  --S_train 4 --K_train 16 --S_eval 16 --K_eval 64 --N_ref 2048 --reference_chunk_size 4 \
  --lambda_obs 1 --beta 0.001 --beta_warmup_epochs 10 --fire_weight 2.8 \
  --diagnostic_level full --log_detail_interval 10 \
  --max_train_batches "$LONG_LIMIT" --max_test_batches "$LONG_LIMIT" "${LONG_EXTRA[@]}"
cat "$LONG_RESULT/train.log.exit"
