#!/usr/bin/env bash
# 独立DEAP fold2单因素对照：只降低fire权重；测试、训练和上传一并执行。
set -euo pipefail
if [ "$#" -ne 3 ]; then
    echo '用法：bash scripts/run_fire_weight_control_and_upload.sh 数据根绝对目录 输出根绝对目录 唯一运行名' >&2
    exit 2
fi
FIRE_ROOT=$1; FIRE_OUTPUT=$2; FIRE_NAME=$3
case "$FIRE_ROOT:$FIRE_OUTPUT" in /*:/*) ;; *) exit 2 ;; esac
case "$FIRE_NAME" in ''|*/*|.|..) exit 2 ;; esac
FIRE_SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
FIRE_PROJECT=$(dirname -- "$FIRE_SCRIPT_DIR")
FIRE_RESULT="$FIRE_OUTPUT/DEAP/arousal/seed42/$FIRE_NAME"
test ! -e "$FIRE_RESULT"
mkdir -p "$FIRE_RESULT"
FIRE_RUN_EXIT=0
(
    cd "$FIRE_PROJECT"
    bash scripts/phase3_run_logged.sh "$FIRE_RESULT/interface_tests.log" \
      env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
) || FIRE_RUN_EXIT=$?
if [ "$FIRE_RUN_EXIT" -eq 0 ]; then
    bash "$FIRE_SCRIPT_DIR/phase3_run_logged.sh" "$FIRE_RESULT/train.log" \
      python -u "$FIRE_PROJECT/mainNeuroFDAN.py" \
      --dataset DEAP --label arousal --root_dir "$FIRE_ROOT" \
      --output_dir "$FIRE_OUTPUT" --run_name "$FIRE_NAME" \
      --seed 42 --device "${NEURO_DEVICE:-cuda:0}" --held_out_subjects 2 \
      --epochs 100 --lr 0.00002 --weight_decay 0 \
      --batch_size 400 --eval_batch_size 400 --workers 0 --sampling_mode window \
      --fd_mode full --no_balance --mediator_dim 64 --hidden_dim 128 --lif_neurons 256 \
      --dropout 0.3 --ref_dim 0 --time_steps 4 --logvar_min -6 --logvar_max 2 \
      --S_train 4 --K_train 16 --S_eval 16 --K_eval 64 --N_ref 2048 --reference_chunk_size 4 \
      --lambda_obs 1 --beta 0.001 --beta_warmup_epochs 10 --fire_weight 0.028 \
      --diagnostic_level full --log_detail_interval 10 --loss_gradient_interval 10 \
      --max_train_batches 0 --max_test_batches 0 || FIRE_RUN_EXIT=$?
fi
echo "服务器测试/对照退出码=$FIRE_RUN_EXIT；尝试自动上传已有产物。"
if [ -z "${GIT_SSH_COMMAND:-}" ]; then
    export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
fi
FIRE_UPLOAD_ARGS=("$FIRE_RESULT" --log "$FIRE_RESULT/interface_tests.log" \
                  --git_timeout 600 --max_file_mib 16 --max_total_mib 64)
bash "$FIRE_SCRIPT_DIR/upload_run_diagnostics.sh" "${FIRE_UPLOAD_ARGS[@]}" --dry-run
FIRE_UPLOAD_ID="$(date +%Y%m%d_%H%M%S)_$$"
FIRE_UPLOAD_LOG="$FIRE_OUTPUT/upload_logs/${FIRE_NAME}_${FIRE_UPLOAD_ID}.log"
FIRE_BUNDLE="$FIRE_OUTPUT/upload_bundles/${FIRE_NAME}_${FIRE_UPLOAD_ID}"
test ! -e "$FIRE_BUNDLE"
bash "$FIRE_SCRIPT_DIR/phase3_run_logged.sh" "$FIRE_UPLOAD_LOG" \
  bash "$FIRE_SCRIPT_DIR/upload_run_diagnostics.sh" "${FIRE_UPLOAD_ARGS[@]}" --bundle_dir "$FIRE_BUNDLE"
cat "$FIRE_BUNDLE/upload_state.json"
echo '运行、退出码、跳过项和上传收据已自动记录；回传SHA/链接即可。'
exit "$FIRE_RUN_EXIT"
