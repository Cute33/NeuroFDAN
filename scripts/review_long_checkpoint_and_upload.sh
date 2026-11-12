#!/usr/bin/env bash
# 一次执行：服务器测试→只读分项梯度→预览→独立上传；训练或测试失败也上传已有日志。
set -euo pipefail
if [ "$#" -ne 6 ]; then
    echo '用法：bash scripts/review_long_checkpoint_and_upload.sh DEAP|HCI arousal|valence 原fold绝对目录 checkpoint轮数 新输出根绝对目录 唯一运行名' >&2
    exit 2
fi
REVIEW_DATASET=$1; REVIEW_LABEL=$2; REVIEW_FOLD=$3
REVIEW_EPOCH=$4; REVIEW_OUTPUT=$5; REVIEW_NAME=$6
REVIEW_SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REVIEW_PROJECT=$(dirname -- "$REVIEW_SCRIPT_DIR")
case "$REVIEW_DATASET" in DEAP|HCI) ;; *) exit 2 ;; esac
case "$REVIEW_LABEL" in arousal|valence) ;; *) exit 2 ;; esac
case "$REVIEW_FOLD:$REVIEW_OUTPUT" in /*:/*) ;; *) echo 'fold与输出必须为绝对目录' >&2; exit 2 ;; esac
case "$REVIEW_NAME" in ''|*/*|.|..) echo '需要独立运行名' >&2; exit 2 ;; esac
[[ "$REVIEW_EPOCH" =~ ^[1-9][0-9]*$ ]] || exit 2
REVIEW_SUBJECT=$(basename -- "$REVIEW_FOLD")
[[ "$REVIEW_SUBJECT" =~ ^fold([0-9]+)$ ]] || exit 2
REVIEW_SUBJECT=$((10#${BASH_REMATCH[1]}))
REVIEW_RESULT="$REVIEW_OUTPUT/$REVIEW_DATASET/$REVIEW_LABEL/seed42/$REVIEW_NAME"
test ! -e "$REVIEW_RESULT"
test -f "$REVIEW_FOLD/last.pt"
test -f "$REVIEW_FOLD/fold.json"
mkdir -p "$REVIEW_RESULT"
REVIEW_RUN_EXIT=0
(
    cd "$REVIEW_PROJECT"
    bash scripts/phase3_run_logged.sh "$REVIEW_RESULT/interface_tests.log" \
      env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
) || REVIEW_RUN_EXIT=$?
if [ "$REVIEW_RUN_EXIT" -eq 0 ]; then
    bash "$REVIEW_SCRIPT_DIR/phase3_run_logged.sh" "$REVIEW_RESULT/train.log" \
      python -u "$REVIEW_SCRIPT_DIR/audit_checkpoint_gradient_terms.py" \
      --dataset "$REVIEW_DATASET" --label "$REVIEW_LABEL" --root_dir /home/fss \
      --fold_dir "$REVIEW_FOLD" --checkpoint_epoch "$REVIEW_EPOCH" \
      --output_dir "$REVIEW_OUTPUT" --run_name "$REVIEW_NAME" \
      --seed 42 --device cpu --held_out_subjects "$REVIEW_SUBJECT" --epochs 0 \
      --batch_size 16 --reference_samples 64 --workers 0 --diagnostic_level full \
      || REVIEW_RUN_EXIT=$?
fi
echo "审查步骤退出码=$REVIEW_RUN_EXIT；即使非零，也尝试上传已生成日志及收据。"
# 不覆盖调用方认证；未指定时只为本脚本使用已验证的SSH443配置。
if [ -z "${GIT_SSH_COMMAND:-}" ]; then
    export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
fi
REVIEW_UPLOAD_ARGS=("$REVIEW_RESULT" --log "$REVIEW_RESULT/interface_tests.log" \
                   --git_timeout 600 --max_file_mib 16 --max_total_mib 64)
bash "$REVIEW_SCRIPT_DIR/upload_run_diagnostics.sh" "${REVIEW_UPLOAD_ARGS[@]}" --dry-run
REVIEW_UPLOAD_ID="$(date +%Y%m%d_%H%M%S)_$$"
REVIEW_UPLOAD_LOG="$REVIEW_OUTPUT/upload_logs/${REVIEW_NAME}_${REVIEW_UPLOAD_ID}.log"
REVIEW_BUNDLE="$REVIEW_OUTPUT/upload_bundles/${REVIEW_NAME}_${REVIEW_UPLOAD_ID}"
test ! -e "$REVIEW_BUNDLE"
bash "$REVIEW_SCRIPT_DIR/phase3_run_logged.sh" "$REVIEW_UPLOAD_LOG" \
  bash "$REVIEW_SCRIPT_DIR/upload_run_diagnostics.sh" "${REVIEW_UPLOAD_ARGS[@]}" \
  --bundle_dir "$REVIEW_BUNDLE"
cat "$REVIEW_BUNDLE/upload_state.json"
echo "运行/测试日志和退出收据、实际轮数与跳过原因已自动记录；回传本次SHA和链接即可。"
exit "$REVIEW_RUN_EXIT"
