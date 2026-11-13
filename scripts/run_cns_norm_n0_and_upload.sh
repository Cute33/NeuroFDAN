#!/usr/bin/env bash
# 先CPU接口/回归测试，再N0合成配对；不自动启动DEAP/HCI或LOSO。
set -euo pipefail
if [ "$#" -ne 1 ]; then echo '用法：脚本 新N0结果绝对目录' >&2; exit 2; fi
NORM_RESULT=$1
case "$NORM_RESULT" in /*) ;; *) exit 2 ;; esac
NORM_SCRIPTS=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
test ! -e "$NORM_RESULT"
NORM_LOGS="${NORM_RESULT}_logs"
test ! -e "$NORM_LOGS"
mkdir -p "$NORM_LOGS"
NORM_EXIT=0
(
cd "$NORM_SCRIPTS/.."
bash scripts/phase3_run_logged.sh "$NORM_LOGS/tests.log" \
    env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python -m unittest discover -s tests -v
) || NORM_EXIT=$?
if [ "$NORM_EXIT" -eq 0 ]; then
    bash "$NORM_SCRIPTS/phase3_run_logged.sh" "$NORM_LOGS/train.log" \
        env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        python -u "$NORM_SCRIPTS/cns_norm_synthetic.py" \
        --output_dir "$NORM_RESULT" --seed 42 --steps 200 || NORM_EXIT=$?
fi
mkdir -p "$NORM_RESULT"
echo "N0步骤退出码=$NORM_EXIT；不启动真实训练，自动上传现有结果。"
if [ -z "${GIT_SSH_COMMAND:-}" ]; then
    export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
fi
NORM_ARGS=("$NORM_RESULT" --log "$NORM_LOGS/tests.log" --git_timeout 600 --max_file_mib 16 --max_total_mib 64)
if [ -f "$NORM_LOGS/train.log" ]; then NORM_ARGS+=(--log "$NORM_LOGS/train.log"); fi
bash "$NORM_SCRIPTS/upload_run_diagnostics.sh" "${NORM_ARGS[@]}" --dry-run
NORM_ID="$(date +%Y%m%d_%H%M%S)_$$"
NORM_BUNDLE="${NORM_RESULT}_bundle_${NORM_ID}"
bash "$NORM_SCRIPTS/phase3_run_logged.sh" "$NORM_LOGS/upload_${NORM_ID}.log" \
    bash "$NORM_SCRIPTS/upload_run_diagnostics.sh" "${NORM_ARGS[@]}" --bundle_dir "$NORM_BUNDLE"
cat "$NORM_BUNDLE/upload_state.json"
echo '回传SHA和GitHub链接即可；不要重复执行同名N0，也不需要手工摘抄测试结果。'
exit "$NORM_EXIT"
