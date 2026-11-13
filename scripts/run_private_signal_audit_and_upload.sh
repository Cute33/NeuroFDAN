#!/usr/bin/env bash
# 只读已有私有权重；测试/推理结束后自动回传日志，不调用训练入口。
set -euo pipefail
if [ "$#" -ne 3 ]; then
    echo '用法：脚本 数据根 已有私有checkpoint 新结果绝对目录' >&2; exit 2
fi
AUDIT_ROOT=$1; AUDIT_CKPT=$2; AUDIT_RESULT=$3
AUDIT_SCRIPTS=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
test -f "$AUDIT_CKPT"
test ! -e "$AUDIT_RESULT"
AUDIT_SIDE="${AUDIT_RESULT}_logs"
test ! -e "$AUDIT_SIDE"
mkdir -p "$AUDIT_SIDE"
AUDIT_EXIT=0
(
cd "$AUDIT_SCRIPTS/.."
bash scripts/phase3_run_logged.sh "$AUDIT_SIDE/tests.log" \
    env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -p test_private_signal_audit.py -v
) || AUDIT_EXIT=$?
if [ "$AUDIT_EXIT" -eq 0 ]; then
    bash "$AUDIT_SCRIPTS/phase3_run_logged.sh" "$AUDIT_SIDE/train.log" \
        python -u "$AUDIT_SCRIPTS/audit_private_signal.py" \
        --root_dir "$AUDIT_ROOT" --checkpoint "$AUDIT_CKPT" \
        --output_dir "$AUDIT_RESULT" --device cuda:0 || AUDIT_EXIT=$?
fi
# 启动前报错也保留并上传已有日志，绝不重训。
mkdir -p "$AUDIT_RESULT"
echo "只读审查退出码=$AUDIT_EXIT；开始上传已有诊断，optimizer step始终为0。"
if [ -z "${GIT_SSH_COMMAND:-}" ]; then
    export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
fi
AUDIT_ARGS=("$AUDIT_RESULT" --log "$AUDIT_SIDE/tests.log" --git_timeout 600 --max_file_mib 16 --max_total_mib 64)
if [ -f "$AUDIT_SIDE/train.log" ]; then AUDIT_ARGS+=(--log "$AUDIT_SIDE/train.log"); fi
bash "$AUDIT_SCRIPTS/upload_run_diagnostics.sh" "${AUDIT_ARGS[@]}" --dry-run
AUDIT_ID="$(date +%Y%m%d_%H%M%S)_$$"
AUDIT_BUNDLE="${AUDIT_RESULT}_bundle_${AUDIT_ID}"
bash "$AUDIT_SCRIPTS/phase3_run_logged.sh" "$AUDIT_SIDE/upload_${AUDIT_ID}.log" \
    bash "$AUDIT_SCRIPTS/upload_run_diagnostics.sh" "${AUDIT_ARGS[@]}" --bundle_dir "$AUDIT_BUNDLE"
cat "$AUDIT_BUNDLE/upload_state.json"
echo '回传SHA及链接即可；无需再次运行拟合或手工摘抄诊断。'
exit "$AUDIT_EXIT"
