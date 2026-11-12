#!/usr/bin/env bash
# 读取已有结果→标准库/本地Git测试→预览→自动上传；不执行模型、优化器或训练入口。
set -euo pipefail
if [ "$#" -ne 2 ]; then
    echo '用法：bash scripts/recover_scheme_a_upload.sh 已有方案A运行绝对目录 独立恢复输出绝对目录' >&2
    exit 2
fi
RECOVERY_RUN=$1
RECOVERY_OUTPUT=$2
case "$RECOVERY_RUN:$RECOVERY_OUTPUT" in /*:/*) ;; *) exit 2 ;; esac
RECOVERY_SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
RECOVERY_PROJECT=$(dirname -- "$RECOVERY_SCRIPT_DIR")
test -f "$RECOVERY_RUN/probe_summary.json"
python - "$RECOVERY_RUN" "$RECOVERY_OUTPUT" <<'PY'
import sys
from pathlib import Path
run, output = (Path(p).resolve() for p in sys.argv[1:])
if output == run or run in output.parents:
    raise SystemExit('恢复日志和快照必须放在原实验目录之外')
PY
RECOVERY_ID="$(date +%Y%m%d_%H%M%S)_$$"
RECOVERY_LOG_DIR="$RECOVERY_OUTPUT/logs/$RECOVERY_ID"
RECOVERY_BUNDLE="$RECOVERY_OUTPUT/bundles/$RECOVERY_ID"
test ! -e "$RECOVERY_LOG_DIR"
test ! -e "$RECOVERY_BUNDLE"
mkdir -p "$RECOVERY_LOG_DIR"

# 先输出摘要，网络失败时也有混淆矩阵/基线/梯度可供分析。
RECOVERY_SUMMARY_EXIT=0
bash "$RECOVERY_SCRIPT_DIR/phase3_run_logged.sh" "$RECOVERY_LOG_DIR/summary.log" \
    python -u "$RECOVERY_SCRIPT_DIR/summarize_scheme_a_fit.py" "$RECOVERY_RUN" || RECOVERY_SUMMARY_EXIT=$?
echo "已有JSON摘要退出码=$RECOVERY_SUMMARY_EXIT；原运行文件保持不变。"

# 只运行上传器及摘要器测试；不重新运行181项模型测试、CLI或300步拟合。
RECOVERY_TEST_EXIT=0
(
    cd "$RECOVERY_PROJECT"
    bash "$RECOVERY_SCRIPT_DIR/phase3_run_logged.sh" "$RECOVERY_LOG_DIR/recovery_tests.log" \
        python -u - "$RECOVERY_PROJECT/tests" <<'PY'
import sys
import unittest
suite = unittest.TestSuite()
for pattern in ('test_upload*.py', 'test_auto_evidence.py', 'test_scheme_a_summary.py'):
    suite.addTests(unittest.TestLoader().discover(sys.argv[1], pattern=pattern))
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
PY
) || RECOVERY_TEST_EXIT=$?
if [ "$RECOVERY_TEST_EXIT" -ne 0 ]; then
    echo "上传器测试未通过，停止发布。摘要/测试日志保留在：$RECOVERY_LOG_DIR" >&2
    echo '无需重训；回传本次终端输出即可。' >&2
    exit "$RECOVERY_TEST_EXIT"
fi

if [ -z "${GIT_SSH_COMMAND:-}" ]; then
    export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
fi
RECOVERY_UPLOAD_ARGS=("$RECOVERY_RUN" --git_timeout 600 --max_file_mib 16 --max_total_mib 64
    --log "$RECOVERY_LOG_DIR/summary.log" --log "$RECOVERY_LOG_DIR/recovery_tests.log")
for RECOVERY_LOG_NAME in interface_tests.log fit_check.log train.log; do
    if [ -f "$RECOVERY_RUN/$RECOVERY_LOG_NAME" ]; then
        RECOVERY_UPLOAD_ARGS+=(--log "$RECOVERY_RUN/$RECOVERY_LOG_NAME")
    fi
done
bash "$RECOVERY_SCRIPT_DIR/phase3_run_logged.sh" "$RECOVERY_LOG_DIR/preview.log" \
    bash "$RECOVERY_SCRIPT_DIR/upload_run_diagnostics.sh" "${RECOVERY_UPLOAD_ARGS[@]}" --dry-run
RECOVERY_UPLOAD_EXIT=0
bash "$RECOVERY_SCRIPT_DIR/phase3_run_logged.sh" "$RECOVERY_LOG_DIR/upload.log" \
    bash "$RECOVERY_SCRIPT_DIR/upload_run_diagnostics.sh" "${RECOVERY_UPLOAD_ARGS[@]}" \
    --bundle_dir "$RECOVERY_BUNDLE" || RECOVERY_UPLOAD_EXIT=$?
cat "$RECOVERY_LOG_DIR/upload.log.exit"
if [ -f "$RECOVERY_BUNDLE/upload_state.json" ]; then
    cat "$RECOVERY_BUNDLE/upload_state.json"
fi
echo "恢复日志：$RECOVERY_LOG_DIR"
echo "恢复快照：$RECOVERY_BUNDLE"
if [ "$RECOVERY_UPLOAD_EXIT" -ne 0 ]; then
    echo '上传未确认成功。无需重训；本次摘要已打印，回传终端输出供继续分析。' >&2
    exit "$RECOVERY_UPLOAD_EXIT"
fi
echo '结果已独立上传。回传本次SHA和GitHub链接即可；不必手工摘抄矩阵或manifest。'
