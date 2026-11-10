#!/usr/bin/env bash
# 继承调用方环境；保存真实管道退出码，不执行环境安装或实验参数改写。
set -u
set -o pipefail
if [ "$#" -lt 2 ]; then
    printf '用法：bash scripts/phase3_run_logged.sh 日志路径 命令 参数...\n' >&2
    exit 2
fi
phase3_log=$1
shift
mkdir -p -- "$(dirname -- "$phase3_log")" || exit 1
"$@" 2>&1 | tee -a -- "$phase3_log"
phase3_codes=("${PIPESTATUS[@]}")
printf '程序退出码=%s，日志tee退出码=%s\n' "${phase3_codes[0]}" "${phase3_codes[1]}"
printf 'python_exit=%s\ntee_exit=%s\n' "${phase3_codes[0]}" "${phase3_codes[1]}" > "${phase3_log}.exit" || exit 1
if [ "${phase3_codes[0]}" -ne 0 ]; then
    exit "${phase3_codes[0]}"
fi
exit "${phase3_codes[1]}"
