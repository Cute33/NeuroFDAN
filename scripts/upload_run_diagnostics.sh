#!/usr/bin/env bash
# 只启动上传工具；不切换实验代码分支，不安装依赖，不运行训练。
set -eu
phase5_script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python "${phase5_script_dir}/upload_run_diagnostics.py" "$@"
