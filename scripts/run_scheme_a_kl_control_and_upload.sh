#!/usr/bin/env bash
# 仅KL单因素、最多300步私有拟合；无论是否通过，均不启动正式fold训练。
set -euo pipefail
if [ "$#" -ne 4 ]; then
    echo '用法：bash scripts/run_scheme_a_kl_control_and_upload.sh 数据根绝对目录 新输出根绝对目录 新运行名 原probe_summary绝对路径' >&2
    exit 2
fi
KLC_ROOT=$1; KLC_OUTPUT=$2; KLC_NAME=$3; KLC_BASELINE=$4
case "$KLC_ROOT:$KLC_OUTPUT:$KLC_BASELINE" in /*:/*:/*) ;; *) exit 2 ;; esac
case "$KLC_NAME" in ''|*/*|.|..) exit 2 ;; esac
KLC_SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
KLC_PROJECT=$(dirname -- "$KLC_SCRIPT_DIR")
KLC_RESULT="$KLC_OUTPUT/DEAP/arousal/seed42/$KLC_NAME"
if [ ! -f "$KLC_BASELINE" ]; then
    echo "找不到原基线摘要：$KLC_BASELINE；停止，不重新训练原基线。" >&2
    exit 2
fi
if [ -e "$KLC_RESULT" ]; then
    echo "新运行目录已存在：$KLC_RESULT；停止以免重复拟合。若只是上传失败，请回传既有日志/收据。" >&2
    exit 2
fi
mkdir -p "$KLC_RESULT"
python - "$KLC_RESULT/config.json" "$KLC_PROJECT" "$KLC_BASELINE" <<'PY'
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from run_reporting import source_version
baseline = Path(sys.argv[3])
with Path(sys.argv[1]).open('x', encoding='utf-8') as stream:
    json.dump(dict(operation='方案A KL单因素固定训练子集拟合对照',
        source_version=source_version(Path(sys.argv[2])),
        changed_training_factor={'beta': [.001, 0.]},
        baseline_summary_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
        baseline_modified=False, full_fold_training=False, automatic_fold_training=False,
        planned_max_private_steps=300, stages={}), stream, ensure_ascii=False, indent=2)
PY
klc_stage() {
    python - "$KLC_RESULT/config.json" "$1" "$2" "$3" <<'PY'
import json,sys
from pathlib import Path
p = Path(sys.argv[1]); value = json.loads(p.read_text())
value['stages'][sys.argv[2]] = dict(status=sys.argv[3], exit_code=int(sys.argv[4]))
t = p.with_suffix('.tmp'); t.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8'); t.replace(p)
PY
}
KLC_RUN_EXIT=0
(
    cd "$KLC_PROJECT"
    bash scripts/phase3_run_logged.sh "$KLC_RESULT/interface_tests.log" \
        env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u - "$KLC_PROJECT/tests" <<'PY'
import sys,unittest
suite = unittest.TestSuite()
for pattern in ('test_scheme_a.py', 'test_scheme_a_control.py'):
    suite.addTests(unittest.TestLoader().discover(sys.argv[1],pattern=pattern))
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
PY
) || KLC_RUN_EXIT=$?
klc_stage tests "$([ "$KLC_RUN_EXIT" -eq 0 ] && echo passed || echo failed)" "$KLC_RUN_EXIT"
if [ "$KLC_RUN_EXIT" -eq 0 ]; then
    # 和原基线逐项相同；唯一训练因素beta=0。epochs20仅保留原接口元数据，不执行20轮。
    bash "$KLC_SCRIPT_DIR/phase3_run_logged.sh" "$KLC_RESULT/train.log" \
        python -u "$KLC_SCRIPT_DIR/scheme_a_fit_check.py" \
        --dataset DEAP --label arousal --root_dir "$KLC_ROOT" \
        --output_dir "$KLC_OUTPUT" --run_name "$KLC_NAME" \
        --seed 42 --device cuda:0 --held_out_subjects 2 --epochs 20 \
        --lr 0.00002 --weight_decay 0 --batch_size 400 --eval_batch_size 400 --workers 0 \
        --sampling_mode window --fd_mode full --no_balance --mediator_dim 64 --hidden_dim 128 \
        --lif_neurons 256 --dropout 0.3 --ref_dim 0 --time_steps 4 --logvar_min -6 --logvar_max 2 \
        --S_train 4 --K_train 16 --S_eval 16 --K_eval 64 --N_ref 2048 --reference_chunk_size 4 \
        --classification_objective integrated_observational --lambda_obs 0 --factual_ce_mode all_samples \
        --beta 0 --beta_warmup_epochs 10 --fire_weight 0.028 \
        --diagnostic_level full --log_detail_interval 5 --loss_gradient_interval 5 \
        --max_train_batches 0 --max_test_batches 0 \
        --fit_max_steps 300 --fit_check_every 50 \
        --fit_baseline_summary "$KLC_BASELINE" --fit_save_checkpoint || KLC_RUN_EXIT=$?
    case "$KLC_RUN_EXIT" in 0) KLC_STATUS=passed ;; 3) KLC_STATUS=budget_not_passed ;; *) KLC_STATUS=failed ;; esac
    klc_stage private_fit "$KLC_STATUS" "$KLC_RUN_EXIT"
fi
if [ -f "$KLC_RESULT/probe_summary.json" ]; then
    KLC_SUMMARY_EXIT=0
    bash "$KLC_SCRIPT_DIR/phase3_run_logged.sh" "$KLC_RESULT/summary.log" \
        python -u "$KLC_SCRIPT_DIR/summarize_scheme_a_fit.py" "$KLC_RESULT" || KLC_SUMMARY_EXIT=$?
    klc_stage summary "$([ "$KLC_SUMMARY_EXIT" -eq 0 ] && echo completed || echo failed)" "$KLC_SUMMARY_EXIT"
    KLC_COMPARE_EXIT=0
    bash "$KLC_SCRIPT_DIR/phase3_run_logged.sh" "$KLC_RESULT/comparison.log" \
        python -u "$KLC_SCRIPT_DIR/scheme_a_control_evidence.py" \
        --baseline "$KLC_BASELINE" --result "$KLC_RESULT/probe_summary.json" \
        --output "$KLC_RESULT/run_summary.json" || KLC_COMPARE_EXIT=$?
    klc_stage comparison "$([ "$KLC_COMPARE_EXIT" -eq 0 ] && echo completed || echo failed)" "$KLC_COMPARE_EXIT"
    if [ "$KLC_SUMMARY_EXIT" -ne 0 ] || [ "$KLC_COMPARE_EXIT" -ne 0 ]; then KLC_RUN_EXIT=1; fi
fi
echo "对照步骤退出码=$KLC_RUN_EXIT；0=通过拟合，3=预算内未通过，其他=错误；均不自动运行正式fold。"
if [ -z "${GIT_SSH_COMMAND:-}" ]; then
    export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
fi
KLC_UPLOAD_ARGS=("$KLC_RESULT" --git_timeout 600 --max_file_mib 16 --max_total_mib 64)
for KLC_LOG in interface_tests.log summary.log comparison.log; do
    if [ -f "$KLC_RESULT/$KLC_LOG" ]; then KLC_UPLOAD_ARGS+=(--log "$KLC_RESULT/$KLC_LOG"); fi
done
bash "$KLC_SCRIPT_DIR/upload_run_diagnostics.sh" "${KLC_UPLOAD_ARGS[@]}" --dry-run
KLC_UPLOAD_ID="$(date +%Y%m%d_%H%M%S)_$$"
KLC_UPLOAD_LOG="$KLC_OUTPUT/upload_logs/${KLC_NAME}_${KLC_UPLOAD_ID}.log"
KLC_BUNDLE="$KLC_OUTPUT/upload_bundles/${KLC_NAME}_${KLC_UPLOAD_ID}"
test ! -e "$KLC_BUNDLE"
KLC_UPLOAD_EXIT=0
bash "$KLC_SCRIPT_DIR/phase3_run_logged.sh" "$KLC_UPLOAD_LOG" \
    bash "$KLC_SCRIPT_DIR/upload_run_diagnostics.sh" "${KLC_UPLOAD_ARGS[@]}" \
    --bundle_dir "$KLC_BUNDLE" || KLC_UPLOAD_EXIT=$?
cat "$KLC_UPLOAD_LOG.exit"
if [ -f "$KLC_BUNDLE/upload_state.json" ]; then cat "$KLC_BUNDLE/upload_state.json"; fi
if [ "$KLC_UPLOAD_EXIT" -ne 0 ]; then
    echo '上传未确认成功；回传终端摘要/收据，不重跑拟合。' >&2
    exit "$KLC_UPLOAD_EXIT"
fi
echo '日志、实际步数、配对比较和退出收据已上传；私有权重仅留服务器。回传SHA和GitHub链接即可。'
exit "$KLC_RUN_EXIT"
