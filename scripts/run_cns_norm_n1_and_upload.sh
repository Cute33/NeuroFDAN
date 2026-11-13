#!/usr/bin/env bash
# 仅CNS入口归一化单因素、最多300步私有拟合；无论是否通过，均不启动正式fold训练。
set -euo pipefail
if [ "$#" -ne 4 ]; then
    echo '用法：bash scripts/run_cns_norm_n1_and_upload.sh 数据根绝对目录 新输出根绝对目录 新运行名 原beta0的probe_summary绝对路径' >&2
    exit 2
fi
N1_ROOT=$1; N1_OUTPUT=$2; N1_NAME=$3; N1_BASELINE=$4
case "$N1_ROOT:$N1_OUTPUT:$N1_BASELINE" in /*:/*:/*) ;; *) exit 2 ;; esac
case "$N1_NAME" in ''|*/*|.|..) exit 2 ;; esac
N1_SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
N1_PROJECT=$(dirname -- "$N1_SCRIPT_DIR")
N1_RESULT="$N1_OUTPUT/DEAP/arousal/seed42/$N1_NAME"
if [ ! -f "$N1_BASELINE" ]; then
    echo "找不到原基线摘要：$N1_BASELINE；停止，不重新训练原基线。" >&2
    exit 2
fi
if [ -e "$N1_RESULT" ]; then
    echo "新运行目录已存在：$N1_RESULT；停止以免重复拟合。若只是上传失败，请回传既有日志/收据。" >&2
    exit 2
fi
mkdir -p "$N1_RESULT"
python - "$N1_RESULT/config.json" "$N1_PROJECT" "$N1_BASELINE" <<'PY'
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from run_reporting import source_version
baseline = Path(sys.argv[3])
with Path(sys.argv[1]).open('x', encoding='utf-8') as stream:
    json.dump(dict(operation='方案A CNS入口归一化单因素固定训练子集拟合对照',
        source_version=source_version(Path(sys.argv[2])),
        changed_training_factor={'cns_input_norm': ['none', 'layernorm']},
        baseline_summary_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
        baseline_modified=False, full_fold_training=False, automatic_fold_training=False,
        planned_max_private_steps=300, stages={}), stream, ensure_ascii=False, indent=2)
PY
n1_stage() {
    python - "$N1_RESULT/config.json" "$1" "$2" "$3" <<'PY'
import json,sys
from pathlib import Path
p = Path(sys.argv[1]); value = json.loads(p.read_text())
value['stages'][sys.argv[2]] = dict(status=sys.argv[3], exit_code=int(sys.argv[4]))
t = p.with_suffix('.tmp'); t.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8'); t.replace(p)
PY
}
N1_RUN_EXIT=0
(
    cd "$N1_PROJECT"
    bash scripts/phase3_run_logged.sh "$N1_RESULT/interface_tests.log" \
        env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u - "$N1_PROJECT/tests" <<'PY'
import sys,unittest
suite = unittest.TestSuite()
for pattern in ('test_scheme_a.py', 'test_cns_norm_pairing.py'):
    suite.addTests(unittest.TestLoader().discover(sys.argv[1],pattern=pattern))
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
PY
) || N1_RUN_EXIT=$?
n1_stage tests "$([ "$N1_RUN_EXIT" -eq 0 ] && echo passed || echo failed)" "$N1_RUN_EXIT"
if [ "$N1_RUN_EXIT" -eq 0 ]; then
    # 和原基线逐项相同；唯一训练因素cns_input_norm=layernorm。epochs20仅保留原接口元数据，不执行20轮。
    bash "$N1_SCRIPT_DIR/phase3_run_logged.sh" "$N1_RESULT/train.log" \
        python -u "$N1_SCRIPT_DIR/scheme_a_fit_check.py" \
        --dataset DEAP --label arousal --root_dir "$N1_ROOT" \
        --output_dir "$N1_OUTPUT" --run_name "$N1_NAME" \
        --seed 42 --device cuda:0 --held_out_subjects 2 --epochs 20 \
        --lr 0.00002 --weight_decay 0 --batch_size 400 --eval_batch_size 400 --workers 0 \
        --sampling_mode window --fd_mode full --no_balance --cns_input_norm layernorm --mediator_dim 64 --hidden_dim 128 \
        --lif_neurons 256 --dropout 0.3 --ref_dim 0 --time_steps 4 --logvar_min -6 --logvar_max 2 \
        --S_train 4 --K_train 16 --S_eval 16 --K_eval 64 --N_ref 2048 --reference_chunk_size 4 \
        --classification_objective integrated_observational --lambda_obs 0 --factual_ce_mode all_samples \
        --beta 0 --beta_warmup_epochs 10 --fire_weight 0.028 \
        --diagnostic_level full --log_detail_interval 5 --loss_gradient_interval 5 \
        --max_train_batches 0 --max_test_batches 0 \
        --fit_max_steps 300 --fit_check_every 50 \
        --fit_norm_baseline_summary "$N1_BASELINE" --fit_save_checkpoint --fit_gate_metric acc || N1_RUN_EXIT=$?
    case "$N1_RUN_EXIT" in 0) N1_STATUS=passed ;; 3) N1_STATUS=budget_not_passed ;; *) N1_STATUS=failed ;; esac
    n1_stage private_fit "$N1_STATUS" "$N1_RUN_EXIT"
fi
if [ -f "$N1_RESULT/probe_summary.json" ]; then
    N1_SUMMARY_EXIT=0
    bash "$N1_SCRIPT_DIR/phase3_run_logged.sh" "$N1_RESULT/summary.log" \
        python -u "$N1_SCRIPT_DIR/summarize_scheme_a_fit.py" "$N1_RESULT" || N1_SUMMARY_EXIT=$?
    n1_stage summary "$([ "$N1_SUMMARY_EXIT" -eq 0 ] && echo completed || echo failed)" "$N1_SUMMARY_EXIT"
    N1_COMPARE_EXIT=0
    bash "$N1_SCRIPT_DIR/phase3_run_logged.sh" "$N1_RESULT/comparison.log" \
        python -u "$N1_SCRIPT_DIR/cns_norm_pairing.py" \
        --baseline "$N1_BASELINE" --result "$N1_RESULT/probe_summary.json" \
        --output "$N1_RESULT/run_summary.json" || N1_COMPARE_EXIT=$?
    n1_stage comparison "$([ "$N1_COMPARE_EXIT" -eq 0 ] && echo completed || echo failed)" "$N1_COMPARE_EXIT"
    if [ "$N1_SUMMARY_EXIT" -ne 0 ] || [ "$N1_COMPARE_EXIT" -ne 0 ]; then N1_RUN_EXIT=1; fi
fi
echo "对照步骤退出码=$N1_RUN_EXIT；0=通过拟合，3=预算内未通过，其他=错误；均不自动运行正式fold。"
if [ -z "${GIT_SSH_COMMAND:-}" ]; then
    export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
fi
N1_UPLOAD_ARGS=("$N1_RESULT" --git_timeout 600 --max_file_mib 16 --max_total_mib 64)
for N1_LOG in interface_tests.log summary.log comparison.log; do
    if [ -f "$N1_RESULT/$N1_LOG" ]; then N1_UPLOAD_ARGS+=(--log "$N1_RESULT/$N1_LOG"); fi
done
bash "$N1_SCRIPT_DIR/upload_run_diagnostics.sh" "${N1_UPLOAD_ARGS[@]}" --dry-run
N1_UPLOAD_ID="$(date +%Y%m%d_%H%M%S)_$$"
N1_UPLOAD_LOG="$N1_OUTPUT/upload_logs/${N1_NAME}_${N1_UPLOAD_ID}.log"
N1_BUNDLE="$N1_OUTPUT/upload_bundles/${N1_NAME}_${N1_UPLOAD_ID}"
test ! -e "$N1_BUNDLE"
N1_UPLOAD_EXIT=0
bash "$N1_SCRIPT_DIR/phase3_run_logged.sh" "$N1_UPLOAD_LOG" \
    bash "$N1_SCRIPT_DIR/upload_run_diagnostics.sh" "${N1_UPLOAD_ARGS[@]}" \
    --bundle_dir "$N1_BUNDLE" || N1_UPLOAD_EXIT=$?
cat "$N1_UPLOAD_LOG.exit"
if [ -f "$N1_BUNDLE/upload_state.json" ]; then cat "$N1_BUNDLE/upload_state.json"; fi
if [ "$N1_UPLOAD_EXIT" -ne 0 ]; then
    echo '上传未确认成功；回传终端摘要/收据，不重跑拟合。' >&2
    exit "$N1_UPLOAD_EXIT"
fi
echo '日志、实际步数、配对比较和退出收据已上传；私有权重仅留服务器。回传SHA和GitHub链接即可。'
exit "$N1_RUN_EXIT"
