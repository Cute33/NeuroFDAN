#!/usr/bin/env bash
# 方案A：测试→固定自然批次完整反传/拟合验收→通过才20轮→一次自动上传。
set -euo pipefail
if [ "$#" -ne 3 ]; then
    echo '用法：bash scripts/run_scheme_a_and_upload.sh 数据根绝对目录 输出根绝对目录 唯一运行名' >&2
    exit 2
fi
SCHEME_ROOT=$1; SCHEME_OUTPUT=$2; SCHEME_NAME=$3
case "$SCHEME_ROOT:$SCHEME_OUTPUT" in /*:/*) ;; *) exit 2 ;; esac
case "$SCHEME_NAME" in ''|*/*|.|..) exit 2 ;; esac
SCHEME_SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SCHEME_PROJECT=$(dirname -- "$SCHEME_SCRIPT_DIR")
SCHEME_RESULT="$SCHEME_OUTPUT/DEAP/arousal/seed42/$SCHEME_NAME"
test ! -e "$SCHEME_RESULT"
mkdir -p "$SCHEME_RESULT"
python - "$SCHEME_RESULT/config.json" "$SCHEME_PROJECT" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[2])
from run_reporting import source_version
Path(sys.argv[1]).write_text(json.dumps(dict(operation='方案A分阶段工作流',
    source_version=source_version(Path(sys.argv[2])),
    classification_objective='integrated_observational', observation_labels=True,
    oracle_supervision=False, stages={}, workflow_state='started',
    fit_weights_transferred_to_fold=False),ensure_ascii=False,indent=2),encoding='utf-8')
PY
scheme_stage() {
    python - "$SCHEME_RESULT/config.json" "$1" "$2" "$3" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);v=json.loads(p.read_text())
v['stages'][sys.argv[2]]=dict(status=sys.argv[3],exit_code=int(sys.argv[4]))
v['workflow_state']=sys.argv[2]+':'+sys.argv[3]
t=p.with_suffix('.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');t.replace(p)
PY
}
SCHEME_ARGS=(--dataset DEAP --label arousal --root_dir "$SCHEME_ROOT"
  --output_dir "$SCHEME_OUTPUT" --run_name "$SCHEME_NAME"
  --seed 42 --device "${NEURO_DEVICE:-cuda:0}" --held_out_subjects 2 --epochs 20
  --lr 0.00002 --weight_decay 0 --batch_size 400 --eval_batch_size 400 --workers 0
  --sampling_mode window --fd_mode full --no_balance --mediator_dim 64 --hidden_dim 128
  --lif_neurons 256 --dropout 0.3 --ref_dim 0 --time_steps 4 --logvar_min -6 --logvar_max 2
  --S_train 4 --K_train 16 --S_eval 16 --K_eval 64 --N_ref 2048 --reference_chunk_size 4
  --classification_objective integrated_observational --lambda_obs 0 --factual_ce_mode all_samples
  --beta 0.001 --beta_warmup_epochs 10 --fire_weight 0.028
  --diagnostic_level full --log_detail_interval 5 --loss_gradient_interval 5
  --max_train_batches 0 --max_test_batches 0)
SCHEME_RUN_EXIT=0
scheme_stage tests running 0
(
    cd "$SCHEME_PROJECT"
    bash scripts/phase3_run_logged.sh "$SCHEME_RESULT/interface_tests.log" \
      env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
) || SCHEME_RUN_EXIT=$?
scheme_stage tests "$([ "$SCHEME_RUN_EXIT" -eq 0 ] && echo passed || echo failed)" "$SCHEME_RUN_EXIT"
if [ "$SCHEME_RUN_EXIT" -eq 0 ]; then
    scheme_stage fit_check running 0
    bash "$SCHEME_SCRIPT_DIR/phase3_run_logged.sh" "$SCHEME_RESULT/fit_check.log" \
      python -u "$SCHEME_SCRIPT_DIR/scheme_a_fit_check.py" "${SCHEME_ARGS[@]}" \
      --fit_max_steps 300 --fit_check_every 50 || SCHEME_RUN_EXIT=$?
    case "$SCHEME_RUN_EXIT" in 0) SCHEME_FIT_STATUS=passed ;; 3) SCHEME_FIT_STATUS=budget_not_passed ;; *) SCHEME_FIT_STATUS=failed ;; esac
    scheme_stage fit_check "$SCHEME_FIT_STATUS" "$SCHEME_RUN_EXIT"
fi
if [ "$SCHEME_RUN_EXIT" -eq 0 ]; then
    scheme_stage fold20 running 0
    # 新进程从seed42初始化，绝不沿用私有拟合权重或读取旧checkpoint。
    bash "$SCHEME_SCRIPT_DIR/phase3_run_logged.sh" "$SCHEME_RESULT/train.log" \
      python -u "$SCHEME_PROJECT/mainNeuroFDAN.py" "${SCHEME_ARGS[@]}" || SCHEME_RUN_EXIT=$?
    scheme_stage fold20 "$([ "$SCHEME_RUN_EXIT" -eq 0 ] && echo completed || echo failed)" "$SCHEME_RUN_EXIT"
else
    scheme_stage fold20 not_started "$SCHEME_RUN_EXIT"
fi
echo "方案A工作流退出码=$SCHEME_RUN_EXIT；自动上传现有日志、拟合结论及实际完成轮数。"
if [ -z "${GIT_SSH_COMMAND:-}" ]; then
    export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
fi
SCHEME_UPLOAD_ARGS=("$SCHEME_RESULT" --log "$SCHEME_RESULT/interface_tests.log"
                   --git_timeout 600 --max_file_mib 16 --max_total_mib 64)
if [ -f "$SCHEME_RESULT/fit_check.log" ]; then SCHEME_UPLOAD_ARGS+=(--log "$SCHEME_RESULT/fit_check.log"); fi
bash "$SCHEME_SCRIPT_DIR/upload_run_diagnostics.sh" "${SCHEME_UPLOAD_ARGS[@]}" --dry-run
SCHEME_UPLOAD_ID="$(date +%Y%m%d_%H%M%S)_$$"
SCHEME_UPLOAD_LOG="$SCHEME_OUTPUT/upload_logs/${SCHEME_NAME}_${SCHEME_UPLOAD_ID}.log"
SCHEME_BUNDLE="$SCHEME_OUTPUT/upload_bundles/${SCHEME_NAME}_${SCHEME_UPLOAD_ID}"
test ! -e "$SCHEME_BUNDLE"
bash "$SCHEME_SCRIPT_DIR/phase3_run_logged.sh" "$SCHEME_UPLOAD_LOG" \
  bash "$SCHEME_SCRIPT_DIR/upload_run_diagnostics.sh" "${SCHEME_UPLOAD_ARGS[@]}" --bundle_dir "$SCHEME_BUNDLE"
cat "$SCHEME_BUNDLE/upload_state.json"
echo '仅回传本次SHA及GitHub链接；无需手工摘抄步数、轮数或manifest。'
exit "$SCHEME_RUN_EXIT"
