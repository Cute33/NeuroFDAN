# Phase6：测试日志被误判为旧评价产物的修复与重试

## 已核实的失败原因

服务器上传提交7aa00421e671d5f4c133bc7fce8abe993f9746c7，目录runs/2026-11-11_175511_unknown_unknown_run_bd476d24。载荷逐文件长度和SHA256全部一致，无跳过项；仅含评价失败日志、4项测试通过日志及上传器生成的摘要/报告，没有四格评价结果。

旧91f1564脚本只允许输出目录预先包含train.log及train.log.exit。但是首次交付命令先执行测试，必然在同一目录生成component_tests.log和component_tests.log.exit；这两个文件被误判为已存在的评价产物，main在读取原实验配置和checkpoint之前抛出FileExistsError。

这是工具与命令配合的工程错误，不是新发现的模型塌缩或checkpoint不兼容。原4项测试通过只证明对应数学和小型接口，不能证明checkpoint已加载。训练/评价日志中的程序退出码1与上传程序退出码0对应不同操作，不能混为一谈。

本次修改：

- scripts/evaluate_phase6_components.py：提取prepare_output_directory；精确允许上述4个普通日志文件，保留内容；仍拒绝config.json、probe_summary.json、其他文件、目录和符号链接，不删除旧文件。
- tests/test_phase6_component_evaluation.py：新增4项目录回归测试，覆盖先测试后评价的4文件组合、旧评价保留拒绝、未知文件/目录拒绝、日志符号链接拒绝。连同原4项共8项，待服务器执行。
- 文档：标注旧命令问题并提供独立002目录重试；保留001失败证据，不修改历史结果。

模型、损失、SCM、积分公式、S/K、原checkpoint和真实数据流程均不改变。云端仅静态AST、文档Shell语法及Git差异检查，不运行CLI、模型、单元测试或训练。新增8项测试及完整评价运行状态以服务器新日志为准。

## 服务器：只重试评价，不重新训练

继续使用当前dslenv，不改主项目分支、不更新环境。首次创建独立修复工具worktree；如果目录已存在，仅在其源码与本次获取的提交完全一致且工作区干净时复用。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
git status --short
git worktree list
git fetch origin fix/neurofdan-phase6-component-output-guard
COMPONENT_FIX_SHA="$(git rev-parse FETCH_HEAD)"
COMPONENT_TOOLS=/home/fss/NeuroFDAN-phase6-component-fix-tools
if [ ! -e "$COMPONENT_TOOLS" ]; then
    git worktree add --detach "$COMPONENT_TOOLS" "$COMPONENT_FIX_SHA"
fi
test "$(git -C "$COMPONENT_TOOLS" rev-parse HEAD)" = "$COMPONENT_FIX_SHA"
test -z "$(git -C "$COMPONENT_TOOLS" status --porcelain)"
git -C "$COMPONENT_TOOLS" log -1 --oneline
COMPONENT_SOURCE=/home/fss/NeuroFDAN/outputs/phase6/synthetic/paired_tanh_normal_20ep_001
test -f "$COMPONENT_SOURCE/config.json"
test -f "$COMPONENT_SOURCE/probe_summary.json"
test -f "$COMPONENT_SOURCE/scm_parameters.npz"
test -f "$COMPONENT_SOURCE/models/gamma0.0_oracle_mediator_last.pt"
test -f "$COMPONENT_SOURCE/models/gamma1.0_oracle_mediator_last.pt"
COMPONENT_RESULT=/home/fss/NeuroFDAN/outputs/phase6/evaluation/tanh_oracle_components_002
test ! -e "$COMPONENT_RESULT"
mkdir -p "$COMPONENT_RESULT"
(
cd "$COMPONENT_TOOLS"
bash scripts/phase3_run_logged.sh "$COMPONENT_RESULT/component_tests.log" \
  python -m unittest discover -s tests -p test_phase6_component_evaluation.py -v
)
bash "$COMPONENT_TOOLS/scripts/phase3_run_logged.sh" "$COMPONENT_RESULT/train.log" \
  python -u "$COMPONENT_TOOLS/scripts/evaluate_phase6_components.py" \
  --run_dir "$COMPONENT_SOURCE" --output_dir "$COMPONENT_RESULT" \
  --seed 42 --device cpu --workers 0 --diagnostic_level full --S 256 --K 128 --repeats 4
cat "$COMPONENT_RESULT/component_tests.log.exit"
cat "$COMPONENT_RESULT/train.log.exit"
python - <<'PY'
import json
from pathlib import Path
p=Path('/home/fss/NeuroFDAN/outputs/phase6/evaluation/tanh_oracle_components_002/probe_summary.json')
r=json.loads(p.read_text())
assert r['completed'] and r['source_files_unchanged']
assert r['training_executed'] is False and r['real_data_training'] is False
assert [g['gamma'] for g in r['gammas']]==[0.0,1.0]
print('评价完成、源文件未变、本轮未训练；源码：',r['source_version']['commit'])
for g in r['gammas']:
    assert g['state_unchanged'] and g['parameter_gradients_absent']
    for cell, values in g['cells'].items():
        print('gamma=',g['gamma'],'组合=',cell,'对直接真值误差=',values['against_direct'])
PY
)
```

本轮无需mainNeuroFDAN.py，无真实数据dataset/label/fold、epochs或训练batch_size参数。原20轮模型、16点真值、训练总体和两个gamma保持不变，仅执行原先计划的S256/K128、4次MC只读评价。若失败，set -e停止后续步骤；保留错误日志，使用下方上传命令回传，不绕过严格加载或另启训练。

新输出：outputs/phase6/evaluation/tanh_oracle_components_002/{component_tests.log,component_tests.log.exit,train.log,train.log.exit,config.json,probe_summary.json,run_report.txt}。失败可能只有日志或部分摘要，不能称为评价完成。

## 上传：预览

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
UPLOAD_TOOLS=/home/fss/NeuroFDAN-upload-timeout-tools
COMPONENT_RESULT=/home/fss/NeuroFDAN/outputs/phase6/evaluation/tanh_oracle_components_002
test "$(git -C "$UPLOAD_TOOLS" rev-parse HEAD)" = 6681381298b74e9b69dc0df0594d5d9c7274acbc
test -z "$(git -C "$UPLOAD_TOOLS" status --porcelain)"
bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" "$COMPONENT_RESULT" \
  --log "$COMPONENT_RESULT/train.log" --log "$COMPONENT_RESULT/component_tests.log" \
  --git_timeout 600 --dry-run
)
```

## 上传：正式执行并保留收据

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
UPLOAD_TOOLS=/home/fss/NeuroFDAN-upload-timeout-tools
COMPONENT_RESULT=/home/fss/NeuroFDAN/outputs/phase6/evaluation/tanh_oracle_components_002
test "$(git -C "$UPLOAD_TOOLS" rev-parse HEAD)" = 6681381298b74e9b69dc0df0594d5d9c7274acbc
test -z "$(git -C "$UPLOAD_TOOLS" status --porcelain)"
COMPONENT_UPLOAD_ID="$(date +%Y%m%d_%H%M%S)_$$"
COMPONENT_UPLOAD_LOG="/home/fss/NeuroFDAN/outputs/phase6/upload_logs/components002_${COMPONENT_UPLOAD_ID}.log"
COMPONENT_BUNDLE="/home/fss/NeuroFDAN/outputs/phase6/upload_bundles/components002_${COMPONENT_UPLOAD_ID}"
mkdir -p outputs/phase6/upload_logs outputs/phase6/upload_bundles
bash "$UPLOAD_TOOLS/scripts/phase3_run_logged.sh" "$COMPONENT_UPLOAD_LOG" \
  bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" "$COMPONENT_RESULT" \
  --log "$COMPONENT_RESULT/train.log" --log "$COMPONENT_RESULT/component_tests.log" \
  --git_timeout 600 --bundle_dir "$COMPONENT_BUNDLE"
cat "$COMPONENT_UPLOAD_LOG.exit"
cat "$COMPONENT_BUNDLE/upload_state.json"
)
```

回传新的SHA及GitHub链接。上传器仍用已确认的6681381，不上传pt/npz。上传操作成功不代表评价成功，需同时检查train.log、摘要completed和源文件未改标志。不要删除或覆盖001目录和原合成模型。

## 下一步的科学判断

本次报错未产生新的机制证据，因此不改变前一轮结论：当前输入保真候选暂不迁移，尚不启动正式LOSO。只完成此次评价即可，不重复20轮、100轮或环境审计。

先用TT检查经验参考和数值精度，再用TL/LT/LL定位条件结果与中介学习的误差；随后依据结果确定最小方法改进。对潜在中介的一般识别边界仍需明确额外假设，不能由目录修复或小测试通过来消除。
