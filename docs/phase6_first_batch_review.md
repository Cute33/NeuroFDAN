# Phase6第一批结果复核与优化器启动诊断

## 1. 三份上传不等于三项实验都成功

固定证据：DEAP=dab9df24cf3e8dcc3c6100f5019599694870bb29，目录161029_unknown_unknown_run_78ba2733；HCI=2a315f45e63a23cf0924a6d4ed4524d1de845e4d，目录161200_unknown_unknown_run_059c9793；smoke=6d366901abba6a8131426e2cf2a291c076a1df6f，目录161328_unknown_unknown_run_b06a417a。完整路径均以runs/2026-11-11_开头。

本轮直接读取固定Git对象，全部载荷长度/SHA256一致，无跳过项。三份都带相同的6项针对性测试通过日志（一次测试日志附到三份快照，不能计为18项独立测试）。DEAP/HCI completed=True，模型与原文件保护通过，无优化器/训练。smoke completed=False，conditions为空；不是合成学习已完成。

上传器的python_exit=0/tee_exit=0只证明上传成功，不证明被上传实验成功。旧smoke在Adam构造之前已完成生成器与两种真值计算；随后首个oracle_mediator的优化器初始化失败，尚未进入epoch循环，训练step为0。保存的training_executed=None是中断状态，结合栈可定位到初始化，不改写原文件。

错误链为torch.optim.Adam→Optimizer.add_param_group→torch._dynamo→trace_rules._recompile_re→re.compile，最终re.error: bad escape / at position 590。虽然未使用torch.compile，Adam内部仍有延迟导入链。先前纯torch导入6次成功不能验证这一链；本次6项测试也没有构造Adam，这是检查覆盖边界。不能据此宣称新损失或SCM错误，也不能确定依赖版本损坏/导入顺序/路径遮蔽的具体根因。

低精度4点直接/前门真值MAE约0.009641；直接SE约0.004，oracle近似SE约0.0165，未触发工程容差。它只是smoke真值检查，不是高精度验证或任何学习结果。

## 2. 梯度证据：任务路径存在，HCI发放率项值得重点检查

下面均为原epoch20、16个训练查询的train模式（BN冻结）局部梯度L2，不是历史全训练统计、实际Adam更新或因果效应。

| 参数组 | DEAP事实CE梯度 | HCI事实CE梯度 |
|---|---:|---:|
| CNS中介 | 0.000500834 | 0.002050350 |
| ANS中介（不含电流层） | 0.016565046 | 0.026916552 |
| CNS→ANS电流层 | 0.008289758 | 0.012594930 |
| OutcomeHead | 0.422209891 | 0.602898911 |

两骨干及训练图也有CE梯度，因此不支持“整个任务监督断开”。不同组维数不同，不能从这张表换算贡献百分比。

DEAP中CNS的加权KLc/KLa梯度分别0.000134490/0.000110347，总梯度与CE夹角cos约0.979。局部证据不支持“此时KL完全压倒CE”；也不能排除更早历史或分布不同批次的影响。DEAP fire权重0，所以其梯度0是设置预期。

HCI中CNS fire梯度0.017946682，为CE的8.75299倍；训练总梯度与CE的cos约0.05753。eval敏感度下fire/CE约11.56049，但真实优化发生在train模式。fire损失标量仅0.001446，不代表参数梯度小。fire与CE cos约-0.05712，接近正交，不能描述成已证明强烈反向对抗；它在该组局部更新方向中占较大份量。先记录，不删除或重设fire_weight。

前门平均P1对共享参数也有局部导数；它不是损失，不是干预标签监督，与CE的夹角没有“因果正确性”含义。eval的dag_adj梯度缺失符合硬argmax不可导，不能当成训练detach错误。

## 3. 模式差异与输入

原输入不是全零或常量：DEAP CNS/ANS元素std约0.9802/0.8493，HCI约1.0387/1.0673；16个查询中每个输入位置都有变化。这不证明所有数据预处理正确，也不证明语义信息充分。

| CNS各维跨输入std的平均（不是旧RMS） | train | eval | eval/train |
|---|---:|---:|---:|
| DEAP posterior mu | 0.000487745 | 0.000646405 | 1.3253 |
| HCI posterior mu | 0.000646680 | 0.020340729 | 31.4541 |

同权重同输入，BN在两模式都冻结；CNS后验参数在采样前计算，CNS参数网络无dropout，Head的dropout不会反向改变此处mu数值。沿代码追踪，该模式敏感性应定位到PhysiologicalBackbone.forward的随机hard Gumbel训练图与固定eval图分支。它是当前样本上的明确模式差异，不是整个训练塌缩的唯一根因证明；也不据此自动改成固定训练图或重写骨干。ANS均值日志另受有限Mc样本影响，不能同样直接归因。

## 4. 本轮决定：暂停20轮与真实训练，补独立Adam启动检查

新增scripts/diagnose_optimizer_startup.py。标准库父进程调度全新子进程，每种2次：普通最小导入、-I最小导入、普通项目模块导入、-I项目模块导入、Dynamo先加载再导入项目。项目模式显式指向原15b7976工具；只导入定义，不运行其main、不构造NeuroFDAN、不读数据/权重。

每次仅创建两个CPU标量参数的Adam并做一次微型step，覆盖原6项测试遗漏的启动链。记录失败阶段、Python/包版本、re/parser源码指纹、re.compile来源、正则字符集合及异常pattern指纹/位置的两个字符码；不打印完整pattern、环境变量或凭证。不能关闭正则校验、猴补re、替换Adam、盲目升级torch，或把TORCHDYNAMO_DISABLE当作已验证可跳过此导入的修复。

父进程退出0仅表示收集完成，必须检查all_optimizer_checks_ok。-I仍可能执行全局site定制，诊断自身的导入也可能影响偶发错误；全部成功不证明环境已永久修好。preload仅作对照，不改真实训练启动顺序。失败时仅上传诊断；全部通过时允许在原15b7976上以新目录smoke_002重试一次原1轮配置，不进入20轮。

## 5. 服务器：获取工具与运行启动诊断

当前dslenv继续使用，无需安装依赖。旧梯度和失败smoke_001完整保留。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
git status --short
git worktree list
timeout 600 git fetch origin \
  refs/heads/fix/neurofdan-phase6-optimizer-startup:refs/remotes/origin/fix/neurofdan-phase6-optimizer-startup
OPT_TOOLS=/home/fss/NeuroFDAN-optimizer-audit-tools
if [ ! -e "$OPT_TOOLS" ]; then
    git worktree add --detach "$OPT_TOOLS" origin/fix/neurofdan-phase6-optimizer-startup
fi
test "$(git -C "$OPT_TOOLS" rev-parse HEAD)" = "$(git rev-parse origin/fix/neurofdan-phase6-optimizer-startup)"
test -z "$(git -C "$OPT_TOOLS" status --porcelain)"
test "$(git -C /home/fss/NeuroFDAN-phase6-tools rev-parse HEAD)" = 15b79769535027f4dfb7de0cb5ab1c11252dec49
test -z "$(git -C /home/fss/NeuroFDAN-phase6-tools status --porcelain)"
OPT_DEST=/home/fss/NeuroFDAN/outputs/phase6/environment/optimizer_startup_001
mkdir -p "$(dirname "$OPT_DEST")"
mkdir "$OPT_DEST"
bash "$OPT_TOOLS/scripts/phase3_run_logged.sh" "$OPT_DEST/train.log" \
  python -u "$OPT_TOOLS/scripts/diagnose_optimizer_startup.py" \
  --project_dir /home/fss/NeuroFDAN-phase6-tools \
  --output_dir "$OPT_DEST" --repeats 2 --timeout 60
cat "$OPT_DEST/train.log.exit"
)
```

## 6. 仅在全部Adam检查通过时，原版本重试一次smoke

以下脚本首先检查all_optimizer_checks_ok；不通过会立即停止，不能手动删掉这项检查。这个重试仍只是接口运行，不是正式学习结果验收。暂时不运行旧指南第4节的20轮。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
python - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('/home/fss/NeuroFDAN/outputs/phase6/environment/optimizer_startup_001/probe_summary.json').read_text())
if not r['all_optimizer_checks_ok']:
    raise SystemExit('优化器检查未全部通过；只上传诊断，停止模型重试。')
PY
PHASE6_TOOLS=/home/fss/NeuroFDAN-phase6-tools
test "$(git -C "$PHASE6_TOOLS" rev-parse HEAD)" = 15b79769535027f4dfb7de0cb5ab1c11252dec49
test -z "$(git -C "$PHASE6_TOOLS" status --porcelain)"
OPT_DEST=/home/fss/NeuroFDAN/outputs/phase6/synthetic/paired_normal_smoke_002
mkdir "$OPT_DEST"
bash "$PHASE6_TOOLS/scripts/phase3_run_logged.sh" "$OPT_DEST/train.log" \
  python -u "$PHASE6_TOOLS/phase6_synthetic_experiment.py" \
  --output_dir /home/fss/NeuroFDAN/outputs/phase6/synthetic --run_name paired_normal_smoke_002 \
  --seed 42 --device cpu --epochs 1 --batch_size 128 --workers 0 --diagnostic_level full \
  --train_size 256 --test_size 64 --points 4 --gammas 1 \
  --lr 0.001 --weight_decay 0 --fidelity_weight 1 --fire_weight 0 \
  --mediator_family normal --mc_repeats 2 --direct_n 4096 --oracle_s 128 --oracle_k 128 --smoke
cat "$OPT_DEST/train.log.exit"
)
```

## 7. 上传预览与正式上传

仅收集新环境诊断及实际生成的smoke_002，不重传三个已成功上传目录。既有6681381上传工具已验证；每次正式上传保留唯一日志与持久upload_state.json。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
UPLOAD_TOOLS=/home/fss/NeuroFDAN-upload-timeout-tools
test "$(git -C "$UPLOAD_TOOLS" rev-parse HEAD)" = 6681381298b74e9b69dc0df0594d5d9c7274acbc
for OPT_REL in environment/optimizer_startup_001 synthetic/paired_normal_smoke_002
do
    OPT_RESULT="/home/fss/NeuroFDAN/outputs/phase6/$OPT_REL"
    if [ ! -d "$OPT_RESULT" ]; then echo "尚未执行：$OPT_RESULT"; continue; fi
    bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" \
      "$OPT_RESULT" --log "$OPT_RESULT/train.log" --git_timeout 600 --dry-run
done
)
```

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
UPLOAD_TOOLS=/home/fss/NeuroFDAN-upload-timeout-tools
test "$(git -C "$UPLOAD_TOOLS" rev-parse HEAD)" = 6681381298b74e9b69dc0df0594d5d9c7274acbc
OPT_UPLOAD_ID="$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p outputs/phase6/upload_logs outputs/phase6/upload_bundles
for OPT_REL in environment/optimizer_startup_001 synthetic/paired_normal_smoke_002
do
    OPT_RESULT="/home/fss/NeuroFDAN/outputs/phase6/$OPT_REL"
    if [ ! -d "$OPT_RESULT" ]; then echo "尚未执行：$OPT_RESULT"; continue; fi
    OPT_NAME="$(basename "$OPT_RESULT")"
    OPT_LOG="/home/fss/NeuroFDAN/outputs/phase6/upload_logs/${OPT_NAME}_${OPT_UPLOAD_ID}.log"
    OPT_BUNDLE="/home/fss/NeuroFDAN/outputs/phase6/upload_bundles/${OPT_NAME}_${OPT_UPLOAD_ID}"
    bash "$UPLOAD_TOOLS/scripts/phase3_run_logged.sh" "$OPT_LOG" \
      bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" \
      "$OPT_RESULT" --log "$OPT_RESULT/train.log" --git_timeout 600 --bundle_dir "$OPT_BUNDLE"
    cat "$OPT_LOG.exit"
    cat "$OPT_BUNDLE/upload_state.json"
done
)
```

回传1或2组SHA/链接；诊断失败也上传，后续按失败阶段和模式差异处理，不索取密码/Token/私钥。保留新旧失败目录，不覆盖任何checkpoint。本轮无需运行mainNeuroFDAN.py，不执行真实训练/LOSO。

## 8. 云端检查与Git边界

本轮只做固定快照完整性/数值复核、源代码追踪、Python3.10 AST（含子进程WORKER字符串）、文档Shell静态语法和diff检查。未运行新增CLI、子进程、Adam、模型、单元测试或训练；新诊断尚待服务器运行。只新增诊断与文档，不修改任何模型、SCM、损失、优化器实现、数据、Phase6实验代码或已完成记录，不创建PR、不合并、不修改main。
