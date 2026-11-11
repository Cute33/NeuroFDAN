# Phase6：接口通过后进入配对学习验证与正式实验的准入条件

## 1. 本次固定结果复核

- 优化器：d26af602af001ed8e4a8fecb4bdd9f388e029b0b，runs/2026-11-11_164643_unknown_unknown_run_c419e634。
- 合成smoke_002：1e1ebfb29c722d9df1ceafaf290a42cd47ec611f，runs/2026-11-11_164802_unknown_unknown_run_131a2ac3。

两份manifest的全部载荷长度/SHA256已重算一致，无跳过项。优化器检查5种模式各2次，10/10通过；普通/隔离/项目导入/预加载对照均完成一次两元素CPU参数的Adam step。目标项目为15b79769535027f4dfb7de0cb5ab1c11252dec49。包版本记录torch2.5.1+cu121、numpy2.2.6、scipy1.15.2、sympy1.13.1；不据此要求升级或降级。旧错误未复现，不能宣称根因已查明或环境永久修复。

smoke_002的completed=True、smoke=True，三组各完成1轮及独立评价。三组初始模型指纹一致，X/Y数据指纹一致，类别数[113,143]；非oracle两组训练字段仅x_c/x_a/y。参考容量配置2048、实际256且shortfall=True，符合smoke训练样本数，不是参考库丢失。

| 条件 | 干预MAE，对直接MC真值 |
|---|---:|
| 真实中介监督参考 | 0.03031936 |
| 仅观察监督 | 0.03487891 |
| 观察+输入保真候选 | 0.03307858 |
| 训练类别比例常数 | 0.03229819 |

输入保真比观察baseline低约0.00180，但未优于常数。只有4个点、1轮/2个训练batch；直接真值SE约0.004、积分重复也有波动，不足以判断改善。直接/Oracle前门MAE约0.009641，未触发预设工程容差，仍是低精度接口检查。三组do对观察测试标签均全预测1，观察测试样本仅64，不能作为方法分类性能的结论，也不能据此判定20轮结果必然失败。

## 2. 当前可进入的阶段

接口关已过，下一步是预定义配对合成学习验证，不重复两个真实梯度探针、10次优化器检查或旧smoke。使用原15b7976代码，不修正模型、不调整loss权重/参考总体。

两个独立运行：normal和tanh_normal；每个运行gamma=0/1各比较三种监督，总计12个小模型。每组训练2048、测试512、16干预点、20轮、seed42、同初始化及X/Y、最后轮评价、MC重复3次。normal显式存在相对tanh生成器的分布错设，tanh_normal用于控制该因素；不得择高ACC的族冒充真实模型。主真实模型仍normal不变。

这是有科学评价目标的学习实验，不再只是接口探针；它仍不是DEAP/HCI正式分类实验。单次短smoke实测拟合约0.04–0.05秒/组，扩大后只能作粗略工作量参考，按分钟级CPU任务预留，受共享资源和真值积分影响，实际以日志计时为准，不承诺固定结束时刻。

## 3. 到什么程度可以正式运行真实数据

| 阶段 | 准入标准 | 当前 |
|---|---|---|
| 工程运行 | 权限测试、优化器启动、三组完整smoke、状态保护 | 已通过本轮小规模检查 |
| 合成学习依据 | 真值数值一致、三组配对完整；检查干预误差/常数基准/MC稳定性与条件学习，而非只看观察ACC | 下一步执行 |
| 真实迁移设计 | 明确保真目标与测量/生成假设、优化/参考总体、真实接口；一次只改一个因素 | 尚未实现新真实训练方案 |
| DEAP/HCI单折受控验证 | baseline与候选独立输出，同seed/fold/预处理/参考及test-selected规则；无异常、完整诊断、重采样稳定性 | 尚未执行 |
| 正式LOSO | 方法/权重/主输出/协议冻结，报告假设与选模偏差，不按结果换路径或挑fold | 尚未放行 |

不设“ACC达到60%/70%就因果正确”的门槛，也不要求真实预测必须恰好两类均衡。主要判定是估计目标是否明确、实现和有限机制证据是否相符、实验协议是否冻结。正式分类评估可以在明确且有限的模型假设下开展；不能把它作为真实生理SCM已被证明的证据。

顺利情况下，后续关键工作是这一批合成验证，加上必要的真实接口实现及一轮DEAP/HCI固定单折短程对照；不是再重复相同环境检查。不能保证跑完固定次数就自动进入LOSO。

### 合成结果的预先判读规则

本批预先以固定干预点上对直接MC真值的概率MSE为主判据（interventional_mean_probability.probability_mse）；MAE、期望NLL、校准及MC稳定性为辅助。相同点/真值下期望Brier与概率MSE仅相差共同的不可约项，不能算成两份独立改善证据。同时报告0.5和训练类别比例两种常数基准，不能事后只选择较弱的比较对象。

1. 若有中介监督参考也拟合不足、或tanh匹配控制仍不能恢复真值，先检查条件模型学习/数值设置，不直接把后两组差异解释为新方法有效。
2. 输入保真若只改善输入变化或观察ACC，不改善干预MAE、概率MSE/适当评分及稳定性，拒绝把该候选迁移为正式新方法；不无限增加alpha/轮数追分。
3. 与观察baseline及训练比例常数比较，关注gamma=0/1预定点上的方向一致性，不能只挑一个成功设置。单种子/16点结果不宣称统计显著或一般识别。
4. 若误差差异与直接/oracle或积分MC误差同量级，优先使用已保存模型提高评价精度或独立点复核，不因评价噪声重新训练。当前脚本保存最后模型和生成参数供后续复核。
5. 即使候选改善，输入保真仍没有新增独立测量信息，不能解决一般非识别反例。真实迁移前还需明确理论假设、辅助目标的生理解释和下采样目标总体。真实潜在中介不可观测的限制不能靠更多epoch消失。

HCI fire梯度与图模式敏感性是后续真实受控验证的两个独立候选因素，当前不同时改变，以免失去归因。保持probs_do为主输出、probs_obs为诊断，保持既有train/test二分与test-selected规则并披露偏差。

## 4. 服务器：直接执行两个预定义配对实验

当前dslenv继续使用。代码已经在原工具worktree，无需切换主项目分支或新建模型版本。下面检查固定源码身份、成功smoke_002和优化器结果，随后串行运行；任何失败停止，保留部分结果，不覆盖旧目录。无需运行mainNeuroFDAN.py。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
PHASE6_TOOLS=/home/fss/NeuroFDAN-phase6-tools
git status --short
git worktree list
test "$(git -C "$PHASE6_TOOLS" rev-parse HEAD)" = 15b79769535027f4dfb7de0cb5ab1c11252dec49
test -z "$(git -C "$PHASE6_TOOLS" status --porcelain)"
python - <<'PY'
import json
from pathlib import Path
root=Path('/home/fss/NeuroFDAN/outputs/phase6')
o=json.loads((root/'environment/optimizer_startup_001/probe_summary.json').read_text())
s=json.loads((root/'synthetic/paired_normal_smoke_002/probe_summary.json').read_text())
if not o['all_optimizer_checks_ok'] or not s['completed'] or not s['smoke']:
    raise SystemExit('前置检查未通过，停止，不重复旧训练。')
if s['source_version']['commit']!='15b79769535027f4dfb7de0cb5ab1c11252dec49':
    raise SystemExit('smoke源码不匹配。')
expected={'oracle_mediator','observation_only','observation_input_fidelity'}
if len(s['gammas'])!=1 or {a['condition'] for a in s['gammas'][0]['conditions']}!=expected:
    raise SystemExit('smoke三组未完整完成。')
print('前置检查通过，开始预定义配对合成；不是正式真实数据训练。')
PY
for PHASE6_FAMILY in normal tanh_normal
do
    PHASE6_RUN="paired_${PHASE6_FAMILY}_20ep_001"
    PHASE6_DEST="/home/fss/NeuroFDAN/outputs/phase6/synthetic/$PHASE6_RUN"
    if [ -e "$PHASE6_DEST" ]; then echo "已有目录，停止，不覆盖：$PHASE6_DEST"; exit 1; fi
    mkdir "$PHASE6_DEST"
    bash "$PHASE6_TOOLS/scripts/phase3_run_logged.sh" "$PHASE6_DEST/train.log" \
      python -u "$PHASE6_TOOLS/phase6_synthetic_experiment.py" \
      --output_dir /home/fss/NeuroFDAN/outputs/phase6/synthetic --run_name "$PHASE6_RUN" \
      --seed 42 --device cpu --epochs 20 --batch_size 128 --workers 0 --diagnostic_level full \
      --train_size 2048 --test_size 512 --points 16 --gammas 0 1 \
      --lr 0.001 --weight_decay 0 --fidelity_weight 1 --fire_weight 0 \
      --mediator_family "$PHASE6_FAMILY" --mc_repeats 3 \
      --direct_n 32768 --oracle_s 1024 --oracle_k 512
    cat "$PHASE6_DEST/train.log.exit"
done
)
```

## 5. 仅上传本次两个20轮目录

先预览。没有生成的第二个目录会明确跳过；失败也上传已有日志，不能仅因上传成功就称学习实验成功。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
UPLOAD_TOOLS=/home/fss/NeuroFDAN-upload-timeout-tools
test "$(git -C "$UPLOAD_TOOLS" rev-parse HEAD)" = 6681381298b74e9b69dc0df0594d5d9c7274acbc
test -z "$(git -C "$UPLOAD_TOOLS" status --porcelain)"
for PHASE6_RUN in paired_normal_20ep_001 paired_tanh_normal_20ep_001
do
    PHASE6_RESULT="/home/fss/NeuroFDAN/outputs/phase6/synthetic/$PHASE6_RUN"
    if [ ! -d "$PHASE6_RESULT" ]; then echo "尚未生成：$PHASE6_RESULT"; continue; fi
    bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" \
      "$PHASE6_RESULT" --log "$PHASE6_RESULT/train.log" --git_timeout 600 --dry-run
done
)
```

确认manifest后正式上传，保留独立日志和持久收据：

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
UPLOAD_TOOLS=/home/fss/NeuroFDAN-upload-timeout-tools
test "$(git -C "$UPLOAD_TOOLS" rev-parse HEAD)" = 6681381298b74e9b69dc0df0594d5d9c7274acbc
test -z "$(git -C "$UPLOAD_TOOLS" status --porcelain)"
PHASE6_UPLOAD_ID="$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p outputs/phase6/upload_logs outputs/phase6/upload_bundles
for PHASE6_RUN in paired_normal_20ep_001 paired_tanh_normal_20ep_001
do
    PHASE6_RESULT="/home/fss/NeuroFDAN/outputs/phase6/synthetic/$PHASE6_RUN"
    if [ ! -d "$PHASE6_RESULT" ]; then echo "尚未生成：$PHASE6_RESULT"; continue; fi
    PHASE6_LOG="/home/fss/NeuroFDAN/outputs/phase6/upload_logs/${PHASE6_RUN}_${PHASE6_UPLOAD_ID}.log"
    PHASE6_BUNDLE="/home/fss/NeuroFDAN/outputs/phase6/upload_bundles/${PHASE6_RUN}_${PHASE6_UPLOAD_ID}"
    bash "$UPLOAD_TOOLS/scripts/phase3_run_logged.sh" "$PHASE6_LOG" \
      bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" \
      "$PHASE6_RESULT" --log "$PHASE6_RESULT/train.log" --git_timeout 600 --bundle_dir "$PHASE6_BUNDLE"
    cat "$PHASE6_LOG.exit"
    cat "$PHASE6_BUNDLE/upload_state.json"
done
)
```

回传两组SHA/链接；若只有一个运行到失败，则先回传该组及错误日志。无需上传pt/npz，原工具白名单会跳过模型和生成参数文件。旧环境诊断、梯度和smoke不再重复上传。

## 6. 本轮代码边界与验证

本轮只更新文档、修正原指南第4节指向已成功的smoke_002，而非失败的smoke_001；没有修改任何Python、模型、loss、数据、采样、主输出或实验参数。无需获取新运行代码，继续使用已验证的15b7976。

已执行固定上传载荷哈希复核、配对/监督字段核对、文档Shell与内嵌Python静态语法和git diff检查。没有在云端执行优化器、模型、CLI、单元测试或训练。服务器10次检查和三组smoke成功来自实际上传日志。本轮不创建PR、不合并、不修改main。
