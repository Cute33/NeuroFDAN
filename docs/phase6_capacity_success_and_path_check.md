# Phase6：基础拟合检查通过，检查成功拟合依赖的路径

## 1. 已核验结果

服务器提交68d608765a47034d9dcdf193d9303aaefd77fa4b，runs/2026-11-11_222737_unknown_unknown_run_d3703968。全部载荷长度与SHA256一致，无跳过项。通用run_summary/run_report/config缺少不影响probe_summary和两份日志的完整性。

6项策略测试通过。completed=True、original_checkpoint_unchanged=True、weights_saved=False；两个私有模型确实更新了参数，各执行300步。来源为训练被试2、固定16个不同trial的窗口，真实计数[4,12]，无相同输入的冲突标签，没有类别配平或held-out样本。源码b7532d055bbf4dd74ff647e54a78a6dc9212907a；原自然训练模型的优化器步数为620。

| 私有模型起点 | 第300步训练态CE | 训练态四次ACC均值 | 评估态CE | 评估态四次ACC均值 |
|---|---:|---:|---:|---:|
| seed42初始化 | 0.045584 | 1.000 | 0.012343 | 1.000 |
| 自然训练epoch20权重 | 0.038442 | 1.000 | 0.012310 | 1.000 |

两种起点、两种模式末端四次单次ACC均为1，满足原预声明小批记忆阈值。最终矩阵均为[[4,0],[0,12]]，不是75%多数类基线。CNS均值跨窗口std在评估态分别由约0.000871/0.000826增加到0.382918/0.363760。CNS/ANS/电流层、骨干和图都有更新阶段梯度；完整数值见原probe_summary。

这说明基础拟合链路在该小批有效，原epoch20参数仍能恢复拟合；不能把它说成完全断梯度、不可逆塌缩或必然train/eval不一致。也不能外推到全部被试或真实因果效应：这16窗被重复用于训练，两个模型重置了Adam moment，目标使用固定epoch20等效KL，并非正式从头warmup或原训练续训。

## 2. 尚未回答的关键问题

旧探针只评分事实路径h(Ma,g(X_i))。它拟合成功可能依赖自身参考，不能仅凭ACC100%或Mc均值变化就证明Ma承载了稳定类别信息、Mc→Ma贡献有效，或h(Ma,g(X'))能推广到跨参考组合。该问题正是前门研究主线，需要在拟合成功状态下检查，而不是继续扩大LOSО。

实际骨干当前输出CNS为[B,96,128]，DEAP ANS为[B,96,12]，完整参考维度13440（HCI为13824），Ma维度64。这些不是本次改动。维度差异本身不证明数值压制或参考捷径，因此以下检查依据固定输入的实际响应，而不新增未经验证的投影或池化。

ref_dim=0时ReferenceEncoder没有额外投影参数，原梯度报告reference组active=False属于预期，不表示参考输入没有参与预测；参考路径仍可更新共享骨干和OutcomeHead。

上次明确没有保存私有权重，因此无法直接对该拟合状态追加推理。下一步按相同来源、随机种子、步数和目标重现私有拟合，只在结束后追加无梯度路径检查。不重跑任何20轮，不修改原last.pt，不增加优化步数或新损失。末端CE/四次ACC须与旧摘要重现一致；这只是端点指标复核，旧探针没有私有参数指纹，不能称逐参数bitwise复现。

## 3. 本次末端诊断

1. 原事实路径的4次单次预测继续记录。
2. 同一组Ma采样，使用自身参考，先softmax再沿S平均；与do路径用相同S，避免把单次采样与S次平均差异混为参考效应。
3. 按原checkpoint全部2048个训练参考成员重建当前私有模型特征，执行原S_eval16/K_eval64积分，4次MC重复，排除查询自身。来源集合不得改变，不使用held-out输入/标签。
4. 固定Ma轮换自身参考、固定自身参考轮换Ma，遍历15种循环移位；另对固定经验参考作一次Ma循环移位。记录概率L1变化，并对供体与当前标签不同的配对报告供体一致率/当前标签ACC。后者只用于读懂功能依赖，不加入损失。
5. 固定Xa、Head、参考和同一ANS标准噪声，循环移位一次Mc，检查ANS后验参数和输出概率变化，保留原电流条件路径。

学习Head继续调用唯一FrontDoorEstimator.marginalize，不平均logits、不引入Mc→Y边。置换只是函数敏感度，组合可能偏离训练支持域，不是物理干预，也不是Y与Mc的条件独立性检验。低变化或高变化均不能单独证明机制正确。

此处do对观察标签的ACC/NLL只是诊断，真实数据没有干预概率真值。小批训练成功、前门输出正确实现、真实生理中介被识别是不同结论。

## 4. 代码和验证边界

- scripts/check_fixed_batch_capacity.py增加可选--mechanism_check/--mechanism_repeats/--previous_summary，默认不改旧探针行为。
- scripts/capacity_path_diagnostics.py重用原参考加载和缓存、原概率积分，检查末端推理前后模型参数/BN缓冲区及已有梯度不变，恢复模式和随机流。
- tests/test_capacity_path_diagnostics.py增加4项小型人工张量测试，区分只依赖Ma与只依赖参考的函数、守卫输入/概率形状及held-out来源。人工函数仅是诊断工具单测，不替换模型或提供正式结果。

正式模型、损失、自然训练协议、BN、优化器及选模规则全部不改。云端只静态AST/Shell与Git差异检查，不执行CLI、模型、测试或训练；新4项测试及机制诊断待服务器执行。原6项通过来自实际上传。

## 5. 服务器完整命令

继续使用dslenv。旧原始模型、capacity_001及旧工具均保留；本轮无需mainNeuroFDAN.py。新目录002用于末端机制结果，不覆盖001。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
git status --short
git worktree list
if [ -n "${GIT_SSH_COMMAND:-}" ]; then echo '已有自定义SSH命令，停止，不覆盖认证设置'; exit 1; fi
export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
timeout 600 git fetch origin feat/neurofdan-capacity-path-check
PATH_SHA="$(git rev-parse FETCH_HEAD)"
PATH_TOOLS=/home/fss/NeuroFDAN-capacity-path-tools
if [ ! -e "$PATH_TOOLS" ]; then
    git worktree add --detach "$PATH_TOOLS" "$PATH_SHA"
fi
test "$(git -C "$PATH_TOOLS" rev-parse HEAD)" = "$PATH_SHA"
test -z "$(git -C "$PATH_TOOLS" status --porcelain)"
PATH_FOLD=/home/fss/NeuroFDAN/outputs/phase6/population_alignment/DEAP/arousal/seed42/deap_arousal_fold01_nobalance20_001/fold01
PATH_PREVIOUS=/home/fss/NeuroFDAN/outputs/phase6/capacity/deap_arousal_fold01_natural_capacity_001/probe_summary.json
PATH_RESULT=/home/fss/NeuroFDAN/outputs/phase6/capacity/deap_arousal_fold01_natural_paths_002
test -f "$PATH_FOLD/last.pt"
test -f "$PATH_PREVIOUS"
test ! -e "$PATH_RESULT"
mkdir -p "$PATH_RESULT"
(
cd "$PATH_TOOLS"
bash scripts/phase3_run_logged.sh "$PATH_RESULT/path_tests.log" \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m unittest discover -s tests -p test_capacity_path_diagnostics.py -v
)
bash "$PATH_TOOLS/scripts/phase3_run_logged.sh" "$PATH_RESULT/train.log" \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -u "$PATH_TOOLS/scripts/check_fixed_batch_capacity.py" \
  --root_dir /home/fss --fold_dir "$PATH_FOLD" --output_dir "$PATH_RESULT" \
  --probe_subject 2 --samples 16 --steps 300 --check_every 50 --seed 42 --device cpu \
  --mechanism_check --mechanism_repeats 4 --previous_summary "$PATH_PREVIOUS"
cat "$PATH_RESULT/path_tests.log.exit"
cat "$PATH_RESULT/train.log.exit"
)
```

数据集/任务/fold/原轮数由脚本严格核验DEAP/arousal/1/20，不编造不存在的CLI参数。最多仍为600个私有小批更新；新增原参考库读取和末端推理，实际耗时以日志为准。失败立即停止并保留日志，不修改默认参数绕过复现守卫。结果在probe_summary.json的models.<起点>.mechanism，原权重不上传、不改写、私有权重仍不保存。

## 6. 上传预览

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
UPLOAD_TOOLS=/home/fss/NeuroFDAN-upload-timeout-tools
PATH_RESULT=/home/fss/NeuroFDAN/outputs/phase6/capacity/deap_arousal_fold01_natural_paths_002
test "$(git -C "$UPLOAD_TOOLS" rev-parse HEAD)" = 6681381298b74e9b69dc0df0594d5d9c7274acbc
test -z "$(git -C "$UPLOAD_TOOLS" status --porcelain)"
if [ -n "${GIT_SSH_COMMAND:-}" ]; then echo '已有自定义SSH命令，停止，不覆盖认证设置'; exit 1; fi
export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" "$PATH_RESULT" \
  --log "$PATH_RESULT/train.log" --log "$PATH_RESULT/path_tests.log" --git_timeout 600 --dry-run
)
```

## 7. 正式上传

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
UPLOAD_TOOLS=/home/fss/NeuroFDAN-upload-timeout-tools
PATH_RESULT=/home/fss/NeuroFDAN/outputs/phase6/capacity/deap_arousal_fold01_natural_paths_002
test "$(git -C "$UPLOAD_TOOLS" rev-parse HEAD)" = 6681381298b74e9b69dc0df0594d5d9c7274acbc
test -z "$(git -C "$UPLOAD_TOOLS" status --porcelain)"
if [ -n "${GIT_SSH_COMMAND:-}" ]; then echo '已有自定义SSH命令，停止，不覆盖认证设置'; exit 1; fi
export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
PATH_UPLOAD_ID="$(date +%Y%m%d_%H%M%S)_$$"
PATH_LOG="/home/fss/NeuroFDAN/outputs/phase6/upload_logs/capacity_paths_${PATH_UPLOAD_ID}.log"
PATH_BUNDLE="/home/fss/NeuroFDAN/outputs/phase6/upload_bundles/capacity_paths_${PATH_UPLOAD_ID}"
mkdir -p outputs/phase6/upload_logs outputs/phase6/upload_bundles
bash "$UPLOAD_TOOLS/scripts/phase3_run_logged.sh" "$PATH_LOG" \
  bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" "$PATH_RESULT" \
  --log "$PATH_RESULT/train.log" --log "$PATH_RESULT/path_tests.log" \
  --git_timeout 600 --bundle_dir "$PATH_BUNDLE"
cat "$PATH_LOG.exit"
cat "$PATH_BUNDLE/upload_state.json"
)
```

回传新的SHA和链接。失败也上传，不重跑真实20轮。上传成功与推理/复现检查成功分别判断。

## 8. 这一步后的决定

若自身参考拟合好、换成训练总体参考后输出趋同，并且固定Ma换参考影响远大于固定参考换Ma，优先研究条件结果监督的参考捷径与潜在中介学习约束；不能只加分类轮数。若Ma变化能稳定影响积分输出、拟合小批的do也能区分，转向全训练总体异质性和优化问题。两种结果都不直接支持完整LOSO放行，也不要求重构SCM。

小批100%只证明基础能力；后续每个真实方法改动仍需明确统计目标和假设。不会恢复类别平衡、调整正类/阈值、替换主输出、引入真实数据不存在的中介标签或自动合并main。
