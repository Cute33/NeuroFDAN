# 收束锚点审查，推进有明确结论出口的学习实验

## 研究主线与当前阻断

最终交付仍是CNS→ANS层级概率中介与前门调整下的情绪识别，主要输出probs_do，主要实证结果是DEAP/HCI arousal/valence的LOSO ACC、F1及跨被试表现。事实输出用于诊断和明确对照，不择高汇报。机制、可视化和合成结果用于支撑、反驳方法，不能替代分类主结果。

当前阻断是完整真实训练总体仍接近多数类平台，而非缺少展示图。先完成一次限定的候选学习比较，再根据结果决定真实单折验证；不继续扩展特征筛查、不先制作大量解释性图，也不直接重跑旧配置完整LOSO。

## 本次服务器证据

DEAP：0fe4b4d4c6887a2e6a6c403bbadf5e7d02cb3dc6，runs/2026-11-12_134213_unknown_unknown_run_8e659dda。
HCI：c01a95b33f05ef422fc9c95ba7b75fb350382a0c，runs/2026-11-12_134442_unknown_unknown_run_87f542dc。

两份manifest各6载荷全部长度/SHA256一致、无跳过。相同8项测试日志通过，只算一套。两份completed及affine_invariance_passed均true，所有三种变换两分支超限数/掩码改变均为0，返回float32目标的最大差值均记录0。数学性质与数值检查一致，不是因果识别证明。

DEAP12400窗、160/4维，HCI2300窗、160/5维，所有特征active，源7b39f4efb0fb6b70777c7b7638c572145ce201a3。没有模型训练，未读取测试被试文件。

尺度不变不等于无身份信息：HCI三路ECG候选的被试间方差占比0.8575/0.9050/0.9177；两路GSR为0.0187/0.0241。DEAP GSR正变化候选仍为0.7147。不得据此称U已消除，也不再按最低被试方差继续筛选，避免把可能的情绪信息一并删除。

## 下一步：一次固定的8模型合成学习比较

仅normal族、gamma=1、aligned/misaligned两机制，每机制四组：oracle_mediator、observation_only、observation_physiological_anchors（scale_stable）、observation_original_anchors（旧统计）。每组20轮，共8个模型/160个epoch，CPU串行；不是8次真实单折，也不是多训练种子。不再额外跑一轮smoke，先执行新增4项接入测试。

旧统计对照必须在本次同一SCM和X/Y上重新拟合，不能拿先前不同aligned生成机制的结果直接对比。模型始终读取完整16/8维系数，前门实现仍是原FrontDoorEstimator。aligned中的中介位置只依赖新的相对统计；misaligned依赖原有符号系数，用于暴露信息遗漏。这是条件性有利/失败机制，不宣称真实生理系统满足aligned。

唯一新增运行开关为--anchor_variant scale_stable及--compare_original；默认original/三组保留原实验行为。两个锚点组仍为相同0.1权重的同一类工作损失，没有新增损失家族。模型维度、KL/warmup、S/K、Head等不变。四组共享X/Y、初始模型、批次顺序及逐批初始随机流；原锚点读出维数不同属于比较因素，模型初始指纹单独核验。

所有组锚点可解码性统一用训练数据拟合同一种新统计ridge探针，再到独立测试数据评分；不比较各自不同训练目标的原始重建误差。配置与记录区分训练锚点版本，无锚定组标记None。四组全部导出，原始统计对照命名Obs+Original。

normal相对tanh生成器有显式错设；此次固定normal对接当前真实模型，不根据成绩切换分布族。若所有组均失败，不立刻声称只有中介不可识别或只怪新约束；参考组也可能存在有限样本学习困难，但本轮不自动追加样本量扫描。

主要判据：相同机制/干预点上对直接真值及独立前门真值的概率MAE/MSE/NLL；同时看常数基线、MC波动、观察ACC/Macro-F1/BA、锚点探针和CNS→ANS响应。差异若与MC误差同量级，不判为优胜。不得只依据观察ACC或锚点R²通过。

结论出口：若仅更好重建锚点却未改善干预误差，则拒绝把该候选接入真实训练，不继续盲调alpha/特征；若结果支持且失败边界明确，下一步接入独立配置，DEAP/HCI固定fold1短程各一次，先看完整训练判别，再看测试主输出，保持全部原协议。之后冻结配置再做主LOSO，正式消融与可视化围绕同一冻结模型开展，不以图是否好看作为进入LOSO的条件。

## 服务器运行命令

继续dslenv，原项目和工具worktree保留，新目录。真实数据本步不读取，无需mainNeuroFDAN.py。测试中包含CPU小型前反向/step，随后仅合成训练，不在云端执行。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
git status --short
git worktree list
if [ -n "${GIT_SSH_COMMAND:-}" ]; then echo '已有自定义SSH命令，停止，不覆盖认证设置'; exit 1; fi
export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
timeout 600 git fetch origin feat/neurofdan-scale-anchor-learning
LEARN_SHA="$(git rev-parse FETCH_HEAD)"
LEARN_TOOLS=/home/fss/NeuroFDAN-scale-anchor-learning-tools
if [ ! -e "$LEARN_TOOLS" ]; then git worktree add --detach "$LEARN_TOOLS" "$LEARN_SHA"; fi
test "$(git -C "$LEARN_TOOLS" rev-parse HEAD)" = "$LEARN_SHA"
test -z "$(git -C "$LEARN_TOOLS" status --porcelain)"
git -C "$LEARN_TOOLS" log -1 --oneline
command -v python
python --version
LEARN_OUTPUT=/home/fss/NeuroFDAN/outputs/phase6/physiology_anchors/scale_learning
LEARN_RUN=paired_scale_normal20_001
LEARN_RESULT="$LEARN_OUTPUT/$LEARN_RUN"
LEARN_TEST_LOG="$LEARN_OUTPUT/interface_tests_001.log"
test ! -e "$LEARN_RESULT"
test ! -e "$LEARN_TEST_LOG"
(
cd "$LEARN_TOOLS"
bash scripts/phase3_run_logged.sh "$LEARN_TEST_LOG" \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m unittest discover -s tests -p test_scale_anchor_learning.py -v
)
bash "$LEARN_TOOLS/scripts/phase3_run_logged.sh" "$LEARN_RESULT/train.log" \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -u "$LEARN_TOOLS/physiology_anchor_experiment.py" \
  --output_dir "$LEARN_OUTPUT" --run_name "$LEARN_RUN" \
  --seed 42 --device cpu --workers 0 --diagnostic_level full \
  --epochs 20 --batch_size 128 --train_size 2048 --test_size 512 --points 16 \
  --gammas 1 --mechanisms aligned misaligned --lr 0.001 \
  --alpha_c 0.1 --alpha_a 0.1 --mediator_family normal \
  --anchor_variant scale_stable --compare_original \
  --mc_repeats 3 --direct_n 32768 --oracle_s 1024 --oracle_k 512 --plots
cat "$LEARN_TEST_LOG.exit"
cat "$LEARN_RESULT/train.log.exit"
)
```

任一失败停止，保留日志，不通过删守卫或改变学习率重跑同目录。completed=true、smoke=false，settings应有2项、每项4条件，history共160条；X/Y及模型初始化指纹各机制内部应一致。源码、参数、监督字段在config/probe_summary中记录。目录内metrics.csv及PNG/PDF供论文分析候选，权重只保留服务器，不上传。

## 上传预览与正式上传

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
UPLOAD_TOOLS=/home/fss/NeuroFDAN-upload-timeout-tools
LEARN_OUTPUT=/home/fss/NeuroFDAN/outputs/phase6/physiology_anchors/scale_learning
LEARN_RESULT="$LEARN_OUTPUT/paired_scale_normal20_001"
test "$(git -C "$UPLOAD_TOOLS" rev-parse HEAD)" = 6681381298b74e9b69dc0df0594d5d9c7274acbc
test -z "$(git -C "$UPLOAD_TOOLS" status --porcelain)"
if [ -n "${GIT_SSH_COMMAND:-}" ]; then echo '已有自定义SSH命令，停止，不覆盖认证设置'; exit 1; fi
export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" "$LEARN_RESULT" \
  --log "$LEARN_RESULT/train.log" --log "$LEARN_OUTPUT/interface_tests_001.log" \
  --git_timeout 600 --max_file_mib 16 --max_total_mib 64 --dry-run
LEARN_UPLOAD_ID="$(date +%Y%m%d_%H%M%S)_$$"
LEARN_UPLOAD_LOG="/home/fss/NeuroFDAN/outputs/phase6/upload_logs/scale_learning_${LEARN_UPLOAD_ID}.log"
LEARN_BUNDLE="/home/fss/NeuroFDAN/outputs/phase6/upload_bundles/scale_learning_${LEARN_UPLOAD_ID}"
mkdir -p outputs/phase6/upload_logs outputs/phase6/upload_bundles
test ! -e "$LEARN_BUNDLE"
bash "$UPLOAD_TOOLS/scripts/phase3_run_logged.sh" "$LEARN_UPLOAD_LOG" \
  bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" "$LEARN_RESULT" \
  --log "$LEARN_RESULT/train.log" --log "$LEARN_OUTPUT/interface_tests_001.log" \
  --git_timeout 600 --max_file_mib 16 --max_total_mib 64 --bundle_dir "$LEARN_BUNDLE"
cat "$LEARN_UPLOAD_LOG.exit"
cat "$LEARN_BUNDLE/upload_state.json"
)
```

回传一份SHA/链接；若测试先失败、尚无结果目录，则回传interface_tests_001.log及.exit错误末尾。所有数字在probe_summary/history内，旧上传白名单不含图/CSV/权重，图留服务器可SFTP下载。训练失败的部分结果同样保留并上传，不把上传成功等同学习完成。

## 交付范围

修改anchor_synthetic_scm.py、physiology_anchor_experiment.py以增加显式选项/同机制原锚点对照；新增tests/test_scale_anchor_learning.py与本文。原SCM节点/边、正式模型、真实损失、预处理、自然总体、seed42、BN和test-selected协议不改。核心前门估计器仍唯一；旧分支受保护。云端仅做AST、Shell、参数与Git差异静态检查，新增4项测试/学习均未执行。不修改main、创建PR或合并。
