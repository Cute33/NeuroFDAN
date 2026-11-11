# Phase6：20轮配对验证结果与下一步只读模块评价

## 1. 已核验的服务器证据

两份不可变上传：

- normal：提交76aff2bf633741493059762963e67576e7b0839b，目录runs/2026-11-11_170847_unknown_unknown_run_78325e34。
- tanh_normal：提交65de589c9b65ecdd7548200d6dd051dfd0ea3095，目录runs/2026-11-11_170955_unknown_unknown_run_97165f6e。

已逐文件重算manifest载荷长度与SHA256，全部一致、无跳过项；缺少的通用run_summary.json不是该合成程序的必需产物。两组completed=True、smoke=False、training_executed=True、real_data_training=False。每组120条历史记录对应2个gamma×3种监督×20轮，各轮连续完整；不是上传程序退出码替代训练完成证据。

源码均为15b79769535027f4dfb7de0cb5ab1c11252dec49且启动时无已跟踪修改。共同配置：seed42、CPU、训练2048/测试512、16个固定干预点、batch128、20轮、Adam lr0.001/weight_decay0、fidelity_weight1/fire_weight0；评价S16/K64、MC重复3次；直接真值32768次，已知前门积分S1024/K512。此处不改变真实训练器的默认参数。

每个gamma下三组及两个分布族的初始化指纹、训练X/Y指纹一致。无中介监督两组的训练字段仅x_c/x_a/y，教师组额外读取m_c/m_a。参考库使用全部2048个自然训练样本，无容量不足。

## 2. 预先确定的主判据结果

以下是重复积分平均概率相对直接干预真值的概率MSE；不是分类ACC。教师是“使用真实中介监督训练的神经网络”，不等同于已知SCM公式的Oracle数值积分。

| 中介族 | gamma | 真实中介监督教师 | 仅观察监督 | 观察+输入保真 | 训练比例常数 | 0.5常数 |
|---|---:|---:|---:|---:|---:|---:|
| normal | 0 | 0.006617 | 0.013430 | 0.037010 | 0.002912 | 0.003300 |
| normal | 1 | 0.004740 | 0.005524 | 0.066545 | 0.001797 | 0.002089 |
| tanh_normal | 0 | 0.004365 | 0.022829 | 0.044279 | 0.002912 | 0.003300 |
| tanh_normal | 1 | 0.004896 | 0.004181 | 0.054535 | 0.001797 | 0.002089 |

候选在四个预定设置均劣于观察baseline及两种常数。这一方向也存在于各独立MC重复，不是仅平均时出现的偶然排名。不能宣称单个训练种子、16点已给出统计显著或一般性结论；但已有结果足以拒绝把当前alpha=1候选直接移植为正式真实数据新方法。不通过继续加轮数、调整alpha或切换主输出掩盖这次失败。

已知SCM的独立前门公式与直接干预真值：gamma0的MAE=0.003533、MSE=0.00001957；gamma1的MAE=0.003033、MSE=0.00001454，均未触发预定真值容差。该数值误差远小于上述学习误差。两个族使用相同的真值，不能把它们当成两次独立机制验证。

教师在分布匹配的tanh_normal下仍劣于常数，说明现在不能把全部失败归因于“无中介监督不可识别”或“高斯族错设”。条件中介和条件结果模型的有限样本学习也需要拆开检查。

## 3. 为什么增强输入依赖仍失败

例如normal、gamma1下，CNS的跨输入posterior均值方差均值由观察baseline的0.05160增至候选0.90420，而平均posterior std由0.87559降至0.32088。tanh_normal、gamma1对应0.05900→1.08657及0.89057→0.29367。后者是tanh之前的高斯参数，不能解释为真实有界中介的方差。

因此，本候选确实学到了更强的输入变化，但没有恢复更好的干预概率。重建完整输入可能保存生成机制并不需要的信息，也可能与结果模型竞争；这些是待验证解释，不是已查明的唯一原因。输入保真不新增独立生理测量信息，不解决现有潜在中介非识别反例。

normal、gamma1候选最终训练事实CE约0.38573，观察测试NLL约0.99231；tanh_normal对应0.31488与1.08979。事实训练拟合改善与测试概率误差恶化并存，不能把更低训练CE当成方法有效证据。

教师Head在真实Ma和事实配对X上的条件概率MAE约0.10824（gamma0）、0.10898（gamma1）。两个族这项误差相同与其使用同一真实Ma监督、同一初始化/随机流一致，不直接构成bug证据。训练和交叉配对的条件误差仍需比较。

## 4. 下一步只做一次已有checkpoint的四格评价

新增scripts/evaluate_phase6_components.py。只读取本地paired_tanh_normal_20ep_001中的两个oracle_mediator_last.pt，不创建优化器、不反向、不改模型参数、不覆写源目录。严格加载原配置、轮数、seed、参数结构，重新核对SCM参数和训练X/Y指纹；评价前后核对模型状态及源文件SHA256。

| 代号 | 中介条件生成 | 条件结果概率 | 要回答的问题 |
|---|---|---|---|
| TT | 真实tanh机制 | 已知p(Y\|Ma,X') | 同一经验参考库及本轮MC能否复现真值 |
| TL | 真实tanh机制 | 已训练OutcomeHead | 固定正确中介后Head误差有多大 |
| LT | 已训练条件中介 | 已知p(Y\|Ma,X') | 固定正确条件结果后中介误差有多大 |
| LL | 已训练条件中介 | 已训练OutcomeHead | 原学习模型在更精细评价下的误差 |

每次重复四格共用标准中介噪声和参考索引，保留分层Mc→Ma。已知Head使用U|X'后验与高斯求积，不调用直接干预函数；学习Head继续调用原FrontDoorEstimator.marginalize，先softmax、后K轴及S轴概率平均。代码没有另造学习前门估计器。

仅对tanh_normal教师拆分：normal的无界Ma可能落到真实tanh支持域之外；无监督组的潜在坐标未被真实中介监督锚定，把真实Head直接接到其潜变量上也存在坐标歧义。故本次不做这些不可解释的替换。教师NLL监督仍有有限样本误差，四格只是固定坐标下的计算误差定位，不是一般潜在识别证明。

采用S256/K128、4次MC，仅用于已有模型评价，不修改checkpoint内的S/K、学习率或训练轮数。参考仍为原2048个自然训练样本；干预点仍为原16点；直接和总体前门真值来自原保存记录。两个gamma输出到同一新目录。

判读：先看TT与原直接/总体前门真值及MC标准误；若TT不稳，先处理经验库/数值精度。TT接近真值而TL差，优先查Head拟合/交叉条件预测；LT差则中介条件学习也不足。双方都差则均需处理。保留每格及每次重复的概率、MSE/MAE/适当评分；不把非线性四格差值写成可加的因果贡献比例。

同时保存真实Ma上的事实配对Head误差、交叉配对Head误差。二者风险分布不同，不直接把数值差当成严格OOD检验。MC重复只反映固定模型/经验库下的积分波动，不是多个训练种子，也未覆盖参考总体抽样误差。不要在同一结果上反复调参后宣称独立验证。

## 5. 服务器执行：不运行主训练函数

继续使用已工作的dslenv；不安装或升级依赖。主项目不切分支，新建隔离工具worktree，保留Phase4/5/6所有结果。首次命令中的分支必须先已发布。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
git status --short
git worktree list
git fetch origin feat/neurofdan-phase6-component-evaluation
PHASE6_COMPONENT_SHA="$(git rev-parse FETCH_HEAD)"
PHASE6_COMPONENT_TOOLS=/home/fss/NeuroFDAN-phase6-component-tools
test ! -e "$PHASE6_COMPONENT_TOOLS"
git worktree add --detach "$PHASE6_COMPONENT_TOOLS" "$PHASE6_COMPONENT_SHA"
test -z "$(git -C "$PHASE6_COMPONENT_TOOLS" status --porcelain)"
git -C "$PHASE6_COMPONENT_TOOLS" log -1 --oneline
PHASE6_ORIGINAL=/home/fss/NeuroFDAN/outputs/phase6/synthetic/paired_tanh_normal_20ep_001
test -f "$PHASE6_ORIGINAL/scm_parameters.npz"
test -f "$PHASE6_ORIGINAL/models/gamma0.0_oracle_mediator_last.pt"
test -f "$PHASE6_ORIGINAL/models/gamma1.0_oracle_mediator_last.pt"
PHASE6_RESULT=/home/fss/NeuroFDAN/outputs/phase6/evaluation/tanh_oracle_components_001
test ! -e "$PHASE6_RESULT"
mkdir -p "$PHASE6_RESULT"
bash "$PHASE6_COMPONENT_TOOLS/scripts/phase3_run_logged.sh" "$PHASE6_RESULT/train.log" \
  python -u "$PHASE6_COMPONENT_TOOLS/scripts/evaluate_phase6_components.py" \
  --run_dir "$PHASE6_ORIGINAL" --output_dir "$PHASE6_RESULT" \
  --seed 42 --device cpu --workers 0 --diagnostic_level full --S 256 --K 128 --repeats 4
cat "$PHASE6_RESULT/train.log.exit"
)
```

这里数据集为synthetic、无arousal/valence或LOSO fold，不存在主函数epochs/batch_size参数；它们沿原实验已保存配置核验，不启动训练。models及scm_parameters.npz保存在服务器原运行目录，GitHub脱敏快照没有模型权重。若任一文件缺失或严格加载失败，停止并回传日志，不重新训练或改成非严格加载。

程序保存probe_summary.json、config.json、run_report.txt，日志包装器保存train.log及train.log.exit。摘要中completed、source_files_unchanged、各gamma的state_unchanged须为true；每格against_direct为主误差。若失败，completed保留false，必须结合日志定位。

新增4项测试仅供服务器执行，云端没有运行。可在同一工具目录运行并将日志一并上传；不重复完整旧测试或训练：

```bash
cd /home/fss/NeuroFDAN-phase6-component-tools
(
set -euo pipefail
bash scripts/phase3_run_logged.sh \
  /home/fss/NeuroFDAN/outputs/phase6/evaluation/tanh_oracle_components_001/component_tests.log \
  python -m unittest discover -s tests -p test_phase6_component_evaluation.py -v
)
```

## 6. 结果上传

只上传新的评价目录，不再上传旧20轮训练目录或模型权重。先预览：

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
UPLOAD_TOOLS=/home/fss/NeuroFDAN-upload-timeout-tools
test "$(git -C "$UPLOAD_TOOLS" rev-parse HEAD)" = 6681381298b74e9b69dc0df0594d5d9c7274acbc
test -z "$(git -C "$UPLOAD_TOOLS" status --porcelain)"
PHASE6_RESULT=/home/fss/NeuroFDAN/outputs/phase6/evaluation/tanh_oracle_components_001
bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" "$PHASE6_RESULT" \
  --log "$PHASE6_RESULT/train.log" --log "$PHASE6_RESULT/component_tests.log" \
  --git_timeout 600 --dry-run
)
```

若未执行可选新增测试，删去对应--log；不要将未运行测试标成通过。正式上传保留持久bundle、独立日志及收据：

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
UPLOAD_TOOLS=/home/fss/NeuroFDAN-upload-timeout-tools
test "$(git -C "$UPLOAD_TOOLS" rev-parse HEAD)" = 6681381298b74e9b69dc0df0594d5d9c7274acbc
test -z "$(git -C "$UPLOAD_TOOLS" status --porcelain)"
PHASE6_RESULT=/home/fss/NeuroFDAN/outputs/phase6/evaluation/tanh_oracle_components_001
PHASE6_UPLOAD_ID="$(date +%Y%m%d_%H%M%S)_$$"
PHASE6_UPLOAD_LOG="/home/fss/NeuroFDAN/outputs/phase6/upload_logs/components_${PHASE6_UPLOAD_ID}.log"
PHASE6_BUNDLE="/home/fss/NeuroFDAN/outputs/phase6/upload_bundles/components_${PHASE6_UPLOAD_ID}"
mkdir -p outputs/phase6/upload_logs outputs/phase6/upload_bundles
bash "$UPLOAD_TOOLS/scripts/phase3_run_logged.sh" "$PHASE6_UPLOAD_LOG" \
  bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" "$PHASE6_RESULT" \
  --log "$PHASE6_RESULT/train.log" --log "$PHASE6_RESULT/component_tests.log" \
  --git_timeout 600 --bundle_dir "$PHASE6_BUNDLE"
cat "$PHASE6_UPLOAD_LOG.exit"
cat "$PHASE6_BUNDLE/upload_state.json"
)
```

成功后回传新SHA和GitHub链接；若失败则回传该上传日志末尾及收据。新run_id自动独立生成，不覆盖历史快照。摘要无需run_summary.json，manifest可能报告这一通用缺少项；其他缺少/跳过项仍须逐条检查。

## 7. 正式实验何时可开始

本轮工程接口和配对实验完整性已过关；方法证据没有过关。当前输入保真候选不进入真实数据迁移。下一次只有这一项只读四格评价，不再重跑环境检查、20轮或100轮。实际耗时由记录的seconds判断，不能承诺固定多久就能解决研究问题。

后续路线取决于本次定位：先修复可证实的实现错误，或定义可被合成反例否定的最小条件学习改进；再在同一机制/权限隔离下复核。仅观察X/Y的一般潜在中介识别缺口仍需额外测量或可辩护的结构限制，不会由教师拟合成功消失。

进入正式LOSO前，至少需要：明确方法目标和额外假设；合成评价中能区分正确机制、常数和非识别失败，而非只提高观察ACC；完成DEAP/HCI固定单折短程受控验证并确认状态、输入依赖和MC稳定性；冻结方法/主输出/配置/统计口径。不是要求某个ACC门槛或必须获得正面结果。失败方法可以作为明确标注的baseline报告，但不能包装成已验证的新因果方法。

主输出仍是probs_do，probs_obs仅诊断/明示对照；不更换SCM、真实骨干、损失、下采样、参考总体、seed42或test-selected checkpoint规则。真实中介不可观测、被试内归一化和测试选模偏差继续披露。无论合成结果如何，都不等于真实生理机制已被识别。

## 8. 本次代码和验证边界

仅新增本报告、只读评价脚本及4项针对性测试。原模型、训练器、生成器、积分器和上传工具均不修改。云端仅执行静态AST解析、文档Shell语法、Git差异与上传载荷分析；未执行模型、CLI、单元测试、真实训练或GPU实验。新增运行行为与测试结果待服务器核验。本分支不创建PR、不合并、不修改main。
