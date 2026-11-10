# Phase 4：训练异常的代码诊断依据

基线为 feat/neurofdan-phase3，提交 cd844c85d28f2335acf4d41083184d8ad98e5bbd。本报告先记录代码证据与最小改动方案，再补充测试记录。没有服务器真实标签、每轮预测或完整日志，不能确认用户所述 epoch 已发生单类别预测，也不能判定根因。

## 1. 当前实际执行路径

mainNeuroFDAN.main → ModelTrain.run_loso → run_fold → train_one_epoch/evaluate → NeuroFDAN.forward → backbone → frontdoor_scm.FrontDoorEstimator。loss.py 的历史 CausalMultiLoss、阈值分类函数不参与当前真实训练。

训练 frontdoor_loss 使用 logits_obs 的双类别交叉熵。观察路径取一次完整层次采样 m_a[:,0]，并使用当前事实样本 g(x)；probs_obs 为其 softmax。总损失保持 lambda_obs×事实CE + warmup beta×(KL_c+KL_a) + 原 firing rate loss。没有 oracle 标签，不对 probs_do 使用观察标签 CE。

full 模式的训练过程指标、测试指标与 BestCheckpoint 都使用 probs_do.argmax(-1)。probs_do 是每个组合 softmax 后沿 K、S 平均的前门估计；它与 factual prediction 不等价。none 模式保留同一中介与 OutcomeHead，仅使用 probs_obs，不执行参考积分。新增主要训练集评估也必须沿用该主路径，并单独标记 factual 对照。

评估仍有高斯重参数化和有放回参考抽样。evaluate 在 fork_rng 内固定 torch seed42，退出后恢复训练随机流；模型 eval 关闭 dropout，骨干使用固定 eval_gumbel，BN 运行统计冻结。同一权重、输入顺序、batch/S/K、参考库与环境下，CPU 重复评估应可复现，需单测验证。直接 model.eval(); model(...) 未自行固定 RNG，不能据此保证重复推理一致；不同硬件/版本、batch 分组、参考库或采样配置也不能承诺逐位相同。

## 2. 原均值离散警告的准确含义

NeuroFDAN.forward 的 mu_c_spread 为 mean_D std_B(mu_c[B,D], unbiased=False)。mu_a_spread 为 mean_D std_(B×S)(mu_a[B,S,D])：将 B 与 S 合并。train_one_epoch 将每 batch 结果乘 batch 样本数再平均；run_fold 在任一值 <1e-3 时输出同一条提示。因此它不是跨 batch 均值的标准差，也不是完整训练集上的 std_N；不同 batch 常数均值可以不同，旧数值仍为零。ANS 还混合了跨样本差异与给定样本不同 m_c 采样引起的条件变化。

1e-3 是未标定的尺度相关启发式，不是统计检验或因果识别条件。低均值变化可同时伴随较大 posterior std 和非零采样变化。logvar 的默认 clamp [-6,2] 还使正常高斯 posterior std 至少 exp(-3)，不可能严格降至零。只凭该提示不能声称方差塌缩或完全无信息。

本轮方案：完整评估流中，分别统计 std_N(E_S mu)、std_S(mu)、样本内潜变量维变化、posterior std/variance、std_N(E_S m)、E_S std_N(m) 与 E_N std_S(m)。CNS mu 无 S 轴，ANS 的 E_S mu 是有限MC估计，仍含采样误差；S=1 时不能判断同样本采样变化。各维均值、标准差及两类差异作为描述性诊断，不解释为因果效应。

## 3. 用户报告指标与单类别预测

DEAP 当前标签映射：valence=labels[:,0]，arousal=labels[:,1]，评分>5为1；没有交换。真实/预测类别索引均为[0,1]，主路径双概率 argmax，平概率选0。正类1，F1=2TP/(2TP+FP+FN)，不存在当前路径 BCE 阈值与双logit混用的证据。HCI 正类0但矩阵次序仍是[0,1]。

以下只是根据“全部预测同一类”的条件构造的可验证候选，不是服务器实测：

| 假设的真实计数[0,1] | 全预测1的矩阵 | ACC/F1(正类1) | 全预测0的矩阵及ACC/F1 |
|---|---|---|---|
| Valence [210,190] | [[0,210],[0,190]] | 0.475 / 0.6440678 | [[210,0],[190,0]]；0.525 / 0 |
| Arousal [160,240] | [[0,160],[0,240]] | 0.600 / 0.750 | [[160,0],[240,0]]；0.400 / 0 |

这些与报告数值相符，但数值相符不是预测记录。Phase 3 history.jsonl 已保存每轮 confusion_matrix，可直接查看；best_predictions.json 只对应最佳轮，不能证明第72轮状态。Phase 4 每轮保存并打印完整真值/预测计数和矩阵，服务器运行后才能验证。云端不读取或编造真实标签分布。

## 4. 需要区分的候选解释

- OutcomeHead 参考维度为 DEAP 13440，而 m_a 为64维；事实 CE 可以依赖 g(x) 学分类，前门边缘化后参考线索被平均。若 factual 与 do 明显分离，这是目标与路径差异的证据，不能自动归因于标签错误。
- KL 拉近标准正态，事实路径对中介依赖不足时可能减小均值差异；应联合看 CE、两支 KL、posterior variance、采样与梯度。不能未经实验删除 KL 或调整权重。
- 训练骨干随机 Gumbel 图与固定评估图不同，可能影响训练/评估差异；BN 统计始终冻结。两项属于既定模型行为，本轮不改。
- SurrogateSpike.backward 对 thresh 的显式返回为 None，但膜电位复位 voltage-spike×threshold 仍有可微路径，并可影响后续时间步。因此不能只看 surrogate 的返回就断言 threshold 完全没有梯度，更不能据此宣布整个中介无梯度；需实际读取 backward 后梯度。CPU 小批测试确有该路径，此点不作为模型 bug 修改。
- 常量分类、近似均值、方差不足、纯噪声采样、梯度缺失、梯度被裁剪后很小是不同问题，应分别检查。

## 5. 最小实施方案与证据边界

目前没有证据证明服务器异常由标签列、类别顺序、前门公式或损失实现 bug 导致，因此不修改这些内容。明确的可观测性缺口是：旧训练指标混合不同 step 权重及训练模式，不能代表 epoch 末完整训练集；均值警告归约混淆；日志缺少预测分布、factual/do 对照和梯度诊断。

新增独立诊断模块，不改 NeuroFDAN.py、frontdoor_scm.py、Datasets.py、loss.py。完整训练集指实际用于优化的 WindowSelection 全部窗口（通常已平衡），评估时不shuffle、不drop_last；参考总体仍是未平衡全部训练窗口。明确记录两者数量，不静默更换训练分布。

测试按既定路径先执行，额外训练集评估随后使用同一权重与参考缓存，并隔离 RNG、独立 DataLoader，避免新增诊断扰动原测试选模或下一轮训练。评估无梯度、不step、恢复模块模式；basic 默认打印主路径与事实对照，full 增加流式中介统计和裁剪前模块梯度。smoke 可以有限评估，但必须标记不完整；正式运行每轮完整评估两集。

每轮保存 epoch_metrics.jsonl、epoch_confusion_matrices.npz、mediator_diagnostics.jsonl、collapse_diagnostics.jsonl；完整中介向量仅保存一处，其他记录引用并保存少量摘要。断点续训按 epoch 更新诊断产物，不伪造缺失历史。既有 history、最佳预测与 checkpoint 格式保留兼容。

## 6. 已执行核验与尚未确定的结论

71项轻量测试中70项通过，1项CUDA测试因无GPU跳过。详细逐项结果见 [测试记录](phase4_test_results.md)。新增14项测试覆盖两种正类、矩阵次序、全预测0/1、full梯度、均值与方差变化区分、完整训练评估不修改参数/BN/优化器/随机流、CPU重复推理、basic/full相同更新、诊断文件续训及日志完整性。额外直接读取Git中的Phase 3原evaluate，与Phase 4 full在相同5窗口CPU夹具上比较：旧指标、逐窗口概率及矩阵完全一致。它们都不是服务器数据实验。

没有修改或修复已证实会造成真实DEAP塌缩的模型bug，因为目前尚无这种证据。已补足的确定问题是诊断口径和可观测性不足：原训练过程指标不能代表epoch末完整训练集，原ANS离散度合并了B/S轴，且原日志缺少直接的类别预测计数和优化梯度。保留旧标量用于历史对照，新的完整流统计才用于分轴诊断；不把该改进描述为已经解决塌缩。

修改ModelTrain.py以增加只读完整训练集评估、分类指标和诊断调用；mainNeuroFDAN.py仅增加默认basic的诊断参数；training_diagnostics.py新增流式统计、梯度读取及文件保存；新增测试、分析与服务器指南并更新README。NeuroFDAN.py、frontdoor_scm.py、Datasets.py、loss.py、CalculateResult.py、result_audit.py及requirements.txt保持基线内容。

下一步必须在服务器上取得相同epoch的两组矩阵、事实/前门对照、中介与梯度诊断，才能区分实现异常、优化失效或估计器目标/建模限制。目前不能确认真实训练集是否已经学会区分两类，不能声称中介已塌缩或已恢复，也没有优化ACC的实验结论。分类判决、采样及checkpoint设置未改；新增1%预测失衡阈值仅用于提示，无validation，无真实训练或GPU实验。
