# Phase 5：真实日志与当前前门计算图审计

基线：feat/neurofdan-phase4，bc4d642b44b408419b489e084cb12d8c7a48c8e9。依据为该提交的实际源码和用户粘贴的服务器日志。只做静态代码/文本分析；本轮不运行测试、模型、真实训练、GPU或数值实验。本报告不宣称模型已修复。

## 1. 证据范围与直接结论

粘贴的逐轮块覆盖fold1、epoch6–25，不包含完整启动参数、epoch1–5、全部100轮结束记录或权重。训练评估10400窗口，真值[5200,5200]；测试400窗口，真值[210,190]。任务名称未出现在这些逐轮块内，不能仅凭计数指定arousal/valence；需要fold.json确认。

已由矩阵确认：epoch6、7、10–25主路径全预测1；epoch8全预测0；epoch9预测两类，但训练ACC=0.4997、测试ACC=0.5175。不是所有epoch都单类，也不能把epoch9视为已经恢复有效分类。

| 轮次 | 测试矩阵（行真实、列预测，[0,1]） | ACC/F1，正类1 | 事实测试矩阵 |
|---|---|---|---|
| 6 | [[0,210],[0,190]] | 0.4750 / 0.6441 | [[45,165],[25,165]] |
| 8 | [[210,0],[190,0]] | 0.5250 / 0 | [[210,0],[190,0]] |
| 9 | [[117,93],[100,90]] | 0.5175 / 0.4826 | [[109,101],[98,92]] |
| 25 | [[0,210],[0,190]] | 0.4750 / 0.6441 | [[0,210],[0,190]] |

事实训练ACC也多接近0.5，例如epoch6为0.5136、epoch9为0.5076、epoch25为0.5。因此这段日志不支持“事实路径已学好，只有前门坏了”；目前两条路径都没有展示可靠训练判别能力。

## 2. 实际计算路径与张量

入口mainNeuroFDAN.main → ModelTrain.run_loso → run_fold。NeuroFDAN.py只有一个有效NeuroFDAN；历史DeepTemporalEncoder/DAGGNN/SpikingMediator定义不参与该forward，不能根据它们的名字推断当前架构。输入仍是联合处理X=(X_c,X_a)，GNN通道图不是SCM。

| 环节 | 实际函数与张量，默认配置 |
|---|---|
| 输入 | PhysiologicalWindows：[B,32,768] CNS；[B,3,768] DEAP ANS或[B,4,768] HCI ANS |
| 骨干 | PhysiologicalBackbone.forward：三层分组时序卷积，CNS再做硬图混合；h_c=[B,96,128]，h_a=[B,96,12/16] |
| CNS | CNSMediator.forward → GaussianMediator.distribution：完整展平12288，经Linear→SiLU→Linear产生[B,4,128]驱动；CurrentLIF输出rates/current各[B,128]；参数头产生mu_c/logvar_c=[B,64] |
| CNS采样 | draw：mu+exp(logvar/2)×独立标准正态，m_c=[B,S,64]；train S=4，eval S=16 |
| CNS→ANS | ANSMediator.forward：W_c m_c+b_c=[B,S,128]，与每次ANS条件分支对齐为[B×S,128]，广播到4个时间步的电流；没有detach |
| ANS | 自身输入展平1152/1536后经同类编码器；额外电流在LIF的input_proj之后相加；mu_a/logvar_a/m_a=[B,S,64] |
| 事实结果 | ReferenceEncoder(h_c,h_a)默认完整拼接为[B,13440/13824]；OutcomeHead读取[m_a[:,0],g(x)]，产生[B,2] logits_obs及probs_obs |
| 经验参考 | build_reference_bank从当前fold未平衡全部训练窗口均匀无放回抽最多2048；ReferenceBank.sample按当前样本有放回抽K，并排除相同subject/trial/window；两模态保持配对 |
| 前门结果 | _LiveReferenceView训练时编码参考且保留梯度；评估按当前权重缓存。reference=[B,K,d_ref]；OutcomeHead组合logits=[B,S,K,2]；softmax(-1).mean(2).mean(1)→probs_do=[B,2] |

OutcomeHead第一层输入DEAP为13504=64+13440，隐藏128，SiLU、dropout0.3、双logit。维度悬殊提供参考捷径的可能性，但不单独证明该捷径实际占主导；输入尺度和权重同样重要。ref_dim=0不等于原始X的充分统计量。

## 3. 任务梯度与目标不一致

frontdoor_loss默认仅计算CE(logits_obs,y)+warmup beta×mean_B(KL_c+KL_a)。ModelTrain.train_one_epoch再按lambda_obs及fire_weight合成原损失。真实调用不提供oracle_probs；默认DEAP fire_weight=0，HCI=1。loss.py的旧CausalMultiLoss不在该路径。

事实CE的两条参数梯度路线：

1. CE→OutcomeHead→m_a[:,0]→ANS参数头/LIF/编码器→h_a；同时经cns_current→m_c[:,0]→重参数化→CNS参数头/LIF/编码器→h_c。
2. CE→OutcomeHead→当前g(x)→ReferenceEncoder→两支骨干。ref_dim=0时ReferenceEncoder是Identity、无独立可训练参数，但骨干仍可从该捷径获梯度。

CNS中介并非在静态图上断开；分类任务通过第一次层次采样连接它。其余S主要参与ANS KL或原发放率统计，不是S次事实CE监督。KL_c对D求和，保留[B]；KL_a对D求和后对S平均，得到[B]；最后对B平均。没有把CNS KL重复乘S的代码证据。logvar clamp为[-6,2]，mu没有tanh或强制归零；clamp边界外的logvar梯度可为零，但当前std≈1意味着logvar≈0，并非默认边界饱和。

probs_do虽然与CE共享中介及结果头参数，其参考组合张量没有被真实数据分类损失读取。**参数共享产生间接更新，不等于前门积分输出被有效校准或直接监督。** 前向参与不意味着该具体参考分支参与当前loss的反向；需要区分相同骨干参数通过事实分支得梯度，与该次参考计算得梯度。观察标签也不能直接改成do标签来“修复”目标不一致。

因此，即便事实CE未来改善，参考g(x)的判别线索在积分时被训练总体参考替换并平均后也可能消失。当前事实CE仍在0.693附近，说明不能仅用该目标不一致解释全部失败；还需查输入、训练图、骨干与中介局部敏感度。

## 4. CNS/ANS日志的准确解释

CNS完整评估的跨样本mu标准差及类均值距离为0，而std≈1，且KL近0，与测得样本上q(M_c|X_c)趋近相同标准正态的现象一致。这是**条件posterior趋同、输入依赖丢失的较强证据**，不是随机变量方差为零，更不是已证明真实生理中介不存在。

KL公式为1/2×sum_D(mu²+exp(logvar)-1-logvar)。mu≈0、logvar≈0正是其最小点；梯度也接近0。日志显示-0.000000可能来自浮点消减/打印舍入，不据此改变公式。需要原JSON参数均值与局部输入敏感度确认绝对mu及logvar，而不能把六位小数的KL当作严格0。

16个独立单位高斯样本采用ddof=0时，其样本标准差均值小于1，约0.952；日志采样内std约0.952与噪声采样一致。它不是输入编码有效的证据。posterior中位数0.999982也可能是4096箱估计的箱中心，应结合mean/min/max及median_bin_width阅读。

ANS的std_N(E_S mu_a)由epoch6约6.2e-5降到后续0，类距离也降到0；但是ANS logvar仍可依赖输入，且E_S mu的相同不严格推出每个条件mu(m_c)都相同。必须读取conditional_mu_within_sample_mc_std、posterior variance极值并做固定噪声扰动检查。

## 5. CNS→ANS、LIF与梯度证据

| 轮次 | CNS裁剪前梯度范数均值 | CNS→ANS均值 | CNS精确零batch/总数 |
|---|---:|---:|---|
| 6 | 1.0840e-5 | 3.4927e-4 | 0/260 |
| 15 | 1.0838e-13 | 1.4893e-8 | 0/260 |
| 25 | 2.3549e-42 | 3.7686e-24 | 84/260 |

这些是总loss的参数梯度，读于backward后、max_norm=1裁剪前，不包含Adam在step内部加入的weight_decay项，不能当作纯CE梯度，也不能从它们推出真实参数更新范数。极小值不是缺失梯度：对应missing_grad_tensors_mean=0。现有精确all_zero警告不会覆盖“多数极小但非零”的情况，原JSON保留这些数字；新的周期摘要会显示量级而不擅自设停止阈值。

早期非零梯度与代码中的实际电流注入排除了“整个实现一开始就detach”的说法；后期路径可能数值上失效，但因果方向与接口仍存在。权重范数、附加电流相对ANS原电流的RMS、固定m_c扰动的mu_a/logvar_a响应尚未提供，不能确定是量纲、权重收缩、输入无信息还是下游忽略。

CurrentLIF并非纯硬脉冲瓶颈：parameters_head拼接rates与未阈值化current.mean(time)，存在连续电流梯度旁路。SurrogateSpike.backward对input使用fast-sigmoid代理梯度，远离阈值时衰减；thresh显式返回None，但复位项仍能影响后续电位。不能仅凭硬阈值认定所有梯度被阻断。DEAP发放率损失虽被打印，默认权重为0，不能把约0.0025误写为正在强制约束DEAP的主因。

## 6. 前门公式、概率平均与参考总体

静态审查未发现把logits先平均、S/K轴颠倒或把参考换为原型的实现。chunk仅切K轴，按原顺序拼接组合logits后统一softmax并平均；eval关闭dropout时数学等价。train有dropout时不同分块可能分配不同随机mask，不能在没有固定mask的情况下承诺逐位相同；本轮不运行数值验证。

若q(M|x)对x几乎相同，且用同一经验p_ref(x')，积分结果本来就会近似不随当前x变化。常数接近0.5、其余约1e-3波动，可由小输入差异及有限MC噪声解释。概率平均可能削弱判别差异，但尚无证据表明平均公式写错；增加S/K可能仅减少噪声，不能自动恢复丢失的输入信息。

参考成员来自未平衡训练总体，优化训练子集则经过trial平衡；两者先验不同是明确设计，不暗改。均匀按窗口抽样不保证每名被试或每类等数，2048子集有抽样误差；还需reference_sources.json结合训练标签统计覆盖。代码有held-out和自身排除检查，但缺少这次服务器的来源文件，不能把静态assert当作这次运行无泄漏的独立证据。

## 7. 其他配置与证据等级

| 等级 | 内容 |
|---|---|
| A 已证实 | 上述轮次单类矩阵、两路径多为接近常数、CNS均值变化为0、总梯度极小；CE与do评价目标不同；无直接do监督；参考Identity无参数而非“参考梯度为零bug”。 |
| B 较强支持 | CNS条件posterior向先验趋同；中介任务信号弱并逐渐失去输入依赖；CNS→ANS数值调节减弱。不是唯一根因认定。 |
| C 待验证 | 骨干信息不足/过拟合参考捷径、KL与参数衰减的相对作用、LIF有效敏感度、评估图丢边、输入模态尺度、支持域不足、结果头忽略m_a、MC过度平滑的具体贡献。 |

输入按每个被试内部沿trial轴zscore，常量归零，不保证后续卷积/LIF各模态同尺度；实际输入标准差/近零通道需服务器统计。held-out自身无标签统计仍参与归一化，非严格在线评估。

BN running statistics冻结在train/eval均成立，仿射参数可训练；它不是使用训练总体估计的BN统计。可能改变激活尺度，但尚无激活记录支持其为根因。

训练CNS图用随机hard Gumbel，eval用固定扰动加当前dag_adj后argmax，再去自环。允许某行去自环后无边；需检查当前checkpoint有效边数及h_c变化。没有DAG惩罚保证当前图为SCM。训练/评估图差异不在本轮修复。

优化器是Adam，lr=1e-4、weight_decay=0.01、betas=(0.9,0.999)、eps=1e-8，全局梯度裁剪max_norm=1；不是AdamW。小任务梯度下L2衰减可能重要，需要单独分解而不是直接改参数。初始化是各nn.Linear/Conv默认方式、邻接单位阵加噪声，seed42。参数没有被每轮重置的代码证据。

额外训练评估用独立DataLoader并隔离torch/Python/NumPy随机流、恢复模式；原测试固定seed42且恢复torch流。静态未见新增评估推进优化sampler或更新参数的路径；不把上轮CPU单测当成本轮服务器验证。

checkpoint按主测试ACC>0.5后取更高ACC、平ACC时更高对应正类F1；0.525的全预测0也能满足初始门槛。epoch8“保存False”可能因早先已有更优best，当前粘贴缺epoch1–5，不能断言选模bug。规则不保证模型有判别能力；仍保持原规则并报告test-selected偏差。

## 8. 最小诊断与改动顺序

优先上传现有机器可读文件，不先重复100轮训练。随后在同一保存checkpoint、同一小批样本和相同MC噪声下，依序检查：输入/骨干变化 → CNS/ANS条件响应 → CE/KL/发放率独立梯度 → 电流及权重尺度 → 固定m_a的参考边缘化比较。具体实验及解释见[行动计划](phase5_action_plan.md)。所有探针是数值依赖诊断，不是对真实生理系统实施干预。

本轮分组：A实施日志分层、摘要；B实施独立分支白名单上传；C仅列出可复核bug修复候选，目前不实施模型训练修复；D仅列出理论/目标修改候选，不实施。没有证据支持把观察标签直接监督probs_do、删除KL或替换骨干。后续C/D必须先说明证据、公式、位置与预期影响。
