# Phase 6：CNS–ANS潜在中介学习、监督权限与可反驳的最小候选

## A. 版本、证据与实际代码路径

本轮基线为6681381298b74e9b69dc0df0594d5d9c7274acbc。远端核验时main为b67bb1fad445f98299a893ffe25cda45edf5710b（已有PR15合并），文件树与6681381无差异。这是远端已有状态，不是本轮创建PR或合并。Phase4=bc4d642、Phase5=967f529、支持复核=cce93f0、上传修复=6681381均实际存在。新工作仅在feat/neurofdan-phase6-mediator-learning。

重新读取三份固定快照及manifest，逐文件重算长度/SHA256，全部一致：

| 用途 | 提交 | 目录后缀 |
|---|---|---|
| DEAP支持诊断 | 8a368c15ad5e1e9fd5f909a99547dd90c3d11f98 | 140435_unknown_unknown_run_4d0fa629 |
| HCI支持诊断 | cfcdbdff5696ed0061a5ef410dca5c1c6820da70 | 140543_unknown_unknown_run_8e5b753f |
| 二者汇总 | ea5283bab17f337698bf987d83a98129f971f851 | 145951_unknown_unknown_run_ebaffdce |

每份各四个载荷通过。缺少通用run_summary/run_report/config不代表探针失败；两份原探针日志各记录5项测试通过，模型/原文件保护通过。第三份是标准库汇总，不是新训练。另读取旧E2提交aa97f71与E3/E4提交e8d2893的实际probe_summary，确认旧E2为epoch1/37、wd=0.01，不把它当成新大batch epoch20的梯度证据。37轮训练是用户主动中止，非完整100轮。

### 真实入口与定位

| 文件、函数、基线行号 | 实际行为与审查结论 |
|---|---|
| mainNeuroFDAN.py:get_args:8、main:81 | root_dir取NEURO_ROOT或/home/fss；seed42；默认grouped，窗口shuffle需显式sampling_mode=window；原训练默认wd=0.01，大batch对照显式为0 |
| Datasets.py:normalize_subject:37 | 每名被试、每个通道/采样位置沿trial轴做zscore；不是沿窗口时间轴；零方差归零；不混合其他被试统计 |
| Datasets.py:PhysiologicalWindows:55 | DEAP原40×通道×8064，去384点基线后10×768窗口；EEG0:32、外周34/35/36（2EMG+GSR）。HCI20×通道×3840，5×768；外周32:36（3ECG+GSR）。标签arousal列1、valence列0，严格>5为1 |
| Datasets.py:balanced_indices:113 | 当前任务trial级两类下采样；各类数量还截成trials_per_subject的整数倍，保留所选trial全部窗口。WindowSelection保留原来源，不改原总体 |
| Datasets.py:GroupedSampler:156、make_loaders:180 | grouped每trial整组shuffle，window以组长1全局shuffle；训练drop_last，评估不shuffle、不丢尾 |
| ModelTrain.py:run_fold:374 | seed先于数据/模型初始化；训练所有非held-out，被试1为当前测试；没有validation。优化子集与未平衡原population明确分开 |
| ModelTrain.py:build_reference_bank:145、cache_reference_features:170 | 经验参考来自未平衡训练population；均匀抽成员，保存三元组；排除held-out/自身；评估按当前权重重建特征缓存 |
| NeuroFDAN.py:PhysiologicalBackbone:473、train:495、forward:503 | 当前有效骨干为三层分组时序Conv+冻结运行统计的BN+SiLU，CNS再经硬Gumbel通道图混合。历史DeepTemporalEncoder/DAGGNN/GRU类未参与此forward，不能依类名声称它们正在训练 |
| NeuroFDAN.py:NeuroFDAN:534、forward:549 | 唯一有效集成模型；h_c=[B,96,128]，h_a=[B,96,12/16]；完整参考13440/13824维。_LiveReferenceView:522训练时按同一骨干编码参考，保留共享参数梯度 |
| frontdoor_scm.py:GaussianMediator:102、CNSMediator:139 | CNS完整展平→Linear/SiLU/Linear→4步电流LIF→rates/current拼接→mu/logvar。CNS只读h_c；mu/logvar[B,64]，独立正态重参数化m_c[B,S,64] |
| frontdoor_scm.py:ANSMediator:149、CurrentLIF:72 | 仅h_a与采样Mc；WcMc+bc加到ANS驱动电流并广播时间，未detach。每个Mc有独立条件参数[B,S,64]。这才是CNS→ANS路径 |
| frontdoor_scm.py:OutcomeHead:167、FrontDoorEstimator:227 | Head只读Ma与参考，不读Mc；观察路径取Ma[:,0]及自身完整参考。ref_dim=0为Identity，保留选定骨干特征，不证明其对原输入充分 |
| frontdoor_scm.py:marginalize:239 | 显式[B,S,K,C]组合，softmax后mean(K)再mean(S)，不是平均logits。logits_do仅为最终概率的log |
| frontdoor_scm.py:gaussian_kl:291、frontdoor_loss:296 | D求和，CNS对B平均、ANS先S后B平均；beta线性10轮warmup。无oracle参数时不监督probs_do |
| ModelTrain.py:train_one_epoch:214 | CE(logits_obs,y)+beta(KLc+KLa)+fire；lambda_obs修正CE权重。DEAP默认fire=0、HCI=1。clip_grad_norm=1后Adam step。loss.py旧CausalMultiLoss不在实际调用路径，保留原文件 |
| ModelTrain.py:evaluate:256、prediction_probabilities:203 | full主输出始终probs_do；none为probs_obs。eval固定MC seed42并恢复随机流；额外训练集评估隔离模式/RNG，不用于选模 |
| ModelTrain.py:BestCheckpoint:125 | 保留ACC>0.5且提高时保存，平ACC时按DEAP正类1/HCI正类0的F1严格提高；test-selected偏差不改 |
| CalculateResult.py:main:53、result_audit.py:audit_fold:46 | 原正式统计完整性规则保留；Phase6合成结果不进入真实LOSO统计 |

三个总体必须区分：优化用标签选择训练子集、参考用未平衡训练总体、评价用held-out被试。held-out内部无标签trial统计参与zscore，因此非严格在线未知被试评估。原规则不使用测试标签计算训练权重。

### 原始数值复核（不是复制旧总结）

从原probe_summary重新提取Var_X(mu)、Var_X(sigma)并开平方得到RMS变化尺度：

| 指标 | DEAP | HCI |
|---|---:|---:|
| CNS均值RMS跨输入std | 0.000663108 | 0.020997723 |
| CNS sigma RMS跨输入std | 0.000361412 | 0.009284899 |
| CNS平均sigma | 0.992170514 | 0.964634703 |
| CNS平均条件KL，D=64求和 | 0.004836359 | 0.151562051 |
| 固定Mc，ANS均值RMS跨Xa std | 0.016291675 | 0.009865535 |
| 固定Mc，ANS sigma RMS跨Xa std | 0.008200763 | 0.004975235 |
| 积分Mc后ANS均值RMS跨输入std | 0.006399212 | 0.005173842 |
| 事实Ma尺度归一局部梯度L2均值 | 0.004384017 | 0.006694356 |
| 自身参考S平均P1跨查询std | 0.024215808 | 0.008726922 |
| S/K平均P1跨查询std | 0.001035499 | 0.001889912 |
| 自身/交叉低密度比例 | 5.47% / 6.26% | 4.30% / 5.62% |
| 事实/交叉最近邻平均距离 | 1.9253 / 1.9192 | 2.2774 / 2.2610 |
| 原训练两类窗口数 | 5270 / 7130 | 1450 / 850 |
| 优化两类窗口数 | 5200 / 5200 | 800 / 800 |
| 类别保留比例 | 98.67% / 72.93% | 55.17% / 94.12% |

这16个查询来自训练总体，不是完整测试集。概率std含MC噪声。ANS对Mc积分后的矩也是有限MC估计。早期单类预测已有原矩阵支持；当前表不生成新的测试ACC或混淆矩阵。

已证实：接口分层、局部Ma梯度非零、DEAP CNS弱输入变化、积分输出接近平坦、优化/参考总体不同、Phase1 learned读取真值中介。尚未证实：真实中介完全无信息、Head完全忽略Ma、真实overlap充分、参考维数“压制”中介是唯一根因、BN/图差异是唯一根因、提高ACC即恢复因果机制。

## B. 方法学判断与梯度归因

固定同次forward骨干Z时，默认完整参考即Z，重新独立采样的潜变量形成p_opt(Z,Y)q_phi(tilde_M|Z)。其事实CE总体最优允许q_theta(Y|tilde_Ma,Z)=p_opt(Y|Z)，不需要利用重新采样的Ma。真实SCM中同次中介噪声会进入Y；重新抽取独立噪声再附事实标签并不等同于观测到该真实中介。

因此完整前门积分虽正确实现，并未自动补足真实中介测量/识别。{Mc,Ma}不是本图最小前门集合；Ma单独也满足图上的前门条件，保留Mc是机制建模。参考g(X')不是新因果节点，不添加X→Y或Mc→Y。

E_X KL(q(M|X)||p(M))=I_q(X;M)+KL(q(M)||p(M))。真正很小的平均KL限制输入信息，但有限16窗、舍入值和采样std不能证明整个总体没有信息。ANS条件KL与q(Ma|X)混合分布KL不同；前者大可能主要反映对Mc噪声响应。

任务CE梯度：Head→Ma[:,0]→ANS及cns_current→Mc[:,0]→CNS；另有Head→自身参考→两骨干。KLc只进CNS分支；KLa经条件Mc还可进CNS；fire项依rates；do分支未被CE读取，但其参数与CE共享。不能将总梯度或一次Ma输入梯度当作分项监督充分性的证明。

新phase6_gradient_audit在原20轮checkpoint、原16查询、原参考库内64成员上，同时测train（BN冻结）/eval的加权CE、KLc、KLa、fire各自参数范数和对CE的夹角，分开CNS/ANS/电流/Head/两骨干/图。另计算平均P1的do局部导数，仅诊断，不加到loss，不使用do标签。原37轮E2无法代替这项新权重证据。夹角只描述局部竞争，不证明某项应删除。

这些导数沿用原脉冲surrogate gradient，是模型训练计算图中的局部诊断，不是对真实生理系统的因果效应测量。

BN运行统计始终冻结，仿射可训练；训练图随机hard Gumbel而eval图固定；Head dropout训练开启、eval关闭。这些模式差异保留并分别测量，未静默修复。主S/K、维数、学习率和选模规则均未更改。

中介LIF的4个驱动步由完整特征经过Linear重新生成，不等于原始信号的四个生理时间段，不能把它当成已经实现已知CNS→ANS生理滞后的证据。当前骨干卷积也不能依未使用的旧类名认定为历史因果时序卷积；本轮保留实际骨干，先报告这一研究描述边界。

实际Conv1d采用对称padding，不是严格左侧padding的因果卷积。这是当前骨干与可能的论文时序表述之间需要区分的事实，不据此擅自替换骨干，也不把6秒窗口内的分类处理自动认定为标签泄漏。

下采样概率若只依赖Y，一般p_opt(y|m,x)∝a_y p_population(y|m,x)。实际trial截断还可能产生有限样本被试差异，不能把实现后保留率直接当作已知总体选择模型。本轮不加class weight、不取消平衡；需要单独定义目标总体后再讨论选择校正。

## C. 候选改进与风险比较

| 方向 | 数学/生理动机 | 复杂度 | 主要风险 | 当前决定与验证 |
|---|---|---|---|---|
| A 本模态随机中介输入保真 | Mc应保留CNS信息，Ma在层级路径下保留ANS信息；以观测输入统计重建作工作似然 | 两个线性训练期读出，低 | 可能保留U或无关输入、重参数化不可辨，仍无识别保证；可能压低sigma | 先做可反驳合成候选，不接真实训练；独立看干预误差、sigma、KL、梯度 |
| B 滞后/条件动态协同 | 检查Mc对未来ANS相对自身历史的增量条件预测 | 中到高，需预定义时轴与滤波/滞后 | U造成时序相关；现有trial评分不是逐时间窗Y；方向先后不等于因果；可能需扩展动态SCM | 暂不实施；需生理滞后依据、同被试trial阻断的预测检验及负滞后对照 |
| C 限制参考捷径或do标签CE | 参考dropout/投影可能迫使Head读取Ma；do标签CE实际优化观察标签风险 | 低到中 | 限制p(y|ma,x')容量、破坏充分性；E_obs[-log pdo(Y|X)]不是无偏干预监督；可增加ACC而偏离目标 | 不作为首选，不删参考、不加伪交叉标签；先检验条件结果误差 |
| D 独立测量proxy/可识别生成约束 | 通过外部测量锚定中介坐标与结果联合关系 | 高，依赖可获得测量 | 独立性/秩/已知噪声等强假设，当前DEAP/HCI未核实 | 作为识别闸门；不能将输入派生特征自动当独立proxy |

## D. 推荐的最小方案：识别闸门优先，A作为失败对照候选

推荐先执行分项梯度复核与三组配对合成。当前没有已论证可识别的真实数据新方案，不能以“解决塌缩”为名直接迁移A。A是有统计工作目标的候选，不是已获识别保证的约束；代码明确使用observation_input_fidelity命名，而不称identified或causal_supervision。

训练期线性读出Dc(Mc)→标准化Xc、Da(Ma)→标准化Xa。标准化统计仅来自合成训练集。用每个真实采样中介而非只用mu：

L_fid,c = (1/2) E_B,S[||Dc(Mc)-standardize(Xc)||² / d_xc]。

L_fid,a = (1/2) E_B,S[||Da(Ma)-standardize(Xa)||² / d_xa]。

这等价于单位方差Gaussian工作读出的负log似然去除常数，具有明确评分目标，但不假设真实SCM存在M→X反向结构边。ANS的CNS信息仍只通过采样Mc的电流进入；读出不进入OutcomeHead，不生成新预测路径。推断仍只用原FrontDoorEstimator。

L_base = lambda_obs CE(logits_obs,Y) + beta_warmup(KLc+KLa) + w_fire L_fire。

输入保真并不改变p_opt(Z,Y)q_phi(tilde_M|Z)的增广分解，所以不能从理论上排除Head忽略Ma的事实CE总体最优。本轮将其作为“非塌缩是否足以改善干预估计”的反驳性候选，不能将它包装成已解决条件结果监督来源的问题。

还需防范相反效果：ANS重建可能为了压制Mc采样噪声而减弱cns_current，即使Xa重建更好也会削弱研究关注的层级机制。评价单列固定Xa时替换CNS后验参数的ANS变化、Mc条件均值方差和电流权重尺度，不只看重建或ACC。合成输入是抽象向量，本轮没有建立它与HRV、频带功率等真实生理量的测量标定关系，不能宣称已经获得生理可解释中介。

候选L = L_base + alpha(L_fid,c+L_fid,a)，默认alpha=1仅为预先固定的首轮合成工作权重，不是调优最优值。alpha=0恢复baseline总损失，测试覆盖。只比较同一组内同初始化、同X/Y、同批次顺序和随机流；不依测试成绩挑epoch或权重。

否定条件：只提高输入重建/均值差异或观察ACC，却未改善干预概率误差；sigma塌缩/不确定性校准恶化；对无CNS路径同样有效而无层级机制证据；或在当前自然参考总体/反例上失效。失败应记录，不能直接加大alpha或删除KL追分。有限单种子/少干预点的改善只支持继续检查，不证明一般识别。

### D方向新增假设清单（均未宣称真实数据满足）

1. **独立测量模型**：若有Zc=LcMc+eta_c、Za=LaMa+eta_a，eta互相及与(X,Mc,Ma,U,Y)独立，L已校准且满列秩、噪声特征函数已知且非零，则原则上可对联合(Zc,Za,X,Y)去卷积恢复相应中介联合分布，再估计S9各项。生理解释为独立仪器/测量通道对同一状态的校准读出。需要重复测量、残差依赖与秩/噪声敏感性检验；不可能仅靠观测完全证明独立性。违反时出现残差相关、条件密度/干预误差随测量噪声模型强烈变化。现有输入派生bandpower等不自动满足独立误差；本轮不伪造Z，不扩展真实SCM。
2. **动态稳定性与序贯外生性**：用时间索引推广时需要明确条件于过去后的独立创新、无未控制的时变共同原因及跨trial稳定机制。生理解释为预定义CNS→ANS响应时序；可反驳检查为负滞后/时间反转对照、给定自身历史后的增量预测与跨trial稳定性。违反时正负滞后均可预测、机制随被试/时段漂移。仅时间先后不识别中介，动态模型需显式修订/批准，不能暗改当前静态S1。
3. **环境辅助的机制变化**：需明确哪些环境只改变外生分布而不改测量/结果机制，且变化足以使参数到观测分布映射对目标效应可辨。被试不是天然有效工具变量；U本就与被试差异相关。检验为预先定义环境的条件机制稳定性/负对照与参数敏感性。若混杂/测量同时变动，可能出现观察泛化改善而干预恢复失败。一般独立性/完备性不能从LOSO自动推出；当前不默认这些条件成立。
4. **可识别生成族**：必须论证theta→p_theta(可观测量)在保持目标效应的等价类上单射，不能只指定神经网络和先验。可用同观测不同干预反例、多个等拟合解及模型错设敏感性反驳；通过有限测试不能证明全局单射。对S9的作用是使潜在条件分布有依据，而非改变积分公式。

当前严格的识别候选是“有已校准独立测量”的条件性方案，但所需真实测量尚未建立。A/B/C都不能凭自身补足该信息。不能在报告中把第三组输入保真称为已经满足D。

## E. 独立代码及权限

- phase6_learning.py：check_training_view白名单；ModalityReadouts两个线性读出；learning_terms分组损失；gradient_components逐项范数与CE夹角。
- phase6_synthetic_experiment.py：三组配对fit，独立evaluate，训练/评价严格分离；保存最后模型而非测试选模；共同噪声与独立噪声积分比较。
- phase6_nonidentifiability.py：解析同观测不同干预反例，只用标准库。
- scripts/phase6_gradient_audit.py：原epoch20权重只读梯度；无优化器、不写.grad，校验模型和文件不变。
- tests/test_phase6_learning.py：六项针对监督权限、分组梯度、样本重建、零权重baseline与反例的测试，待服务器执行。

原NeuroFDAN.py/frontdoor_scm.py/Datasets.py/ModelTrain.py/mainNeuroFDAN.py/loss.py/统计/上传工具均不修改。真实模型baseline完整保留，尚未增加真实训练新方案开关，以免跳过识别与合成闸门。新增读出只在独立合成实验中存在。

## F. 三组对等合成验证

同一原SyntheticSCM（tanh-Gaussian中介、独立外生噪声、非零Cca、Bernoulli结果）及相同固定点。三个条件：

- oracle_mediator：训练视图Xc/Xa/Y/Mc/Ma；真实中介条件NLL及真实Ma结果CE，另保留共同KL/fire。它是额外信息参考，不保证数值上总是最优，也不声称真实数据可用。
- observation_only：训练函数仅收到Xc/Xa/Y，按现有事实首样本CE+KL/fire。
- observation_input_fidelity：同样仅Xc/Xa/Y，另加上述输入保真工作约束，不读取真实中介/U/干预概率。

fit不接收SCM对象或测试真值；两个非oracle组连训练字典中也不保留禁用字段。三组共用X/Y指纹、初始模型指纹、每epoch数据顺序、每批初始随机流、lr/batch/轮数。训练为自然总体，不下采样；用于隔离学习问题，不声称复制真实trial平衡。合成训练只计算有监督事实分支，省去无loss的K组合；正式评价调用原完整前门积分。

预定义两种学习分布族normal/tanh_normal。前者对接真实模型分布，但相对tanh生成器存在显式错设；后者为分布匹配控制，KL解释为tanh推前先验，与真实normal先验不同。全部比较必须在各自族内配对，不能挑更高ACC的族冒充真实新方案。tanh参数诊断为变换前Gaussian参数，不能当tanh后实际sigma。默认维度64/128、S/K=4/16与16/64、Nref2048、beta0.001/warmup10、dropout0.3保留；合成lr0.001、batch128沿用Phase1实验尺度，不改真实lr0.0001。

先独立计算直接干预（重采边缘U）及oracle前门积分，不复用同一路径；保存各自时间、MC标准误。差异超过4倍组合近似SE加0.01的预设工程容差时保存部分结果并停止学习；这不是严格统计等价检验。测试只评价最后轮，不提供给fit或选择权重。

评价包括：观察路径完整二分类矩阵、ACC/Macro-F1/BA/Precision/Recall/Specificity、观察NLL/Brier；do对观察标签的分类诊断单列；对直接/oracle真值的干预MAE、概率MSE、期望Brier/NLL、10箱概率校准；条件中介NLL、真实Ma上的条件结果误差；mu/sigma输入依赖；重复MC稳定性；0.5与训练类别比例常数基准。固定坐标中介NLL及真值Ma上的Head误差都对可逆重参数化敏感，不能作为唯一恢复判据。MC重复不是多训练种子；本轮seed42没有跨种子显著性结论。

独立观察测试与干预点不相同，三组共享两者。条件NLL及结果真值只在evaluate中读取。已有合成中介尾部float32截断到±(1-1e-6)用于数值评分/监督，真实生成及真值积分不变；不能将尾部近似说成精确真分布。

非识别反例保留同一DAG：Xc=U+ec、Xa=U+ea、Mc=Xc+emc；A的Ma=Xa+Mc+ema、Y*=Ma+U+ey；B的Ma=(2/3)(Xa+Mc)+ema、Y*=Ma+2U+sqrt(5/9)ey；Y=1[Y*>0]。两观测协方差同为[[2,1,4],[1,2,4],[4,4,14]]，但do均值/方差不同。代码报告同一观测预测器至少对一个机制存在误差的解析下界，不伪造训练成功。Y*只是函数记号，不加SCM节点。

## G. 服务器顺序与真实数据准入

详见[完整命令](phase6_server_guide.md)。第一批运行：六项小测试→DEAP/HCI原权重只读分项梯度→三组1轮合成smoke，上传三个目录。第一批不能判断方法已改善。运行行为确认后，再执行预定义normal与tanh_normal各20轮配对合成，不直接进入真实LOSO。

真实数据方案暂不接入：尚无第三组改善结果，也无真实测量识别依据。后续即使合成误差下降，仍须明确额外假设、选择总体并保持原train/test二分与test-selected规则。先各单折短程、单因素，再考虑正式LOSO；主预测仍probs_do，事实只作明确对照，不择高汇报。

## H. 本轮验证状态

已执行仅静态审阅、固定Git产物哈希/数值复核、Python3.10 AST、文档Shell语法与git diff --check。没有在云端导入/执行新模型、CLI、单元测试、合成训练、真实训练或GPU/LOSO。新增代码均待服务器验证；本轮没有改善ACC或恢复因果效应的实测结论。目录、源码SHA和权限在配置/探针中保存；日志与参数文件上传，真实数据及模型权重不上传。
