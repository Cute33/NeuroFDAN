# 第二阶段交付报告

## 版本保护与本次范围

第一阶段检查时尚未纳入Git：HEAD为19bf18f，分支feat/front-door-scm，新增源码未跟踪。已完整保存到 **feat/neurofdan-phase1**，提交 **f1af90792e170c2c8fa183775cf22c3757c594f6**，已push并用git ls-remote核验远端SHA一致。核验日志与历史配置见phase1_test_recheck.txt、phase1_experiment_config.json、phase1_version_preservation.md。

第二阶段从上述提交创建 **feat/neurofdan-phase2**；最终提交SHA和远端核验结果见交付回复，也可用git log与git ls-remote查询。未直接提交/推送main，未合并、打发布标签或创建PR。

第一阶段ZIP是结果归档，不含独有代码，仍保留在云端且不上传。生成的results、虚拟环境和缓存不提交；主要交付为Git中的源码、配置、测试与文档。第一阶段历史报告内的“未提交”描述其当时状态，不代表保护分支未发布。

本阶段未访问服务器、下载真实数据、执行真实LOSO或GPU训练，也未重做五种子合成实验。仅运行既有单元测试、临时结构夹具和小型随机tensor接口测试。

## 修改文件

| 文件 | 作用、修改及原因 |
|---|---|
| NeuroFDAN.py | 保留原基础组件，清除被覆盖的重复模型、不可达旧前门路径及原型模块；只保留一个有效NeuroFDAN。复用实际有效卷积/图混合，接入Phase 1核心，输出统一字典。完整历史在Phase 1分支。 |
| frontdoor_scm.py | 不重写识别公式；增加参考K分块、数据集标识、可保留输入梯度的参考数据、实际发放率接口。原合成默认不分块，两个条件中介及概率平均核心不变。 |
| Datasets.py | 消除两套重复窗口处理，新增未平衡总体与轻量索引子集；保留被试内归一化、基线/通道、窗口和标签定义；修复valence标签传递、NumPy/Tensor兼容和来源丢失。保留原数据集/loader构造名及窗口payload主要键。 |
| ModelTrain.py | 统一LOSO train/test二分流程、固定种子、事实损失、参考总体、缓存失效、诊断和checkpoint；不增加validation。删除不再被调用的重复训练/PCA绘图分支。 |
| mainNeuroFDAN.py | 实际调用最终NeuroFDAN，中文CLI；保留原常用参数名并增加fold、参考、模式、输出与接口检查参数。 |
| CalculateResult.py | 支持显式第二阶段结果路径，跳过未完成的全零行，拒绝静默统计smoke目录；默认正类0与旧统计脚本相同。 |
| loss.py | 仅修复CrossEntropyLoss的weights拼写为weight；旧损失数学定义和权重不变。新真实训练使用Phase 1损失，不将旧损失重新命名。 |
| requirements.txt | 明确torch>=2.0，保留项目其他依赖；TensorBoard用于实际运行记录。 |
| .gitignore | 补充虚拟环境、临时文件、缓存与ZIP排除。 |
| tests/test_phase2_integration.py | 临时数据格式、来源、归一化、模型/梯度/模式、缓存、选模、checkpoint、续训和矩阵汇总测试。不是生理实验。 |
| README.md及本报告、phase2_server_guide.md | 当前运行入口、模型边界、服务器命令及待验证内容。测试/静态日志单独保存在docs。 |

## 实际骨干与SCM接口

基线最后一个NeuroFDAN才是实际生效模型：三层分组Conv1d、CNS邻接混合、联合LIF、原型注意力、分类器。较早因果时序卷积/GNN版本被覆盖且在中介前提前返回，不能误称其参与了原训练。

本阶段保留**实际生效的三层分组时序卷积和CNS图混合**。没有无依据启用历史未执行的ANS GNN或替换为另一种骨干。旧因果卷积、DAGGNN、SpikingMediator、VectorizedLIFLayer及surrogate基础定义仍保留，供研究代码使用；当前骨干的对称padding不能宣称严格时间因果，通道图不保证无环，也不等同于高层SCM。

必要且明确的行为调整：

1. 保留BatchNorm模块、参数、running统计buffer及可学习仿射参数，但在训练中冻结运行统计。新模型初始统计为均值0、方差1，不用同batch样本/时间点重新计算统计，避免其他样本参考表示依赖当前输入。这与基线训练态BN行为不同。
2. 训练保留原hard Gumbel采样图。评估使用固定seed42、保存在checkpoint中的Gumbel扰动，再按当前权重取硬图并去自环；保证批次、缓存和重复评估一致。不能仅取初始单位阵权重argmax，否则去自环会把全部CNS特征归零。
3. 原共享LIF/原型分类路径替换为Phase 1两分支概率中介；原型不作为参考输入。既有LIF方程和surrogate函数由Phase 1直接复用。

| 接口 | DEAP | HCI |
|---|---|---|
| CNS原始窗口 | [B,32,768] | [B,32,768] |
| ANS原始窗口 | [B,3,768]：2EMG+GSR | [B,4,768]：3ECG+GSR |
| CNS骨干输出 | [B,96,128] | [B,96,128] |
| ANS骨干输出 | [B,96,12] | [B,96,16] |
| 完整参考维度ref_dim=0 | 13440 | 13824 |
| 结果模型输入，D=64 | 13504 | 13888 |

骨干的96个时间位置完整保留，在Phase 1已有显式展平适配器处进入条件模型，没有新增池化。Phase 1曾测试的1120/1152维是旧未执行GNN的候选接口，不是当前有效骨干的真实维度。若ref_dim>0，会额外投影，属于压缩对照，不能声称保留充分识别信息。

CNS参数为[B,64]、样本[B,S,64]；ANS参数和样本均为[B,S,64]。采样CNS中介通过W_c m_c+b_c作为ANS LIF附加电流，未detach。ANS没有当前CNS encoder特征入口。OutcomeHead只接收m_a和g(x')，没有m_c直接输入。

联合处理X=(X_c,X_a)的唯一SCM不变：U→X,Y，X_c→M_c，X_a→M_a，M_c→M_a，M_a→Y。M_a单独也可满足给定SCM前门条件，不称联合中介为最小集合。

核心只在frontdoor_scm.py实现：分层采样→配对参考→[B,S,K,2] logits→softmax→K平均→S平均。K分块只是内存优化，最终仍拼回K轴后平均概率。

输出同时保留probs_do、logits_obs/probs_obs、logits_factual别名、各中介参数/样本/KL，以及mediator_statistics、reference_statistics。full模式的训练诊断、测试和checkpoint选择统一用probs_do；none按Phase 1定义用一次事实路径probs_obs，不执行参考积分。no_cns2ans是估计器消融，不是对真实生成机制的物理干预。

## 数据与参考总体

沿用根目录Dataset/DEAP-python或Dataset/HCI-python，数字被试文件1.npy等。虽后缀npy，仍用原pickle字典格式，字段data和labels。HCI同时支持CPU Tensor和NumPy。

- DEAP要求40个trial、至少37个通道、8064点；丢弃384点基线，选择EEG 0:32、EMG 34:36、GSR 36；每trial产生10个768点窗口。
- HCI要求20个trial、至少36个通道、3840点；选择EEG 0:32、ECG 32:35、GSR 35；每trial产生5个窗口。文件应已完成原有30秒截取，本读取器不自行从未知原始长记录重新截取。
- labels第0列valence、第1列arousal；>5为1，否则0。
- 保留原zscore默认axis=0，即被试内部沿trial轴、每通道每时间点归一化；不混合训练与测试被试统计。零方差位置归零，修复原NaN；实际数据以float32存储，输入非有限值明确报错。
- 测试被试同样使用该被试自身完整trial的无标签统计，属于保留的离线被试内预处理，不能宣称在线未知被试归一化验证或SCM假设已成立。
- subject_id从1计数，trial_id/window_id从0计数，dataset标识随每个窗口传递。

分类优化默认保留原trial级下采样：两类各取较少类数量，再截断到DEAP的40/HCI的20个trial整组并随机打乱，所有配对信号、标签及真实来源共用同一索引。DEAP正确按当前任务选择平衡列，不再误用arousal。若截断为空，明确报错，可由用户显式--no_balance，不静默改变规则。

Reference Bank从**当前fold全部未做标签下采样的训练窗口**抽取，最多N_ref个，无放回选择库成员；不足时使用全部并记录shortfall。积分从该经验库有放回抽K次并按来源排除当前窗口。没有validation集合，held-out测试输入绝不进入库。

训练原始参考保留在CPU，Phase 1 ReferenceBank负责唯一抽样及来源检查；轻量适配器按当前骨干编码抽到的配对窗口，保留骨干/ReferenceEncoder梯度。评估每epoch使用当前权重重建特征缓存，缓存记录模型身份、权重版本和来源；更新或加载权重后必须重建，过期缓存拒绝使用。缓存不写入checkpoint。

fold.json及reference_sources.json保存训练被试、held-out、总体/库样本数、实际来源被试数、短缺与检查结果、dataset/trial/window。构造总体与缓存均核验来源；错误来源直接报错。训练loader只构造一次数据，保留trial组打乱；drop_last=True，丢弃数量明确记录，参考总体不随此丢弃。

## 损失和选模协议

真实数据无中介标签、U或干预oracle概率。采用：

L = lambda_obs * CE(logits_factual,y)
    + beta * min(epoch / warmup,1) * mean_B(KL_c + mean_S(KL_a))
    + lambda_fire * L_fire。

KL按中介D求和；CNS不重复计S；ANS对S平均后再与CNS相加，对batch平均。默认lambda_obs=1、beta=0.001、warmup=10。使用双logits、未加权事实CE，避免把旧pos_weight=0.5的BCE校准含义误移植到干预概率。未计算测试标签类别权重，默认也不额外计算训练类权重。没有观察标签对probs_do的NLL，未调用合成oracle监督。

旧rec_loss约束原型重构发放率，div_loss是原型使用熵，依赖已经不属于新估计器的原型库，因此在新路径不调用；没有把KL改名为这两项。旧定义完整保留在Phase 1保护分支。

保留既有目标发放率0.05：L_fire = 0.5 mean_H[(mean_B rates_c-0.05)^2] + 0.5 mean_H[(mean_{B,S} rates_a-0.05)^2]。H是Phase 1 LIF神经元数，默认128。原单共享SNN变成两分支后取分支平均，不能称与原单分支正则完全等价。默认HCI权重1、DEAP权重0，与旧实际训练调用范围一致；可显式--fire_weight覆盖。

保留Adam lr=1e-4、betas=(0.9,0.999)、eps=1e-8、weight_decay=0.01、amsgrad=False，默认100 epochs、batch40、无scheduler、梯度裁剪1。默认Phase 1中介64、隐藏128、dropout0.3、S/K训练4/16、测试16/64、N_ref2048。固定seed42在Dataset/模型/优化器初始化之前设置，覆盖Python、NumPy、Torch和可用CUDA；worker与sampler有明确生成器。

**LOSO只有train/test。** 每fold重建模型、参考库、优化器；训练用全部非held-out被试，测试为唯一held-out。没有额外validation。沿用ACC上升且>0.5保存，ACC相等时F1严格上升保存；DEAP正类1，HCI正类0（旧代码覆盖后的实际含义）。该策略存在测试集选模偏差，不宣称独立无偏泛化评估。

原规则可能整fold均无合格最佳结果。此时只保存last.pt，明确has_eligible_best=false，不伪造best.pt或把全零混淆矩阵当成已完成被试结果。不能仅汇总有结果的被试就宣称完整LOSO；需同时检查所有fold状态。

新run不读取旧全局混淆矩阵来初始化最佳指标。基线DEAP曾用200作为恢复旧矩阵的分母，但实际测试有400个窗口；现在ACC始终按实际计数计算，续训从新checkpoint恢复指标，保留的是选模比较规则而非该分母错误。

## 结果与checkpoint

输出分离到output_dir/dataset/label/seed42/run_name/foldXX。每fold有fold.json、reference_sources.json、history.jsonl、TensorBoard、last.pt，满足原规则时还有best.pt、旧格式命名NeuroFDAN-数据集-任务-被试-正确数.pth、best_predictions.json、best_confusion.npy。run目录有跨被试混淆矩阵，Linux加锁及原子替换避免不同fold覆盖；同一fold/run不要多进程同时写。

新checkpoint是包含模型、优化器、配置和来源的结构化字典，不再是裸state_dict；用load_checkpoint并传map_location加载，不自动把旧单logit/原型模型权重当成新模型。续训限同一输出run及fold，核验来源/模型/预测模式/接口检查设置；继续epoch必须更大。续训固定seed但不承诺与未中断训练逐位一致，因为没有恢复所有worker/sampler随机状态。

TensorBoard记录事实CE、KL、两中介均值/方差及跨样本均值离散程度、发放率、调整概率分布、参考数/短缺和路径消融配置；均值离散程度<1e-3只提示检查塌缩，不是统计识别检验。

## 检查与待服务器验证

实际测试结果见phase2_tests.txt，静态/import/CLI/差异检查见phase2_static_checks.txt。测试覆盖原Phase 1条件采样/概率平均及新的数据结构、来源、模型、真实参考梯度、CPU优化器step、缓存失效、checkpoint加载/续训、原选模规则和汇总。

实际整套47项：46项通过，1项CUDA测试因无GPU跳过；随后补充来源约束和none统计后，受影响的23项第二阶段测试再次全部通过。Python编译、模块导入、中文帮助、依赖一致性、git diff --check通过。日志中的临时矩阵数值仅是统计脚本格式测试，不是正式分类成绩。

临时HCI数组文件和DEAP读取结构夹具仅验证代码；仅运行有限batch，带smoke=true，不是DEAP/HCI分类实验。正式入口不会生成替代数据；单fold smoke仍需服务器完整的该fold训练文件。

未验证：真实文件格式与数值分布、服务器依赖/CUDA版本、GPU显存和吞吐、完整LOSO及消融准确率、真实潜在中介充分性与可识别性。其他对比方法源码不在本仓库，未核验其seed或checkpoint策略，需服务器统一执行。参考表示、潜在中介和SCM假设仍需研究证据，公式实现不等于真实生理因果效应已无条件识别。
