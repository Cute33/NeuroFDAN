# NeuroFDAN Phase 3 交付报告

## 版本与范围

Phase 2 稳定分支为 `feat/neurofdan-phase2`，基线提交为 `c66b4d3c1aa17453a0d024033b1c8ab85b67cdde`。本轮起始工作区干净，本地与远端 Phase 2 SHA 一致；从该提交创建 `feat/neurofdan-phase3`，不在 Phase 2 上继续修改。

核验时远端 main 为 `0606c3a5cbfdeeda69e6aaab20c5643fe6f52baf`，已通过 PR #3 合并 Phase 2。这与“应该尚未合并”的预期不同；本轮仍严格从指定 Phase 2 提交起步。不向 main 提交或推送，不创建或合并 PR，不打发布标签。Phase 3 的最终 SHA 和推送核验在本次交付回复中给出，避免在提交内部循环引用自身 SHA。

Codex 仅审查源代码、进行轻量测试、整理服务器流程；不访问真实服务器和真实数据，不执行正式训练。

## 修改文件与理由

| 文件 | 修改与原因 |
|---|---|
| docs/phase3_server_guide.md | 新增可复制的环境、数据、单折检查、单折完整训练、四项 LOSO、续训、统计及 Xshell 隧道流程，命令对应真实 CLI。 |
| docs/phase3_checks.md | 记录本轮实际测试、跳过项与未验证范围。 |
| docs/phase3_delivery_report.md | 本报告，记录版本、核验及行为边界。 |
| README.md | 更新开发分支及 main 合并状态，链接第三阶段操作与交付说明。 |
| CalculateResult.py | 保留旧接口，新增按 run 目录审计、导出和完整性验收参数。原脚本仅排除全零矩阵，不能证明 32/24 折完整。 |
| result_audit.py | 新增只读产物审计；对全部预期 fold 核验来源、结束状态、完整测试、历史选模、最佳预测及矩阵，导出逐被试和汇总表。 |
| ModelTrain.py | 在主输出进入训练更新/测试指标前检查概率形状、有限性、范围和归一化。此前 argmax 可能将异常概率转成类别并继续记指标；合法输出的值、梯度和损失不变。 |
| scripts/phase3_run_logged.sh | 用 tee 追加终端输出并记录程序及 tee 的退出码，避免管道掩盖失败。 |
| tests/test_phase3_support.py | 增加 10 项 CPU 产物、概率与 CLI/日志接口测试，无真实数据训练。 |
| .gitignore | 忽略指南可能新建的 .venv-phase3 等环境目录，防止误上传。 |

没有修改模型结构、损失定义、数据预处理、标签、输入维度、默认 S/K/N_ref、随机种子或 checkpoint 选择规则。训练部分只有异常概率提前报错；统计新增严格审计，不重写训练产物、不重新选择 checkpoint。没有提交真实数据、权重、虚拟环境、缓存、训练日志或 ZIP。

## Phase 2 代码核验

| 核验项 | 当前有效实现与结论 |
|---|---|
| 模型定义 | NeuroFDAN.py 只有一个有效 NeuroFDAN；mainNeuroFDAN → ModelTrain.run_fold → 该模型。 |
| 前门实现 | 直接复用 frontdoor_scm.FrontDoorEstimator；没有重复另一套积分器。 |
| 联合处理 | CNS/ANS 成对输入，同一窗口构成 X；不作为两个独立识别问题。 |
| CNS→ANS | 采样 m_c 经可训练电流投影驱动 ANS LIF，保留梯度；no_cns2ans 切断该路径。 |
| 结果头 | 只输入 m_a 和参考联合表示，不直接输入当前 m_c。 |
| 概率积分 | 每个中介/参考组合先 softmax，再对 K 和 S 平均；分块计算不改变公式。 |
| 真实骨干 | 保留现有分组时序卷积与 CNS 图混合，CNS 特征 [B,96,128]，ANS [B,96,12/16]，完整联合参考维数 DEAP 13440、HCI 13824。 |
| 参考总体 | 先从全部未标签平衡训练窗口抽取参考库；优化用的平衡子集独立。参考库保存 dataset/subject/trial/window，排除 held-out。 |
| 数据来源 | Dataset、窗口索引、DataLoader、参考来源及预测结果保留元数据；CNS/ANS 不拆散抽样。 |
| DEAP | 32 人，40 trial，去 384 点基线后每 trial 10×768 点；32 EEG、2 EMG、1 GSR。valence 使用第 0 列正确标签进行 trial 平衡，arousal 使用第 1 列；阈值 >5。 |
| HCI | 24 人，20 trial，文件已截 3840 点，每 trial 5×768 点；32 EEG、3 ECG、1 GSR；支持 NumPy/CPU Tensor，转换 float32 并核验有限值。 |
| 归一化 | 沿被试内 trial 轴使用无标签统计，零标准差置零；held-out 自身统计仍被使用，不是严格在线未知被试评估。 |
| 随机性与设备 | seed42 在每 fold 数据、模型与加载器初始化前设置；CUDA 调用有可用性判断，checkpoint 使用 map_location。 |
| 保存与续训 | last/best 为格式版本 2 的结构 checkpoint；resume 限同一 run 单 fold，并检查模型配置和来源；已完成末轮但缺 summary 的异常需反馈，不能擅自增轮数绕过。 |
| 选模 | 无 validation；held-out ACC>0.5 后按 ACC 升高、相同 ACC 时 F1 严格升高保存，DEAP 正类 1、HCI 正类 0；无合格 best 不伪造。 |
| 模式 | full 的主输出是 probs_do，none 使用同一中介/结果头的事实 probs_obs，不执行 K 参考边缘化；评估和选模同口径。 |

上述是源代码核验及轻量接口验证，不是服务器真实数据运行结论。

## 损失和统计协议

真实数据损失继续为

\[
L=\lambda_{obs}L_{obs}+\beta\min(e/10,1)(KL_c+KL_a)+w_{fire}L_{fire}.
\]

这里 beta=0.001、lambda_obs=1；KL 对潜变量维求和、样本维平均，ANS 还对条件采样维平均。发放率约束沿用 Phase 2，DEAP 默认权重 0、HCI 默认 1。观察标签用于事实分类监督；没有 oracle 中介标签或干预概率监督，不直接监督 probs_do。loss.py 的历史损失定义保持不变；依赖旧原型模块的 reconstruction/diversity 项仍按 Phase 2 不参与新训练，不把 KL 改名为 reconstruction。

统计矩阵行是真值、列为预测。DEAP 正类 1，HCI 正类 0；ACC、Sensitivity、Specificity、F1 先逐 held-out subject 计算，再取均值与 np.std(ddof=0)。标准差不是多随机种子统计。

新审计区分未运行、未完成、完成但无合格 best、完整且合格、smoke 和产物不一致。失败与中断必须结合退出码、进程及日志确认，不能仅凭目录缺失推断原因。--require_complete 仅在全部 32/24 折均有完整且一致的合格 best 时成功；不完整时仍导出明确标记的诊断并返回非零。无合格 best 的折不能填零指标或用 last 冒充 best。

产物审计不运行 checkpoint 推理、不核验真实输入或权重内容；报告明确 checkpoint_contents_verified=false。均值只覆盖合格子集时标记非完整 LOSO，不能直接作为完整论文表。

## 实际检查与服务器执行

本轮 57 项测试中 56 项通过、1 项 CUDA 测试跳过；Python 编译、核心 import、CLI、Bash 语法和 Git 空白检查通过。详见 [实际轻量检查记录](phase3_checks.md)。

完整服务器命令集中在 [Xshell/Linux 操作指南](phase3_server_guide.md)，包括已有环境复用、新环境安装、精确数据文件检查、GPU/CPU smoke、单折完整训练、DEAP/HCI × arousal/valence 四项正式命令、单折 resume、四项统计导出和 TensorBoard 隧道。服务器项目目录统一为 /home/fss/NeuroFDAN，数据根显式 /home/fss，正式输出显式 outputs/phase3。

待服务器验证：真实文件的全部形状/内容、真实 smoke、服务器 GPU/驱动与 PyTorch 兼容性、显存与耗时、正式单折和完整 LOSO、服务器实际中断续训及论文效果。云端临时夹具和随机 tensor 测试不能替代这些项目。

真实中介不可观测，潜在分布、参考总体和 SCM 假设仍限制解释；前门积分实现不自动证明真实中介可识别。no_cns2ans 是估计器消融，非物理干预。held-out test 参与选模存在选择偏差，不能称独立无偏泛化，也不能未经核实其他方法口径就宣称公平。合成 oracle 数值精度不等于真实数据因果识别精度。
