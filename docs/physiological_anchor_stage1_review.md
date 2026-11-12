# 生理锚点首批结果：版本核验、损失边界与锚点风险

## 1. GitHub代码确已发布，但未合并main

重新核验远端feat/neurofdan-physiology-anchors，SHA为a3f458316cc809e07fb4764e9c26143eb5f33456。该提交中7个新增文件均真实存在，包括docs/physiological_anchor_design_and_server_guide.md。三份服务器产物也都记录该源码、detached HEAD、无已跟踪修改，说明服务器实际执行了独立分支代码。

核验时远端main为1c956ccd8611db293fde8dd6d05e3a147cdfcc4c，其树中不存在上述7个文件，且a3f4583不是它的祖先。main当前已有的PR22合并属于旧文档分支，不是本轮创建或合并。

用户此前明确要求不创建PR、不合并、不改main，本轮遵守这一要求。push一个功能分支不会自动创建Pull Request，GitHub也不保证一直显示Compare & pull request提示。缺少提示不是上传失败。通过分支选择器选择feat/neurofdan-physiology-anchors即可查看；无需合并main也能使用已有工具worktree运行。

直接链接：

- 分支：https://github.com/Cute33/NeuroFDAN/tree/feat/neurofdan-physiology-anchors
- 固定版本指南：https://github.com/Cute33/NeuroFDAN/blob/a3f458316cc809e07fb4764e9c26143eb5f33456/docs/physiological_anchor_design_and_server_guide.md

## 2. 是否增加损失：候选合成实验是，真实训练目前没有

实际新增调用为physiology_anchor_experiment.py:anchor_learning_terms：

\[
L=L_{base}+\alpha_c L_{anchor,c}+\alpha_a L_{anchor,a}.
\]

当前alpha_c=alpha_a=0.1，因此也可在文字中将L_anchor=L_anchor,c+L_anchor,a定义为一组锚定正则。但两个组成、归约、有效掩码、权重和消融必须写清，不能隐藏新增损失。

只有observation_physiological_anchors组计算并加入这两项；observation_only不加。合成oracle_mediator组另有原既定真实中介监督，不能与真实数据训练混同。

ModelTrain.py/mainNeuroFDAN.py/NeuroFDAN.py/frontdoor_scm.py/Datasets.py/loss.py相对本轮基线5a77e1b无修改。真实train_one_epoch仍调用原frontdoor_loss，未调用锚点模块，没有悄悄向真实总损失增加任何项。原checkpoint和正式结果没有更改。

新增有效结构为训练期两个线性读出，Mc→CNS统计、Ma→ANS统计。OutcomeHead、CNS→ANS电流、参考表示和前门积分不变。7个文件包括审查、合成机制、测试、脚本与说明，不等于新增7个模型组件。

合成读出6/8维、D=64，共65×(6+8)=910个参数；若未来采用192/8或192/10的真实读出，则分别为13000或13130个训练期参数。后两个数只是按当前接口计算的拟议接入规模，尚未进入真实模型。读出不参与推断，但训练开销和方法学论证仍增加。

正式方法若采用，需要至少说明锚点选择依据、归一化前来源、训练总体标准化、读出目标、权重及消融；不能说模型完全没变，也不能把新增项包装成识别保证。锚点由X派生，不是独立proxy或中介真值。是否值得保留应取决于干预误差与机制证据，而非术语更有生理意味。

## 3. 三份上传的实际核验

| 产物 | SHA | runs目录后缀 |
|---|---|---|
| DEAP审查 | 90493280ee790e3a9562f8f872bd6f649bba3f12 | 2026-11-12_131252_unknown_unknown_run_1330021b |
| HCI审查 | dfa4575a68c01669ea6e02ef3d575081dafc3b41 | 2026-11-12_131510_unknown_unknown_run_35a260b5 |
| 合成smoke | c034a37293660cc2a027b331c760f4dedc1ef3f3 | 2026-11-12_131803_unknown_unknown_run_33dd1e82 |

逐载荷重算字节数及SHA256，三份均一致，没有跳过项。通用run_summary/run_report缺少不是这些探针的失败。三份都附带同一份13项测试通过日志，耗时0.781秒，只算一套测试，不能合计39项。

DEAP读取训练被试2–32全部12400窗，192/8维；HCI读取2–24全部2300窗，192/10维。没有读取held-out1文件，没有访问或使用labels字段。pickle会反序列化整份字典，不能把后者说成文件中不存在标签。所有名义维度都有有效样本和非零训练变化；completed=true、training_executed=false。

smoke完成两机制×三条件、每个1轮，共6个小模型；completed=true、smoke=true、real_data_training=false。各机制内三组X/Y和初始模型指纹一致。候选组训练字段为X/Y及从X派生的c/a/mask，不含M/U/do真值。独立直接/前门真值MAE为0.007800/0.009641，满足此次粗MC工程容差；不称高精度验证。1轮学习误差不能用于宣称方法改善或决定最优权重。

没有发现本批测试/执行的明确实现错误；这不是所有潜在问题均已排除。云端本轮仅审阅源码与已上传记录，没有重新执行模型、测试或训练。

## 4. 新证据：有效特征不等于适合学习的生理目标

被试间方差占比由Σ_s n_s(μ_s−μ)^2/Σ_s n_s除以总体方差计算，仅是描述性分解。被试ID不等同于真实U，高占比也不证明个体差异毫无生理意义；但它说明锚定目标可能主要鼓励个体尺度保留。

| 特征组 | DEAP：被试间占比中位数 | HCI：被试间占比中位数 |
|---|---:|---:|
| EEG log总功率，32维 | 0.8053 | 0.6489 |
| EEG相对功率，160维 | 0.4292 | 0.4747 |

对应原始/原归一化输出的锚点Pearson相关中位数：DEAP总功率0.3481、相对功率0.9862；HCI总功率0.5178、相对功率0.9800。相关接近1不是充分性证明，相对功率仍有被试差异。

HCI三路ECG的6个幅值/导数锚点，被试间占比0.9105–0.9932；原始/归一化后的对应统计相关为−0.3478至0.1198。DEAP GSR_log_std占比0.8204、对应相关−0.2150。

因此，将这些原始幅值统计作为已标准化骨干必须重建的目标存在风险：可能鼓励身份/采集尺度信息，也可能要求恢复预处理已削弱的信息。不能从低相关直接证明非线性不可预测，亦不能直接称数据或预处理错误。

HCI原始GSR均值/斜率/导数的数值尺度很大，目前没有单位校准、设备增益或预处理来源证据，不能直接解释为皮肤电导物理量，也不能擅自判定通道错位、取绝对值、裁剪或修改现有数据。

## 5. 当前推进决定

接口闸门通过，但生理目标的合理性闸门尚未通过。上一版指南第9节24个20轮合成拟合条件暂不执行；更不接入真实训练或启动LOSO。这一暂停来自实际锚点审查的新证据，不是要求补跑旧baseline。

下一步应收窄和论证锚点，而非再增加损失种类。优先比较已明确的幅值缩放不变统计：已有EEG相对频带功率，以及波形导数相对RMS等候选。其理由是限制采集增益尺度作用，不是默认消除真实U。EMG幅值也可能包含任务信息，不能仅因有被试差异就自动删除；每一项保留或移除都需要科学依据与可反驳对照。

若采用无量纲统计，仍保留同一组anchor工作损失和原SCM，不因特征调整继续叠加新分类头或新正则。需要显式修改候选规范、增加幅值/偏置变换的已知不变性测试，并保留本次原始锚点版本作对照。未论证前不改预处理，也不声称新的统计必然有效。

本轮只交付审查与澄清，无新增运行代码，无需服务器重跑、额外上传或合并。三份结果均已收到。即使未来候选合成结果改善，真实潜在中介识别、固定fold验证和test-selected偏差仍需分别讨论。

本报告保存在独立文档分支；代码分支feat/neurofdan-physiology-anchors继续保持a3f4583。没有创建PR、合并或修改main。
