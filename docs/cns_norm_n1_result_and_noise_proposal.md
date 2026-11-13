# CNS入口归一化N1负结果与采样噪声预热提案

本文是结果审查和待确认的实验设计，不是已经实施的训练改动。当前没有新的噪声预热CLI开关；不能据此启动预热训练。N1未通过，因此不执行原计划中以N1通过为前提的正KL N2或完整fold/LOSO。

## 1. 原始上传证据

- 结果提交：`832fa1bd2472bc8777f3c9ecfe4ee564045ce7e9`。
- 目录：`runs/2026-11-13_130900_unknown_unknown_run_ff863bb9`。
- 源码：`2514db61f5923123794750b0a91a8fa9b24a7094`，无tracked changes。
- 逐项核验manifest：9个载荷的字节数和SHA256均一致；skipped为空，missing仅未生成的通用`run_report.txt`，不是训练缺失。
- 服务器13项配对/接口测试通过。`completed=true`、`passed=false`、`actual_steps=300`，是完成预算的负结果，不是程序中断。
- 查询400窗、标签计数[153,247]、2048参考成员及顺序与原beta0基线相同；原摘要哈希不变。当前none模板与norm初始参数指纹相同。历史原基线未记录初始权重指纹，不能宣称对历史权重逐字节核验。
- 新私有权重已保存在服务器原结果目录的`private_fit_last.pt`，未上传、未覆盖旧权重，不用重新拟合以补权重。

服务器目录：

`/home/fss/NeuroFDAN/outputs/phase6/cns_input_norm/n1_real/DEAP/arousal/seed42/deap_arousal_fold02_cnsln_beta0_fit300_001`

## 2. 分类结果没有改善

step=1/50/100/150/200/250/300，每点3次完整400窗随机评价，主要输出和事实输出均为单类预测。末端主要输出每次均为：

```text
confusion_matrix = [[0,153],[0,247]]
ACC = 0.6175
positive_class_F1 = 0.7635239567
class_0_recall = 0
```

F1是项目原选模口径的正类F1，不是Macro-F1。较高正类F1来自类别构成，不能用它证明区分了两类。这里全部是优化子集拟合成绩，不是held-out subject成绩。

最优常数概率的观察NLL为0.6652747443075995。末端三次NLL：0.6651989817619324、0.6652566790580750、0.6654729247093201，均值0.6653095285097758，仍停留在类别先验附近。P1均值约0.61685–0.61694，跨输入std约0.00170–0.00179。

| CNS统计，均取第300步第一次评价 | 原beta0/none | CNS LayerNorm N1 |
|---|---:|---:|
| 条件mu跨输入逐维std的均值 | 0.003696448 | 0.010696614 |
| 条件sigma跨输入逐维std的均值 | 0.002621138 | 0.005098674 |
| 平均posterior sigma | 0.938887078 | 0.979799235 |
| 400窗主要ACC | 0.6175 | 0.6175 |

归一化改变了条件表示，却未带来分类能力；“仅CNS入口尺度校准足以解决当前拟合问题”的假设在本预算下未获支持。不能将mu变化增加写成中介学习成功。sigma与mu跨输入std的统计含义不同，这个比较不是经过估计的信噪比，更不是信息量。

## 3. 梯度证据及边界

代码路径：

- `frontdoor_scm.py:GaussianMediator.distribution`：drive→LIF发放率/平均电流→mu/logvar。零发放率不等于平均电流路径断开。
- `GaussianMediator.draw`：`mu + exp(0.5*logvar)*epsilon`。CNS与ANS分别重参数化抽样。
- `ANSMediator.forward`：CNS采样先映射成附加电流，再生成ANS条件分布，没有删除CNS→ANS路径。
- `FrontDoorEstimator.marginalize/forward`：交叉结果logits先softmax，再依次平均K和S，主要输出仍为`probs_do`。
- `classification_objectives.py:integrated_observational_nll/classification_terms`：方案A使用观察标签对积分概率做NLL，不是平均交叉组合CE，也不是干预真值监督。
- `scripts/scheme_a_fit_check.py:measure/fit_gate/main`：更新前记录分项梯度，更新后eval记录三次MC。两者模式、抽样和时点不同，不能视为同一随机前向的联合测量。

第300步更新前的任务梯度L2：CNS约0.001535、ANS约0.003382、CNS→ANS约0.001361、OutcomeHead约0.048605。CNS fire梯度约0.001674，与任务梯度余弦约-0.013；ANS fire约0.000253。KL权重确为0。

由此可以排除这些模块在该步任务梯度全部断开，不能证明任务梯度足够或有利。CNS fire仍会影响方向，但不能把全部失败归因fire。也不能仅由posterior尺度断言采样噪声是唯一原因，或认定Head完全忽略中介。现有记录未在固定参数下分离MC梯度方差。

## 4. 下一项推荐：有限预算的采样噪声预热，待用户确认

不继续长训N1；不把失败的CNS LayerNorm默认纳入最终模型。建议返回已经有结果的beta0/none基线，仅比较训练初期采样尺度是否影响优化。

原抽样为：

\[
M_c=\mu_c(X_c)+\sigma_c(X_c)\epsilon_c,\qquad
M_a=\mu_a(X_a,M_c)+\sigma_a(X_a,M_c)\epsilon_a.
\]

拟议训练抽样为：

\[
M_c^{(t)}=\mu_c(X_c)+\tau_t\sigma_c(X_c)\epsilon_c,\qquad
M_a^{(t)}=\mu_a(X_a,M_c^{(t)})+\tau_t\sigma_a(X_a,M_c^{(t)})\epsilon_a.
\]

两个独立标准高斯噪声流、CNS→ANS条件路径、参考成员和参考抽样规则保留；不改mu/sigma数值输出，不强制它们增大/减小。预先只试一条固定路径：

\[
\tau_t=\begin{cases}
0.1,&1\leq t\leq50,\\
0.1+0.9(t-50)/100,&51\leq t\leq150,\\
1,&151\leq t\leq300.
\end{cases}
\]

全部评价从第一步起一律使用tau=1。训练后半程恢复原抽样；不能把低噪声训练时的ACC作为通过结果。tau至少0.1，不把Gaussian直接替换成确定性点质量；这个数是有意提出的优化超参数，不是生理常数或识别条件。

**数学目标：** 用上述抽样得到积分概率`p_hat_tau`，优化

\[
L_t=-\frac1B\sum_i\log \widehat p_{\tau_t}(Y_i\mid X_i)
+\beta_t\{\mathrm{KL}(q_c\Vert p_c)+\mathbb E_{M_c^{(t)}}\mathrm{KL}(q_a(\cdot\mid X_a,M_c^{(t)})\Vert p_a)\}
+\lambda_{fire}L_{fire}.
\]

q_c/q_a在KL中指原参数mu/sigma定义的分布，不是缩放后的q_tau。这是明确的优化代理目标，不声称是缩放分布的ELBO。首次真实对照仍beta=0，KL只记录不更新；若将来恢复正KL，必须单独验证这个目标选择，不能静默变更KL定义。tau=1时恢复当前方案A目标及抽样语义。有限S/K下NLL也有MC误差，不声称该目标给出了无偏干预监督。

它没有新增loss项、分类头或SCM边，但**改变训练初期的条件采样分布，是新的训练策略**，不能说完全没改方法。理论动机是：若随机中介使初期分类梯度不稳定，较小扰动可能帮助条件均值先形成任务相关变化，随后检验该变化能否承受完整采样。这不是效果保证。降低扰动也可能只是暂时抬高ACC，恢复tau=1后仍塌缩。

相较其他候选：

| 候选 | 依据与风险 | 当前优先级 |
|---|---|---|
| 更小lr或更长同配置训练 | 尚不能排除预算影响，但多项完整训练及本次固定子集均停在先验；继续长训缺乏新证据 | 不推进 |
| 再加ANS LayerNorm | 可能增强Xa驱动，也可能丢失生理幅度；N1没有直接检验ANS驱动是否尺度不足 | 暂缓，不连续堆归一化 |
| 采样噪声预热 | 最终抽样和总损失形式可恢复；直接检验低扰动学习是否能转移到完整概率预测，但可能仅有过渡期收益 | 推荐一次预声明、可失败的对照 |

## 5. 确认后的实现与执行范围

目前不实施以下改动：拟议在`frontdoor_scm.py`新增默认关闭的训练采样尺度控制，只由训练脚本显式设置；eval强制尺度1。必要的训练入口和诊断同步传参/记录。`NeuroFDAN.py`、原始数据处理、Head输入、SCM、参考总体和选模规则不改。新实验独立分支、phase6新目录；旧checkpoint及结果不覆盖。

确认后先交付以下两项，不自动运行完整真实fold：

1. **W0合成对照**：复用已知Gaussian SCM及非识别/幅度失败反例，数据和测试样本固定。原观察监督tau=1、观察监督加tau预热、oracle参考三组均300步；原观察none200步的旧结果不能直接当新300步基线。两种观察组训练只能访问X/Y；真中介、U和干预概率仅由独立评价读取，oracle组显式例外。比较观察ACC/F1/NLL/Brier及真实do概率MAE/MSE，不把观察ACC提高等同识别成功。
2. **W1真实小批对照**：仅在W0完成后，原beta0/none与新预热方案对照；旧真实基线不重跑。固定原400窗、参考2048、fold2、seed42、lr2e-5、train/eval400、LIF256、fire0.028和S/K，最多300个私有step。配对守卫只许噪声日程/输出位置变化，起始参数指纹和来源显式记录。默认原模型不启用预热。

每50步记录训练tau、native/effective sigma、分项梯度、mu/sigma输入依赖、发放率及完整噪声评价的ACC/F1、矩阵和NLL；日志明确预热训练与tau=1评价的区别。第250和300步各三次tau=1 MC均达到既定工程门槛（ACC≥0.8、两类最小Recall≥0.6、NLL≤0.6152747443）才认为本次有拟合支持，不提前在低噪声阶段宣布通过。Recall是塌缩守卫，论文主指标仍ACC/F1。

若仅低噪声训练时改善、完整噪声评价仍单类，否定“预热足以解决问题”；停止该路线，不凭早期ACC进入LOSO。若W1通过，再验证原正KL配置和完整训练总体，最后用预定验证协议冻结训练/选模规则后才做LOSO。不能用当前400窗ACC证明泛化或潜在中介因果识别。

## 6. 本轮服务器与Git操作

**本轮无需运行mainNeuroFDAN.py、新探针或训练，也无需重复上传N1。** 上传已完整核验。下一步是确认新的采样策略范围，不是手工补日志或重复实验。

如需在服务器查看本设计，可执行以下只读Git操作；不切换主工作区，不修改模型和结果：

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
if [ -z "${GIT_SSH_COMMAND:-}" ]; then
    export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
fi
timeout 600 git fetch --no-tags origin feat/neurofdan-n1-review-noise-proposal
git show FETCH_HEAD:docs/cns_norm_n1_result_and_noise_proposal.md
)
```

这不是实验命令，不产生新实验结果，因此没有本轮结果上传命令。新训练策略尚未实现，不编造不存在的参数。确认后交付经过静态核对的完整服务器运行器及dry-run/正式上传动作，自动保存日志、退出码、实际步数和upload_state收据，仍只需回传SHA及链接。

需要确认来自用户既有明确约束：“如果要微调模型框架的话，先说明要调整那一部分，先不要对现有的关键py文件进行修改，我们沟通确认可以改后才能修改比如NeuroFDAN.py这类关键文件。”本轮已完成可审查的公式、固定预算、失败条件及文件范围；没有先行改变关键实现。

本轮仅新增本文。云端只读取Git上传JSON、校验哈希和Git差异；没有运行模型、CLI、单元测试、合成/真实训练或GPU。服务器13项测试通过仅指N1原代码；采样预热未实施/未测试。无main改动、PR或合并。
