# 支持审计结果复核：输入信号、采样噪声与潜在中介识别边界

## 1. 固定证据及执行状态

- DEAP：8a368c15ad5e1e9fd5f909a99547dd90c3d11f98，runs/2026-11-11_140435_unknown_unknown_run_4d0fa629。
- HCI：cfcdbdff5696ed0061a5ef410dca5c1c6820da70，runs/2026-11-11_140543_unknown_unknown_run_8e5b753f。
- 两份记录的探针源码为c6fbb61f41bb956a64ae2a8d8a4152cbfa420e49；原训练源码为446b0b591a216a5718baaca69a76f9af63bcb08d。
- 两份manifest各四个载荷的长度与SHA256均一致，无跳过项。各缺run_summary.json、run_report.txt、config.json；本操作的必要结果是probe_summary.json与显式附加的日志，缺少上述可选文件不代表失败。
- 两份服务器日志分别记录5项数学测试通过，随后探针完成。不是云端重新执行测试。
- 两份记录均为epoch20，model_state_unchanged/checkpoint_file_unchanged/parameter_grad_fields_unchanged为真，optimizer_created/training_executed为假。
- 查询16、事实参照128、距离校准64、参考抽取64；参考有放回抽取后均为63个唯一来源，符合原定义。四组来源互不重合，无测试被试1。此检查针对已保存的探针来源，不代替原2048成员参考库的完整检查。

本轮只读Git对象并整理数值，未访问真实数据、读取权重或运行模型/CLI/单元测试/训练。新增脚本只使用标准库；修改不涉及模型、损失、数据、训练器、选模协议、main或原诊断分支。

## 2. 支持检查：没有发现明显恶化，但不能证明真实overlap

| 指标 | DEAP | HCI |
|---|---:|---:|
| 自身条件下低于独立5%分位的比例 | 5.47% | 4.30% |
| 交叉条件下低于参考专属5%分位的比例 | 6.26% | 5.62% |
| 事实组合最近邻平均距离 | 1.9253 | 2.2774 |
| 交叉组合最近邻平均距离 | 1.9192 | 2.2610 |
| 交叉距离超过评估事实校准95%分位 | 8.05% | 1.29% |

当前有限样本和度量下，交叉组合没有表现为明显比事实组合更远或更低密度。没有统计置信区间，也没有建立接受/拒绝规则，不能把8.05%或5.62%解释为显著或通过理论检验。

密度由已学习q计算，最近邻使用高维骨干表示及有限事实参照。后验弱输入依赖时，本来就可能让任意中介/参考组合看起来相容。不能据此排除真实支持不足、支持充分条件未满足或条件结果模型拟合错误。

## 3. CNS和ANS的均值、方差都不是严格输入无关

下表的变化尺度为sqrt(mean_d Var_X(parameter_d))，即跨输入std的RMS；不是各维std的算术平均，不是采样内std。仅代表所抽训练窗口。

| 统计 | DEAP | HCI |
|---|---:|---:|
| CNS均值跨输入变化尺度 | 0.000663 | 0.020998 |
| CNS sigma跨输入变化尺度 | 0.000361 | 0.009285 |
| CNS平均posterior sigma | 0.992171 | 0.964635 |
| CNS平均条件KL（64维求和） | 0.004836 | 0.151562 |
| 固定Mc，ANS均值随Xa变化尺度 | 0.016292 | 0.009866 |
| 固定Mc，ANS sigma随Xa变化尺度 | 0.008201 | 0.004975 |
| ANS对Mc积分后，均值随X变化尺度 | 0.006399 | 0.005173 |
| ANS对Mc积分后，总std随X变化尺度 | 0.003145 | 0.002730 |
| ANS平均条件sigma（共同CNS噪声网格） | 0.996944 | 0.994193 |

DEAP CNS非常接近弱输入依赖的先验形态；HCI CNS有更明显变化。ANS的均值与sigma都保留非零输入依赖，不能写成“两个中介完全没有信息”或“只有mu变化，sigma不变”。输入参数变化远小于单位量级采样std也不单独给出互信息估计；本表不是正式SNR或信息量证明。

ANS全方差分解中，E_Mc[conditional variance]分别为0.99656/0.99693；Var_Mc[conditional mean]分别为0.009414/0.031553；Var_X[E_Ma]分别为0.00004095/0.00002677。当前条件均值随Mc的波动明显大于边缘均值跨输入差异。有限MC矩估计存在误差，这些不等维统计不应机械换算成“噪声贡献百分比”或因果路径贡献。

## 4. Head没有完全忽略Ma；积分输出仍接近平坦

尺度归一输入梯度L2的平均值如下。它们不是模块贡献占比，也不是因果效应。

| 位置/输入块 | DEAP | HCI |
|---|---:|---:|
| 事实Ma | 0.004384 | 0.006694 |
| 事实参考CNS | 0.004059 | 0.005131 |
| 事实参考ANS | 0.002393 | 0.003831 |
| 交叉Ma | 0.004429 | 0.006692 |

Ma梯度非零，且并不比参考块低多个数量级。因此“Head已经完全忽略Ma”的强判断不受本次结果支持。非零梯度可能主要响应Ma中的采样波动，并不证明它使用了与类别有关的稳定输入信息，或学到了真实条件结果。

| 类别1概率（16个训练查询窗口） | DEAP | HCI |
|---|---:|---:|
| 自身参考、S次概率平均：均值 | 0.479051 | 0.501899 |
| 自身参考、S次概率平均：跨窗口std | 0.024216 | 0.008727 |
| 公共参考、S/K积分：均值 | 0.486101 | 0.505458 |
| 公共参考、S/K积分：跨窗口std | 0.001035 | 0.001890 |

差异明显收缩，但跨窗口std还包含有限S的MC误差，不是纯输入效应。自身参考结果是S次平均，不等同于训练CE/原日志事实ACC使用的第一次采样。这里没有完整测试集评估，不能计算或更新正式ACC、F1、checkpoint。只有均值/std不能推出每个查询的预测类别；不能据此伪造探针混淆矩阵。

## 5. 优化与参考总体的差异已得到数量证据

| 数量/比例 | DEAP | HCI |
|---|---:|---:|
| 未平衡训练窗口[类0,类1] | [5270,7130] | [1450,850] |
| 优化窗口[类0,类1] | [5200,5200] | [800,800] |
| 类0保留比例 | 98.67% | 55.17% |
| 类1保留比例 | 72.93% | 94.12% |

标签选择概率若为a_y，则一般有p_opt(y|m,x) ∝ a_y*p_pop(y|m,x)。原算法是trial级有限下采样，上表是实现后的保留比例，不是证明选择仅由y决定的总体概率模型。参考抽取成员未进入优化集的比例也不能直接叫“OOD比例”：仍可能处于相同特征支持域。

仅变换二分类截距、逆概率加权或取消下采样，都不能自动修复潜在中介识别问题；本轮没有采取任何一项。后续需先明确所估计的条件结果和参考总体，再论证相应选择模型及修订。

## 6. 真正需要补足的证据：无中介标签时的学习识别

重审synthetic_frontdoor_experiment.py的fit实际路径：nll_c使用真实mc；ANS条件输入使用真实mc；nll_a使用真实ma；OutcomeHead事实CE也使用真实ma。Phase 1文档原本明确了这一监督边界。

因此Phase 1 learned结果验证了“有合成中介真值监督时的条件模型学习和积分”，不能用来证明Phase 2仅有X/Y、重新采样中介、事实CE+KL的学习方案也能恢复同一机制。它不是错误数据或无效工作，但两类证据不能互相替代。真实路径的发放率项并未提供真实中介—结果联合观测。

在当前完整事实表示Z下，人为增广分布为p_opt(z,y)q_phi(tilde_ma|z)。其事实CE允许Head忽略tilde_ma，而KL也不提供真实中介的测量锚点。这个理论允许的退化与本次弱输入依赖/平坦积分输出相容，但非零Ma梯度表明不能宣称已经严格达到该总体最优；训练/评估图差异等其他因素仍可能同时存在。

### 为什么不能只凭X/Y和当前DAG宣称一般识别：一个解析反例

以下只检验“给定DAG及外生独立性是否足够”，不替换项目SCM、网络、既有tanh合成生成器或论文研究目标。U及所有epsilon均相互独立标准正态。两模型共享：

Xc=U+epsilon_c，Xa=U+epsilon_a，Mc=Xc+epsilon_mc。

模型A：Ma=Xa+Mc+epsilon_ma；Y*=Ma+U+epsilon_y。

模型B：Ma=(2/3)Xa+(2/3)Mc+epsilon_ma；Y*=Ma+2U+sqrt(5/9)epsilon_y。

两模型均令Y=1[Y*>0]，Y*只用于写出阈值函数的解析表达，不是新增SCM节点。两模型都严格保留规定的边，无Mc→Y或X→Y直接边。它们的(Xc,Xa,Y*)都是零均值联合高斯，且具有同一协方差矩阵：

[[2,1,4],[1,2,4],[4,4,14]]。

所以观测到的(Xc,Xa,Y)分布也完全相同。但：

- A：P(Y=1|do(xc,xa))=Phi((xc+xa)/2)。
- B：P(Y=1|do(xc,xa))=Phi(((2/3)(xc+xa))/sqrt(6))。

二者一般不等。中介如果被观测，前门公式分别成立并可以识别各自效应；只观测X/Y时，DAG及当前独立性假设本身不能区分两模型。固定、已知参数的合成机制当然不受这个未知机制的反例否定。若另加可识别测量/生成假设，必须给出定义和论证，不能默认分类CE或一个神经网络参数化已经提供了它。

## 7. 下一阶段的研究准入条件

目前已足够确定工作优先级，无需再开一轮相同20/100轮或完整LOSO，也不安排新的模型探针。需要先完成“真实中介不可观测时，哪些现有观测/测量/结构限制能识别所需条件分布”的方法学设计；若当前信息不够，应明确列出缺口，不能自行补造假设。

任何后续训练方案应先写清目标总体与损失，并说明为何排除上面的不可区分机制或对识别只作条件性声明。在相同无中介监督的合成设置中验证时，真值Mc/Ma/U/干预概率仅供独立评价，不能进入训练或选模。继续保留有中介监督的Phase 1作为独立条件模型基准，不覆写其结果。有限生成器上的成功仍不等于一般非参数识别证明。

本轮不擅自决定删除参考表示、对do使用观察标签、改SCM、替换骨干、增加伪交叉标签或改变标签平衡。分类结果作为服务器诊断存在，不改写为已可靠恢复的真实因果效应。

## 8. 可选的服务器结果整理与上传（只用标准库）

无需重跑模型。如果需要在服务器保存同一复核记录，在已经激活的dslenv窗口执行。该操作不读last.pt、不读DEAP/HCI、不导入torch，使用已上传的两个固定快照。新目录只保存整理结果，不能当成新模型实验。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
REVIEW_TOOLS=/home/fss/NeuroFDAN-support-review-tools
git status --short
git worktree list
git fetch origin refs/heads/feat/neurofdan-support-results-review:refs/remotes/origin/feat/neurofdan-support-results-review
git fetch origin server-diagnostics
if [ ! -e "$REVIEW_TOOLS" ]; then
    git worktree add --detach "$REVIEW_TOOLS" origin/feat/neurofdan-support-results-review
fi
test "$(git -C "$REVIEW_TOOLS" rev-parse HEAD)" = "$(git rev-parse origin/feat/neurofdan-support-results-review)"
test -z "$(git -C "$REVIEW_TOOLS" status --porcelain)"
REVIEW_OUTPUT=/home/fss/NeuroFDAN/outputs/phase5_probes/support_results_review_001
mkdir -p /home/fss/NeuroFDAN/outputs/phase5_probes
mkdir "$REVIEW_OUTPUT"
bash "$REVIEW_TOOLS/scripts/phase3_run_logged.sh" "$REVIEW_OUTPUT/train.log" \
  python "$REVIEW_TOOLS/scripts/review_support_snapshots.py" \
  --repository /home/fss/NeuroFDAN --output_dir "$REVIEW_OUTPUT"
cat "$REVIEW_OUTPUT/train.log.exit"
)
```

预览与正式上传分别执行；仍使用原上传器，目标为server-diagnostics独立目录，不覆盖既有记录。

```bash
cd /home/fss/NeuroFDAN
bash /home/fss/NeuroFDAN-support-review-tools/scripts/upload_run_diagnostics.sh \
  /home/fss/NeuroFDAN/outputs/phase5_probes/support_results_review_001 \
  --log /home/fss/NeuroFDAN/outputs/phase5_probes/support_results_review_001/train.log \
  --dry-run
```

```bash
cd /home/fss/NeuroFDAN
bash /home/fss/NeuroFDAN-support-review-tools/scripts/upload_run_diagnostics.sh \
  /home/fss/NeuroFDAN/outputs/phase5_probes/support_results_review_001 \
  --log /home/fss/NeuroFDAN/outputs/phase5_probes/support_results_review_001/train.log
```

结果文件为support_results_review_001/probe_summary.json及日志。脚本校验并摘录已有服务器日志，不重新执行测试。回传上传器打印的SHA和链接即可；这项可选整理不构成开始方法学分析的前置条件。

## 9. 本次检查记录

云端实际完成：两份固定快照的全部载荷长度/哈希核对、全部JSON浮点值有限性检查、新整理脚本的Python3.10 AST静态语法检查、三组文档命令的bash -n、git diff --check，以及核心模型/数据/损失/训练/合成学习代码无改动核对。解析反例使用有理数恒等式核算，两模型观测Cov(Xc,Y*)均为4、Var(Y*)均为14，干预方差分别为4和6；这是数学核算，不是训练或合成实验。

未在云端执行：新增脚本CLI、任何模型前向/反向、单元测试、真实数据或合成训练、GPU/LOSO。服务器已有5项数学测试通过的结论来自各自上传日志，不能记作本轮云端测试。新增整理脚本尚待服务器执行；即便不执行它，以上直接读取固定Git对象所得复核结论也已成立。
