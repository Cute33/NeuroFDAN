# 事实监督、前门推断与正式ACC汇报口径

## 1. 一个模型，两种概率计算，并非两个独立分类器

当前真实数据代码没有两套分类头或两次独立训练。共享CNSMediator、ANSMediator、ReferenceEncoder和OutcomeHead；只有事实监督CE及KL/fire等既定项更新参数，前门概率由共享条件模型再执行积分得到。

应将此前容易引起误解的“事实训练/前门训练”改称“事实监督/前门推断”。full模式训练前向确实计算probs_do，但观察标签未用于probs_do的分类损失。训练日志里的前门ACC只是对当时计算概率的评价，不意味着该ACC对应一个前门CE。

## 2. 事实概率与事实ACC

联合输入X=(Xc,Xa)经过原骨干，再分层采样Mc~q_c(Mc|Xc)、Ma~q_a(Ma|Xa,Mc)。对当前样本自身联合表示g(x_i)，计算：

q_obs_i(y)=softmax(OutcomeHead(ma_i^(1),g(x_i)))。

实际FrontDoorEstimator.forward使用m_a[:,0]形成logits_obs，softmax得到probs_obs，NeuroFDAN中的logits_factual为同一logits的别名。它只取第一次完整分层采样，是当前实现的观察预测近似，不能称为解析精确的p(y|x)。

真实标签y_i是该生理窗口所属trial的观察情绪标签。事实ACC在指定数据集D上计算argmax_y q_obs_i(y)与y_i是否相等，再除以样本数。它回答模型对观察到的输入和标签有多少判别能力，不检验do因果效应。

q(y|ma,g(x))里的x是条件变量。给条件概率模型输入x并不在最终SCM里新增X→Y结构边；同理，g不是新的因果节点。SCM结果方程仍为Y=f_Y(Ma,U,epsilonY)，OutcomeHead不额外读取Mc。

默认参考表示完整拼接选定骨干输出，不额外投影；这不等于对原始X无损，也不自动成为识别所需的充分统计量。表示充分性、条件分布支持与潜在中介可识别性不能仅由分类ACC证明。

## 3. 前门概率与前门ACC

最终识别表达式为：

p(y|do(xc,xa)) = ∫∫ p(mc|xc)p(ma|xa,mc)[∫p(y|ma,x')p(x')dx'] dma dmc。

实现中采样S组层次中介，再从当前fold未平衡训练窗口参考库抽K个配对联合参考，逐组合计算softmax(OutcomeHead(ma_i^s,g(x'_ik)))，之后沿K/S平均概率形成probs_do。默认测试S=16、K=64，原N_ref=2048；不平均logits替代概率平均。当前输入通过中介影响do结果，参考输入来自经验训练总体，不是原型或测试被试。

前门ACC同样在D上计算argmax_y probs_do_i(y)与同一个观察标签y_i是否相等的比例。它是前门调整输出用于分类任务时的预测正确率；这里没有真实的do结果标签。

因此，前门分类ACC不等于因果效应估计准确率，不是识别定理成立的数值证明。干预分布和观察分布本来就可能不同，即便模型完全正确，也不保证在自然观察标签上do决策比p(y|x)更准确。跨被试上能否更稳健必须实证检验，而不是由前门公式自动推出。

## 4. 为什么真实训练仍需要事实监督

前门公式要求条件结果模型p(y|ma,x')。观察数据中的事实配对为(ma_i,x_i,y_i)，所以用事实CE拟合这个共享结果模型，并使梯度经Ma回到层次中介和骨干。参考边缘化不能替代学习该条件模型。

当前真实训练总目标为lambda_obs*CE(logits_obs,y)+beta(epoch)*mean(KLc+KLa)+fire_weight*fire_loss。默认lambda_obs=1、beta=0.001、warmup10；本次DEAP fire=0、HCI fire=1，其他当前实验配置按记录。未对真实观察标签计算do CE，也没有真实oracle介入概率。

合成SCM已知真实干预概率时，才能显式开启单独的oracle概率监督。真实DEAP/HCI不能拿观察标签冒充该oracle。

真实中介不可观测，当前高斯后验和条件结果模型是联合学习的。事实CE+KL不自动保证潜在中介的非参数可识别性或对输入的充分性；结果模型可能更多利用g(x)。事实预测很好也不能证明中介机制学对，事实预测本身很弱则还需检查基础拟合。保持真实front-door研究目标意味着正视这一学习/识别缺口，不擅自改SCM或用普通融合换名。

## 5. 两种输出与两种数据集合，共有四类ACC

| 统计 | 使用概率 | 数据 | 用途 |
|---|---|---|---|
| 事实训练集ACC | probs_obs | 实际优化训练子集，epoch末eval全覆盖 | 检查观察拟合能力 |
| 前门训练集ACC | probs_do | 同上 | 观察调整后预测行为；不是训练目标已被do标签监督 |
| 事实测试ACC | probs_obs | held-out测试被试 | 观察预测泛化诊断 |
| 前门测试ACC | probs_do | 同一held-out测试被试 | 当前完整NeuroFDAN预设的主要分类指标 |

epoch_metrics.train/test是epoch末eval模式的完整集合统计，有限batch的smoke另标记不完整；history.train还包含训练模式下逐优化batch统计，模型权重在epoch中持续变化，不能与epoch末完整训练集ACC当作完全相同的量。

所有ACC均为窗口级argmax，没有自行更换阈值；矩阵行真值、列预测，顺序[0,1]。DEAP正类1/HCI正类0只影响F1、Precision、Sensitivity等正类指标定义，不改变ACC或矩阵顺序。

## 6. 与其他方法比较应报告什么

按当前已确定的完整前门分类方法，主表报告probs_do的测试ACC，以及既定F1等指标；不能看到事实ACC更高就临时替换，也不取max(事实ACC,前门ACC)、不对两个ACC求平均。

无需在主表强制每个方法都列两种ACC。建议主表给各方法的预设主预测结果；单独诊断/消融表给同一个完整模型checkpoint的probs_obs与probs_do。若独立训练fd_mode=none，则另外命名并明确训练模式，它不是同一个full checkpoint的事实输出。none跳过参考采样，会改变随机流消费，不能把两次训练差异全部归因于积分。

同一full模型的事实对照应使用由原probs_do规则选出的同一checkpoint，不额外以事实测试ACC选一个更好的epoch来替代而不说明。事实单采样与前门S/K平均包含MC差异，需解释机制时用已实现的E4同Ma/同S配对，不能把ACC差直接命名为“去混杂收益”。

若未来决定方法的最终部署输出为事实概率，那是另一个明确的方法/汇报选择，不能仍将其分类成绩作为已执行前门调整的收益。本轮不改变预设主输出或checkpoint规则。

正式LOSO先逐held-out subject计算测试ACC，再报告跨被试均值/标准差，标准差不是多种子标准差。每个预定fold必须有状态，排除smoke，不只挑成功或高分被试。严重类别不平衡时同时展示各类指标/混淆矩阵，可补充平衡准确率，但不替换现有选模协议。

当前保留held-out测试ACC/F1选best，没有validation，结果有测试选模偏差；被试内归一化也使用测试被试自身无标签统计。与其他方法比较还需核实其划分、预处理、标签、选模和统计口径，不能仅因都叫LOSO就宣称完全公平或无偏。

因此“不用观察标签监督probs_do”准确指不对其计算分类梯度；观察测试标签仍通过当前既定规则参与checkpoint选择，不能说它们从未影响模型选择。该选模过程不是合成oracle干预监督。

主分类表使用观察标签评测，不能单独验证p(y|do(x))估计准确性；合成SCM需另报干预概率误差、校准及oracle/learned信息条件，真实数据分类和因果识别证据分层呈现。
