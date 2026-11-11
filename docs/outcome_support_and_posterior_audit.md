# 事实配对与交叉配对支持、中介均值/方差依赖的联合审计

## 1. 对用户提出问题的明确回应

事实配对(ma_i,g(x_i),y_i)上的CE只能直接约束该观测联合分布的风险。前门计算需要(ma_i,g(x'_k))，其联合分布不同；两个边缘都来自训练总体不保证交叉组合有足够联合支持。训练full前向也计算交叉logits，但没有交叉组合专属真实结果标签或分类loss；共享参数会被事实CE更新，不等于交叉条件结果已充分学习。不能将原y_i直接复制给全部交叉组合当作真实标签。

若真实中介可观测，观测联合支持充分，且条件模型一致估计所需p(y|ma,x')，事实监督原则上可以支持前门识别；不是必须为每个交叉组合另造标签。有限样本/压缩表示/潜在中介/选择偏差使这些前提需要逐项验证。真实Ma未观测，因此本轮只能审计模型重建的支持与依赖，不能直接证明真实生理条件分布已学准。

两个条件概率模型可能在事实高密度区域给出相近拟合，却在交叉低支持区域给出不同结果；前门公式本身不会解决这种统计外推问题。support不足与中介完全不依赖输入也需要区分：后者可能让交叉配对看似容易覆盖，但并不代表识别成功。

### 1.1 更基本的问题：重采样潜变量的监督联合分布

令Z为当前同一次forward的完整骨干特征。默认ref_dim=0时，事实ReferenceEncoder只是拼接Z；CNS/ANS生成器也仅使用Z和重新抽取的独立噪声。给定固定模型权重，训练构造的联合分布是：

p_aug(z,y,tilde_m_a) = p_opt(z,y) q_phi(tilde_m_a|z)。

所以在这个人为增广分布中，Y与tilde_Ma在给定Z后独立。tilde_Ma特指本次代码重采样的潜变量，不是擅自给最终SCM增加变量或改变真实Ma方程。真实SCM的结果由同一次生成中的Ma决定，不能把重新独立采样后附上原标签的配对等同于观测到该真实Ma。

若Head足够灵活、达到增广分布事实CE的总体最优，则q_theta(y|tilde_ma,z)=p_opt(y|z)，允许完全忽略tilde_ma。之后在固定eval映射下对参考总体积分，结果退化为E_ref[p_opt(y|Z')]，不再依赖当前输入；优化和参考总体一致时才等于该总体类别边缘。KL进一步偏好弱输入依赖的posterior，也不能补足缺少的真实中介—结果联合信息。

这是当前监督设计的可识别性/退化风险，不是证明有限网络已经达到该最优，也不是已确诊服务器唯一根因。训练/评估图不同、欠拟合、支持不足等还可能同时存在。它解释了为何事实ACC有所改善、交叉支持看似充分，也未必带来可靠do估计；必须检查Head是否实际依赖Ma，并在缺少真实中介标签的同等合成设置下验证学习依据。不能靠给交叉配对复制事实标签、直接监督do标签或擅自改变SCM来规避。

## 2. 后验诊断的三个层次及KL的数学边界

分别统计Var_X(mu)、Var_X(sigma)、Var_X(logvar)，以及条件Gaussian KL、给定输入的理论sigma和有限次采样std。不把参数维度变化当作跨输入变化，不把采样内变化当作输入信息。

精确、在几乎所有输入上等于先验的posterior没有输入信息；在同一q(X)分布下还有恒等式：

E_X KL(q(M|X)||p(M)) = I_q(X;M) + KL(q(M)||p(M))。

因此真实平均KL很小时对互信息给出上界，并非小KL还可以任意包含大量信息；但有限诊断样本、局部输入、数值误差和不同分布下的统计不能直接证明所有真实中介信息已消失。必须检查sigma依赖。训练float32 KL公式不改，审计用float64的mu²+expm1(logvar)-logvar提高近先验数值稳定性。

ANS必须额外区分q(Ma|Xa,Mc)与边缘混合q(Ma|X)。固定同一Mc网格改变Xa，才隔离ANS自身输入；用共同CNS标准噪声跨输入配对，诊断整个X通过均值和方差进入ANS的变化。对Mc积分时按全方差公式：Var(Ma|X)=E_Mc sigma_a² + Var_Mc mu_a。ANS条件Gaussian KL不是这个混合分布的精确KL，较大的条件KL也可能主要反映对Mc噪声的响应。

有限S=16、unbiased=False的采样std即使来自N(0,1)，均值也可约0.952；这不是posterior sigma小于1的证据。诊断结果同时保留理论sigma与经验采样std。

## 3. 新发现的抽样总体一致性候选

原优化集按标签trial下采样，参考库来自未平衡训练总体。按标签随机选择的概率若为a_y，所学结果条件一般满足p_opt(y|m,x) ∝ a_y*p_pop(y|m,x)，不能自动等同于前门公式需要的总体条件结果。这里选择指示只是说明数据抽样机制，不是擅自给最终SCM增加因果节点。

本轮读取训练标签、复用原balanced_indices，报告原总体/优化集各类数量和保留比例、参考抽样中真正进入优化集的比例。即使参考库无测试泄漏，这项选择一致性仍须检查。先报告证据，不取消平衡、不增加class weight、不重新定义p(X')；任何后续修订须明确数学目标和影响。它不是本轮已确诊根因。

## 4. 一次综合只读审计，不进行新的训练

脚本scripts/audit_outcome_support.py读取DEAP400/HCI100的完整epoch20 last.pt。保留SCM、所有模型模块、损失、采样器及选模协议；无优化器、不backward/step，不写参数grad。局部敏感度仅autograd.grad对输入求导。检查state_dict与checkpoint文件哈希不变。

普通与隔离导入各3次的快照1387e1d6eaeeccb3b16c554234e265eb0544a2a1已核验：全部成功，Python3.10.19、torch2.5.1+cu121，模块来源指纹一致。四个上传载荷的长度及SHA256均一致；manifest缺少run_summary.json、run_report.txt、config.json，这些不是本纯导入操作生成的必要文件，不能据此判定导入失败。原导入错误未复现，不声称找到根因，不重装/升级。按用户当前优先级，先做本联合审计，暂停重跑固定小批记忆检查。

来源分开：128个事实参照、64个距离校准窗口都从原优化集抽取，16个两类各半查询来自原优化集但不在原参考库内；64个参考从原2048成员库均匀有放回抽取。查询、参照、距离校准、所抽参考之间不重合，参考本身允许重复。校准只是只读诊断阈值，不增加LOSO validation split，不参与训练/选模。统计仅代表所抽样本，不能说是完整训练总体的精确Var_X。

原模型hook收集固定权重下的训练态/评估态事实OutcomeHead输入；重建两种模式的交叉输入边缘摘要。不是声称拿到了历史训练所有head输入。训练态图采样、评估态固定图属于模式差异，单独记录；BN继续冻结。边缘摘要相近不能代替联合支持检查。

### A. 两种互补支持诊断

1. 原完整参考特征分块标准化距离：Ma、完整CNS参考、完整ANS参考各按块方差及维数归一，计算事实查询和同Ma交叉组合到训练态事实参照的最近邻距离。分别用训练态/评估态事实距离校准的95%分位作启发式阈值，报告跨配对变化。不投影或改变ReferenceEncoder，不把高维NN距离当作密度或positivity证明。
2. 模型条件密度：q(Ma|x')需对q(Mc|xc')积分，不能把当前Mc误作为参考的条件。用L=16个参考CNS样本构成对角Gaussian条件混合，log域平均；查询Ma与参考密度的MC独立。参考专属的C=64次独立Ma样本校准5%低密度阈值，同时报告自己来源条件下的密度作参照。混合不是单高斯，也不重复调用直接干预函数。

高斯模型在所有有限点密度为正并不保证有限样本overlap充分。q若已退化为输入无关的先验，模型密度检查可能显示良好；真实支持未被观测，两个检查都不认证真实因果识别。必须结合后验依赖和Head敏感度判断。

### B. CNS与ANS都检查均值和方差

CNS直接报告跨输入mu、sigma、logvar变化、逐潜变量维度及稳定KL。ANS分别报告共同CNS噪声条件下的变化、固定Mc网格时仅Xa变化、对Mc积分后的均值和总std变化，以及同输入随Mc采样造成的条件均值变化。记录理论与经验采样噪声，不把两种变化混为同一量。

### C. OutcomeHead局部依赖

在eval模式下，对事实/交叉配对分别计算P1对Ma、CNS参考块、ANS参考块的输入梯度；记录原始梯度RMS及尺度归一L2，再施加1%训练块尺度的局部随机方向扰动读概率变化。量纲和维数不同，原始梯度不可直接当作贡献占比；局部敏感度不是因果效应、条件分布正确性或残余Mc直接因果边的证明。P1只用于数值敏感度，HCI正类定义不改。

## 5. 服务器执行：两个数据集串行，不训练

继续复用dslenv。先测试5项数学小例子（不创建NeuroFDAN），通过后才运行实际只读审计。若torch导入再失败，保留日志，不以随机模型替代。

```bash
cd /home/fss/NeuroFDAN
export SUPPORT_TOOL_DIR=/home/fss/NeuroFDAN-support-audit-tools
(
set -euo pipefail
git status --short
git worktree list
git fetch origin refs/heads/feat/neurofdan-outcome-support-audit:refs/remotes/origin/feat/neurofdan-outcome-support-audit
if [ ! -e "$SUPPORT_TOOL_DIR" ]; then
    git worktree add --detach "$SUPPORT_TOOL_DIR" origin/feat/neurofdan-outcome-support-audit
fi
test "$(git -C "$SUPPORT_TOOL_DIR" rev-parse HEAD)" = "$(git rev-parse origin/feat/neurofdan-outcome-support-audit)" || {
    echo '版本不符，停止，不覆盖。'; exit 1;
}
test -z "$(git -C "$SUPPORT_TOOL_DIR" status --porcelain)" || {
    echo '工具有修改，停止，不覆盖。'; exit 1;
}
for SUPPORT_DATASET in DEAP HCI
do
    if [ "$SUPPORT_DATASET" = DEAP ]; then SUPPORT_PREFIX=deap; SUPPORT_BATCH=400; else SUPPORT_PREFIX=hci; SUPPORT_BATCH=100; fi
    SUPPORT_RESULT_DIR="/home/fss/NeuroFDAN/outputs/phase5_probes/${SUPPORT_PREFIX}_arousal_fold01_support_001"
    mkdir -p outputs/phase5_probes
    mkdir "$SUPPORT_RESULT_DIR"
    bash "$SUPPORT_TOOL_DIR/scripts/phase3_run_logged.sh" "$SUPPORT_RESULT_DIR/train.log" \
      bash -c 'cd "$1" && python -m unittest discover -s tests -p test_outcome_support_math.py -v' _ "$SUPPORT_TOOL_DIR"
    bash "$SUPPORT_TOOL_DIR/scripts/phase3_run_logged.sh" "$SUPPORT_RESULT_DIR/train.log" \
      python -u "$SUPPORT_TOOL_DIR/scripts/audit_outcome_support.py" \
      --root_dir /home/fss \
      --fold_dir "/home/fss/NeuroFDAN/outputs/phase5_large_batch/$SUPPORT_DATASET/arousal/seed42/${SUPPORT_PREFIX}_arousal_fold01_b${SUPPORT_BATCH}_20ep_001/fold01" \
      --samples 16 --anchors 128 --calibration 64 --K 64 --L 16 --C 64 \
      --seed 42 --device cpu --output_dir "$SUPPORT_RESULT_DIR"
    cat "$SUPPORT_RESULT_DIR/train.log.exit"
done
)
```

输出两个独立目录的probe_summary.json、train.log、train.log.exit；不写pt/pth、不运行主训练函数。原权重只在服务器本地，GitHub快照不含权重。任何阶段失败停止，不覆盖原实验。主要开销是训练被试文件读取和有界MC/Head计算，CPU耗时以服务器实测为准，不是新的训练实验。

## 6. 上传命令

先预览：

```bash
cd /home/fss/NeuroFDAN
for SUPPORT_PREFIX in deap hci
do
    SUPPORT_RESULT_DIR="/home/fss/NeuroFDAN/outputs/phase5_probes/${SUPPORT_PREFIX}_arousal_fold01_support_001"
    if [ ! -d "$SUPPORT_RESULT_DIR" ]; then echo "尚未创建：$SUPPORT_RESULT_DIR"; continue; fi
    bash /home/fss/NeuroFDAN-support-audit-tools/scripts/upload_run_diagnostics.sh \
      "$SUPPORT_RESULT_DIR" --log "$SUPPORT_RESULT_DIR/train.log" --dry-run
done
```

检查manifest后正式上传：

```bash
cd /home/fss/NeuroFDAN
for SUPPORT_PREFIX in deap hci
do
    SUPPORT_RESULT_DIR="/home/fss/NeuroFDAN/outputs/phase5_probes/${SUPPORT_PREFIX}_arousal_fold01_support_001"
    if [ ! -d "$SUPPORT_RESULT_DIR" ]; then echo "尚未创建：$SUPPORT_RESULT_DIR"; continue; fi
    bash /home/fss/NeuroFDAN-support-audit-tools/scripts/upload_run_diagnostics.sh \
      "$SUPPORT_RESULT_DIR" --log "$SUPPORT_RESULT_DIR/train.log"
done
```

回传两个固定SHA/链接。导入检查的原001与容量失败记录均不覆盖。更改规模或重跑要新编号；不把统计校准窗口说成正式validation，不把只读校准当训练。

## 7. 按结果收敛研究方向

交叉低密度/距离明显增加：优先查support、选择机制和条件结果模型外推，不加伪交叉标签。支持差异不明显但mu/sigma输入变化都很小：优先查潜在中介信息和学习依据，不能把良好模型overlap称为成功。后验有响应但Head对相应变化很弱：查条件结果模型的训练约束与泛化；局部梯度非零也不证明关系学准。

如果保留比例有差异，必须先明确结果条件拟合的目标总体；不静默改权重/平衡/参考总体。即使以上诊断未见问题，真实Ma不可观测、压缩表示和有限条件估计仍限制识别结论。下一项修订应在与真实训练同样缺少Ma真值的合成设置下验证，不能拿有中介真值监督的oracle成绩替代此证据。

本轮不改SCM、训练目标或实验协议；不将问题降级为普通特征融合。本轮云端没有运行模型、CLI、单元测试或训练；新审计及数学测试均待服务器执行。已执行Python 3.10 AST语法检查、三组文档Shell命令的bash -n、git diff --check；与父提交比较，模型、数据、训练、损失和统计核心文件无变化。代码审查和语法检查不是本实验结果。
