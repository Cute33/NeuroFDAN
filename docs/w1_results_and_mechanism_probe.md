# W1结果与E3/E4：围绕分层中介和真实前门估计的只读机制诊断

## 1. 固定证据与核验

A组：提交d11717bf478ad276299e90255ea869f4424d06e2，runs/2026-11-10_225557_DEAP_arousal_fold01_88af788c。

B组：提交5b105440f5b80ada0e80e6f80476c1fa0438d536，runs/2026-11-10_225646_DEAP_arousal_fold01_9c04d1a5。

每组manifest所列13个载荷大小、SHA256均匹配，skipped为空，日志没有Traceback。缺少的是白名单中的可选config/探针摘要或位于非预期子目录的run摘要；根目录run摘要均存在。这些不是训练失败。两组各有epoch1–5完整记录，smoke=false，正常收尾；JSON与NPZ的epoch、完整标记、训练/测试矩阵一致，ACC与矩阵重算一致。

同一源码9d698f3fcfbd6b41579673ddb43e3a8074b9d435、seed42、cuda:0、batch40、相同模型/损失/采样配置。CLI仅run_name与weight_decay不同；实际优化器分别保存0.01、0。起始逐参数RMS与参考来源完全一致，但RMS不是完整权重哈希，不据此承诺逐位相同。

每组训练被试2–32，优化训练窗口10400，未平衡参考总体12400，参考库2048、覆盖31名训练被试，测试被试1的400窗口。已逐条核对参考来源为训练被试且不含1。没有validation，没有改变测试选模协议。源码没有因本次探针再次改变这些协议。

快照没有模型权重，也未包含服务器单元测试的独立输出，不能据此声明那些测试通过。Phase4 epoch37是用户主动终止的历史快照，不需要补齐。

## 2. 5轮结果：解决了一个退化因素，但没有完成研究目标

| epoch5量 | A：wd=0.01 | B：wd=0 |
|---|---:|---:|
| CNS首卷积核RMS | 0.165860 | 0.259699 |
| CNS中介encoder.0.weight RMS | 1.00036e-11 | 0.00601447 |
| ANS中介encoder.0.weight RMS | 8.57678e-9 | 0.0171680 |
| 完整训练集CNS posterior mean跨样本std均值 | 0 | 0.00232736 |
| 完整测试集CNS posterior mean跨样本std均值 | 0 | 0.00119796 |
| 事实训练集ACC | 0.500000 | 0.515481 |
| 事实测试集ACC | 0.400000 | 0.402500 |
| 前门训练集ACC | 0.500000 | 0.499904 |
| 前门测试集ACC | 0.400000 | 0.400000 |

两组epoch5前门测试矩阵同为[[160,0],[240,0]]；B事实训练矩阵[[4191,1009],[4030,1170]]，事实测试矩阵[[122,38],[201,39]]。行是真值、列是预测，顺序[0,1]。两组被原规则选中的best均为epoch1，ACC=0.6/F1=0.75，但当时全预测1；不能据此宣布模型学会区分两类。

本次单折单种子对照强烈支持原Adam耦合L2是中介权重收缩的贡献因素：B在相同短程配置下避免了已测权重的剧烈收缩。它不证明这是全部原因，也不保证wd=0是最终正式设置。A在第5轮记录的CNS posterior mean跨样本std已为0，而骨干卷积权重仍非零；该汇总值不等于逐元素相等的直接检验，仍由新探针核对，也不能把所有早期退化都定位到骨干完全常量。

B保住了参数尺度，但事实分类仍接近无判别，前门主输出没有恢复两类判别。不能通过延长训练、看最高测试ACC或改阈值来代替机制诊断。

## 3. 当前统计还混合了哪些因素

B训练集CNS posterior mean跨样本std均值约0.00233，而posterior std约0.97164，已记录的比值约0.00260；这提示均值通道的输入变化相对采样噪声很弱。不能仅据此认定整个后验无信息，因为logvar也可能依赖输入。

B的ANS posterior mean跨样本std均值约0.04514，同一窗口随Mc采样的条件均值std约0.17170。ANS均值非零变化混合了真实ANS输入、CNS输入及CNS外生噪声；不能直接把这些变化都称为生理信息恢复。正常约1的采样方差不排除输入无关的后验塌缩。

原事实指标使用一次Ma样本，前门输出对S/K平均；W1两条路径概率std或矩阵的差异，还混有采样平均的降噪效应。它们不能单独证明参考边缘化导致了退化，也不能证明前门公式有错。真实p(y|do(x))本来就不必等于p(y|x)，更不以观察标签ACC越高作为识别正确性的判据。

## 4. 研究主线与本次探针

目标仍是联合处理X=(Xc,Xa)、CNS→ANS分层中介与真实前门概率积分。保留最终SCM，不增加Mc→Y或X→Y直接边，OutcomeHead只读取Ma与参考表示。g(X')仍是计算表示，不是新因果节点。真实潜在中介不可观测，仅事实CE/KL不能保证潜在中介或因果效应被非参数识别。

新增scripts/probe_mechanism_paths.py，在独立分支feat/neurofdan-mechanism-probe。只读取W1两组现有epoch5 last.pt，model.eval、no_grad、不创建优化器、不算损失、不反向、不更新权重；前后核对state_dict和文件哈希。不再启动训练。本分支不修改任何模型、训练器、损失、数据或默认配置。

E3逐项隔离：

1. 同一组CNS/ANS标准正态噪声，在窗口间共享，S轴独立；对比CNS输入变化引起的mu_c、logvar_c及Mc变化。
2. 固定ANS特征，将上一项配对Mc输入原ANSMediator，读取mu_a、logvar_a、Ma及前门概率变化。
3. 固定Mc，仅替换为另一个真实窗口的ANS特征，读取ANS自身输入依赖。
4. 固定真实输入和ANS噪声，只换CNS噪声，测量Mc外生噪声对ANS条件分布的影响。
5. hook原ANS SNN，读取实际input_proj输出和CNS附加电流的RMS；不重复实现或替换SNN。

E4复用同一Ma张量、同一经验参考和Phase1唯一的marginalize方法，分别计算单次事实、同S事实平均、同S/K前门概率；先softmax再平均。再固定Ma替换事实参考，测量条件Head对参考输入的敏感度；分别沿条件组合的S/K配对检查其对Ma样本和参考的数值依赖。条件组合的置换差不等于积分差：K置换本就不应改变积分均值。

这不是重训fd_mode=none、不是把噪声均值设为新的默认推理路径，也不是用观察标签监督do。共享随机数只用于配对比较：每点的高斯边缘分布和原公式不变，跨点MC误差相关。局部样本替换可能落在稀疏联合支持区，不是对人体的物理干预或Y独立于Mc的统计检验。

## 5. 数据来源与shape

仅选训练被试2、原参考库外的16个窗口，每类8个，记录subject/trial/window。约束的是探针取样，不改变优化训练集或参考总体，不把测试被试1的输入加入参考库。

在checkpoint原2048成员参考库内均匀有放回抽K=64，A/B及所有比较共用同一配对参考。当前探针不在库内，因此共享K个参考仍满足排除自身；不需要从参考总体额外删除其他窗口。每次只加载所需被试并复制所需窗口，不重建完整训练集，也不下载数据。

CNS原输入[B,32,768]，ANS[B,3,768]；原骨干输出[B,96,128]和[B,96,12]。默认完整参考表示为13440维，Ma为[B,16,64]，组合logits为[B,16,64,2]。这些来自当前实际模型，不套用旧1120维接口。

均值/方差参数通过原CNSMediator和ANSMediator取得；以相同显式epsilon重构其原高斯采样公式，避免两个独立forward的噪声混淆。小批矩阵只作描述，不作为正式训练集/测试集成绩或新论文结果。

## 6. 服务器执行：只读E3/E4，不运行mainNeuroFDAN.py

继续使用dslenv；原项目分支、W1结果和所有旧快照保持原样。先在同一Xshell会话执行：

```bash
cd /home/fss/NeuroFDAN
export MECH_TOOL_DIR=/home/fss/NeuroFDAN-mechanism-tools
export MECH_RESULT_DIR=/home/fss/NeuroFDAN/outputs/phase5_probes/deap_arousal_fold01_w1_e3e4_001
(
set -euo pipefail
git status --short
git branch --show-current
git worktree list
git fetch origin refs/heads/feat/neurofdan-mechanism-probe:refs/remotes/origin/feat/neurofdan-mechanism-probe
if [ ! -e "$MECH_TOOL_DIR" ]; then
    git worktree add --detach "$MECH_TOOL_DIR" origin/feat/neurofdan-mechanism-probe
fi
test "$(git -C "$MECH_TOOL_DIR" rev-parse HEAD)" = "$(git rev-parse origin/feat/neurofdan-mechanism-probe)" || {
    echo '工具版本不符，停止，不覆盖。'; exit 1;
}
test -z "$(git -C "$MECH_TOOL_DIR" status --porcelain)" || {
    echo '工具有本地修改，停止，不覆盖。'; exit 1;
}
test -f outputs/phase5_controls/DEAP/arousal/seed42/deap_arousal_fold01_wd001_5ep_001/fold01/last.pt
test -f outputs/phase5_controls/DEAP/arousal/seed42/deap_arousal_fold01_wd0_5ep_001/fold01/last.pt
mkdir -p outputs/phase5_probes
mkdir "$MECH_RESULT_DIR"
bash "$MECH_TOOL_DIR/scripts/phase3_run_logged.sh" "$MECH_RESULT_DIR/train.log" \
  python -u "$MECH_TOOL_DIR/scripts/probe_mechanism_paths.py" \
  --root_dir /home/fss --dataset DEAP --task arousal \
  --held_out_subject 1 --probe_subject 2 --expected_epoch 5 \
  --samples 16 --S 16 --K 64 --seed 42 --device cpu \
  --fold_dir /home/fss/NeuroFDAN/outputs/phase5_controls/DEAP/arousal/seed42/deap_arousal_fold01_wd0_5ep_001/fold01 \
  --control_fold_dir /home/fss/NeuroFDAN/outputs/phase5_controls/DEAP/arousal/seed42/deap_arousal_fold01_wd001_5ep_001/fold01 \
  --output_dir "$MECH_RESULT_DIR"
cat "$MECH_RESULT_DIR/train.log.exit"
)
```

缺少原权重、轮次/配置不符、参考来源不一致或可用两类窗口不足时停止，不重新训练或用随机模型替代。输出为新目录下的probe_summary.json、train.log、train.log.exit；正常退出码均0。新JSON应有两组model_state_unchanged=true和checkpoint_files_unchanged=true。再次执行须统一换新编号。

## 7. dry-run与正式上传

```bash
cd /home/fss/NeuroFDAN
bash /home/fss/NeuroFDAN-mechanism-tools/scripts/upload_run_diagnostics.sh \
  /home/fss/NeuroFDAN/outputs/phase5_probes/deap_arousal_fold01_w1_e3e4_001 \
  --log /home/fss/NeuroFDAN/outputs/phase5_probes/deap_arousal_fold01_w1_e3e4_001/train.log \
  --dry-run
```

查看manifest及脱敏内容后：

```bash
cd /home/fss/NeuroFDAN
bash /home/fss/NeuroFDAN-mechanism-tools/scripts/upload_run_diagnostics.sh \
  /home/fss/NeuroFDAN/outputs/phase5_probes/deap_arousal_fold01_w1_e3e4_001 \
  --log /home/fss/NeuroFDAN/outputs/phase5_probes/deap_arousal_fold01_w1_e3e4_001/train.log
```

追加server-diagnostics独立记录，不覆盖W1或历史快照；不上传权重/信号。回传工具打印的固定SHA与链接。纯探针目录可能仍命名unknown，JSON内明确标识两组与任务。

## 8. 以研究问题决定后续动作

若真实CNS输入改变几乎不影响ANS，而换CNS噪声影响明显，不能称为已学到有效CNS调节；优先查后验均值/方差及电流尺度，不增加Mc→Y捷径或删除层次中介。

若Head对参考敏感、对携带输入差异的Ma不敏感，说明事实条件预测可能主要依赖g(X)，仍需解决潜在中介和结果条件模型的学习问题；不能盲目对probs_do加观察标签CE来伪造干预监督。

若同S事实平均已经接近常量，单次事实变化可能主要是噪声；若同S事实仍有变化而参考边缘化后减弱，应区分去除观察关联与估计失败，不能仅凭分类ACC判定前门公式错误。

只有配对输入依赖与中介机制的证据清楚，再设计下一项最小验证或有明确数学依据的建模修订。W1不预设新默认值；本轮不扩展到100轮、其他fold或完整LOSO。理论SCM识别条件、代码正确性、合成干预真值恢复、真实生理实证仍需分层报告。

## 9. 实际执行边界

本轮在云端只读取既有GitHub产物、做标准库JSON/NPZ核对和静态源码审查；没有执行模型、CLI、单元测试、训练或GPU实验。新E3/E4探针尚无服务器运行结果，语法检查不能代替运行验证。没有修改W1训练入口、优化器默认值或任何核心模型/数据/损失文件，没有PR或合并main。

已执行并通过：新Python探针AST解析、三个文档bash块的bash -n、探针参数名核对、git diff --check。逐文件差异确认模型、SCM、训练器、数据、损失和上传器相对W1基线无变化。上传产物的完整性检查与源代码静态检查分别记录，不能替代新探针运行测试。
