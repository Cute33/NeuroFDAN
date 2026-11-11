# 20轮复盘：停止扩大训练，审计优化批次而非继续重复中介探针

## 1. 结果与末尾报错分开判断

已读取固定提交de890945f5c7b7915731dc931be9e1e79fd7ee2c，目录runs/2026-11-11_101851_DEAP_arousal_fold01_e2026bbe。manifest各载荷大小和SHA256匹配。epoch_metrics和history均记录1–20轮，最终summary.last_test与history最后一条test完全一致。epoch_metrics另有单类连续轮数和跨轮概率变化两个诊断字段，因此不能直接以其完整字典与summary不相等判错。

日志先记录2026-11-10 23:58:36的完整020/020，再出现“指定epochs不大于checkpoint已完成轮数”。原run_fold在加载checkpoint后、进入epoch循环和改写fold.json之前检查first_epoch=checkpoint.epoch+1；当目标20不大于已完成轮次时拒绝继续。它与完成后重复执行同一续训命令相符，不是第20轮反向传播错误。

必须分别报告：20轮研究产物完整；日志中的最后一次启动失败。包装器tee -a追加日志、train.log.exit会被最近调用覆盖，因此最后退出码也不能代表此前调用。当前未读取服务器权重，不能独立保证其现存轮次/哈希；不删除错误日志，不清空旧结果，不为了消除报错改为100轮，不把失败启动称为成功。此处不需更改保护性报错。

## 2. 未达到正式训练推进门槛

epoch20事实训练ACC=0.525673，事实测试ACC=0.47，优化阶段事实CE=0.692247；均衡标签的常量0.5预测CE基线为ln2≈0.693147。该跨训练batch的平均CE与epoch末评估ACC不属于同一种统计，但均未显示稳定的强判别。最后5轮事实训练ACC约0.499–0.528，没有持续改善。

前门训练矩阵[[0,5200],[1,5199]]，测试矩阵[[0,160],[0,240]]。测试ACC0.6/F1=0.75来自全部预测1，不是新学习突破；best仍是继承的epoch1。测试中完整单类记录14/20，期间11–16轮短暂出现两类预测，不能说20轮全程精确单类，也不能把短暂波动当作稳定修复。

关闭衰减后首卷积及中介编码权重仍保留，确认不是原权重收缩再次复发。CNS均值跨样本变化小、posterior std约0.998，KL逐步趋小，结合E3/E4支持后验靠近先验、输入信号弱的解释，但不是“方差趋零”的塌缩。阈值1e-3的真假翻转不能作为恢复标准；logvar可能编码信息，ANS均值变化还混有Mc采样噪声。

研究目标仍为联合X=(Xc,Xa)、Mc→Ma、OutcomeHead(Ma,g(X'))和真实概率前门积分。事实CE可以利用g(X)中的观察关联；CE+KL不自动识别真实潜在中介。尚无证据支持修改SCM、改阈值、用观察标签监督do或以普通融合替代前门。本轮停止扩展epoch/fold，不安排新的E1–E4复跑。

## 3. 新的具体优化候选：窗口数不等于独立trial数

静态路径为balanced_indices保留trial的10个窗口 → GroupedSampler按10窗口组打乱 → batch40 → 每批只有4个trial。窗口标签在同trial内相同，所以全训练集10400窗口类别均衡，不保证每个batch平衡。

在独立等概率trial标签近似下，4个trial组成的batch全同类概率约12.5%；这是理论参考，不是本次实测值。窗口逐条打乱的40个窗口正类比例波动通常会更小。相关窗口也使每批多样性降低，批次先验摆动可能推动分类bias反复切换。它是可检验的优化候选，不是已证实唯一根因；很多模型仍能在这种分组下训练，不能仅凭分组认定代码错误。

因此下一步只读原训练标签并复用原balanced_indices和GroupedSampler，量化：每批正类比例、单类batch数量、不同trial数量、末批/最后8批的先验。附加窗口级打乱只是虚拟统计对照，不执行其训练，不改正式DataLoader，不改变窗口/标签/LOSO/参考库。

重建已知epoch6重新创建采样器的断点，不假装连续运行20轮；以服务器原数据未改变为前提，历史没有保存标签哈希，不能承诺逐位重放。只读2–32号被试文件的labels；pickle格式可能同时反序列化信号，但不做预处理/模型前向，也不上传原信号、逐窗口标签或权重。

## 4. 完整服务器命令

继续使用dslenv。本次不执行mainNeuroFDAN.py、不训练，不需要再加载checkpoint。保留全部旧结果。

```bash
cd /home/fss/NeuroFDAN
export BATCH_TOOL_DIR=/home/fss/NeuroFDAN-batch-audit-tools
export BATCH_RESULT_DIR=/home/fss/NeuroFDAN/outputs/phase5_probes/deap_arousal_fold01_batch_audit_001
(
set -euo pipefail
git status --short
git worktree list
git fetch origin refs/heads/feat/neurofdan-batch-composition-audit:refs/remotes/origin/feat/neurofdan-batch-composition-audit
if [ ! -e "$BATCH_TOOL_DIR" ]; then
    git worktree add --detach "$BATCH_TOOL_DIR" origin/feat/neurofdan-batch-composition-audit
fi
test "$(git -C "$BATCH_TOOL_DIR" rev-parse HEAD)" = "$(git rev-parse origin/feat/neurofdan-batch-composition-audit)" || {
    echo '工具版本不符，停止，不覆盖。'; exit 1;
}
test -z "$(git -C "$BATCH_TOOL_DIR" status --porcelain)" || {
    echo '工具存在修改，停止，不覆盖。'; exit 1;
}
mkdir -p outputs/phase5_probes
mkdir "$BATCH_RESULT_DIR"
bash "$BATCH_TOOL_DIR/scripts/phase3_run_logged.sh" "$BATCH_RESULT_DIR/train.log" \
  python -u "$BATCH_TOOL_DIR/scripts/audit_batch_composition.py" \
  --root_dir /home/fss \
  --fold_dir /home/fss/NeuroFDAN/outputs/phase5_controls/DEAP/arousal/seed42/deap_arousal_fold01_wd0_20ep_followup_001/fold01 \
  --restart_epoch 6 --output_dir "$BATCH_RESULT_DIR"
cat "$BATCH_RESULT_DIR/train.log.exit"
)
```

输出probe_summary.json、train.log、train.log.exit。正常两个退出码为0。再次运行使用新编号，不覆盖。它只审计已确定的DEAP arousal fold1、seed42、batch40、1–20轮设置，参数不符明确停止。

## 5. 上传命令

```bash
cd /home/fss/NeuroFDAN
bash /home/fss/NeuroFDAN-batch-audit-tools/scripts/upload_run_diagnostics.sh \
  /home/fss/NeuroFDAN/outputs/phase5_probes/deap_arousal_fold01_batch_audit_001 \
  --log /home/fss/NeuroFDAN/outputs/phase5_probes/deap_arousal_fold01_batch_audit_001/train.log \
  --dry-run
```

检查manifest后：

```bash
cd /home/fss/NeuroFDAN
bash /home/fss/NeuroFDAN-batch-audit-tools/scripts/upload_run_diagnostics.sh \
  /home/fss/NeuroFDAN/outputs/phase5_probes/deap_arousal_fold01_batch_audit_001 \
  --log /home/fss/NeuroFDAN/outputs/phase5_probes/deap_arousal_fold01_batch_audit_001/train.log
```

回传独立SHA与链接；原20轮快照无需再上传。这里没有声称窗口打乱训练已运行或已改善ACC。

## 6. 决策边界与实际工作

这次只增加一项有具体优化假设的标签统计审计，不继续重复中介/参考探针。若实际批次先验剧烈波动，下一项候选是保持样本、SCM、损失和参考总体不变，仅改变优化窗口排列的单变量训练对照；必须先说明这与原分组协议的差异，不能静默替换默认采样器。若审计不支持该候选，停止沿此方向追加测试，转入潜在中介/条件结果模型的学习方案审查。即使虚拟窗口打乱更均衡，也不证明它足以修复真实学习，不据此直接启动LOSO。

本轮仅核验GitHub产物和静态源码，未运行云端模型、CLI、单元测试、训练或GPU实验。新审计尚待服务器执行。新增独立脚本与本文，不改变核心训练或模型文件，不修改main、不创建PR或合并。不能保证再花固定分钟就找到唯一原因或取得预定ACC。

实际通过的静态检查：新脚本AST解析、三个bash块的bash -n、git diff --check；核心模型/SCM/训练器/数据/loss/上传器相对64bc3a4无差异。上述语法和差异检查不等于审计脚本已在真实数据上通过。
