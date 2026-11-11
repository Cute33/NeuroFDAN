# 大batch结果复盘与固定小批记忆能力检查

## 1. 四个快照的真实状态

DEAP smoke：588af6e1f20dea09dd9d28c41f5ea2c53897325a，runs/2026-11-11_112510_DEAP_arousal_fold01_9b5839fb。

DEAP20：f93d7a0cf3d032f92c122cba5fb623bf172f3571，runs/2026-11-11_112608_DEAP_arousal_fold01_4b8151f2。

HCI smoke：463cddd61bb7b1d581efd253c2a80feaf1e0e75f，runs/2026-11-11_112712_HCI_arousal_fold01_40df6559。

HCI20：8b8f4a69081cae2b254b37c86f037279241714d0，runs/2026-11-11_112813_HCI_arousal_fold01_bdc1fb23。

每组13个载荷大小、SHA256均一致，日志无Traceback/OOM；20轮组完整记录epoch1–20、完整评估，summary与history末条一致。smoke只有1个更新batch/1个评估batch，complete=false符合设置；DEAP smoke的100%来自40个测试窗口全为class1，绝不能作正式成绩。可选文件缺失不影响现有结果，不声称服务器单元测试日志已回传。

参考库各2048，DEAP覆盖31个训练被试、HCI覆盖23个，不含测试被试1。优化训练样本DEAP10400、HCI1600；未平衡参考总体分别12400、2300。大batch没有丢弃尾窗。真实大batch配置已在用户服务器完成，云端未运行这些训练。

## 2. 主要结果：工程可运行，稳定学习尚未建立

| epoch20量 | DEAP400 | HCI100 |
|---|---:|---:|
| 每epoch更新/20轮总更新 | 26/520 | 16/320 |
| 优化阶段事实CE | 0.691504 | 0.688245 |
| 完整训练集评估事实CE | 0.691127 | 0.686499 |
| 事实训练ACC | 0.518269 | 0.598125 |
| 事实测试ACC | 0.4325 | 0.54 |
| 前门训练ACC | 0.5000 | 0.5075 |
| 前门测试ACC | 0.40 | 0.77 |
| 前门测试平衡ACC | 0.50 | 0.50 |
| 20轮前门测试精确单类记录数 | 14 | 16 |

DEAP前门测试矩阵[[160,0],[240,0]]；HCI为[[1,19],[4,76]]。HCI真值为class0=20、class1=80，原正类仍为0；其best ACC0.8、选择F1=0来自全部预测class1，而不是评价定义错误。epoch20的0.77也没有超过始终预测多数类的0.8，平衡ACC仍0.5。

HCI事实训练从0.505到0.598125、训练评估CE从0.693385到0.686499，存在学习迹象；其事实测试平衡ACC仅0.50625，不能称为稳定跨被试泛化。两个数据集输入模态、样本量和fire权重不同，不能把差异直接归因于fire项。

大batch和shuffle不是充分修复。原收缩贡献因素已有证据，但剩余问题仍需区分基础拟合能力、训练/评估模式、数据异质性和潜在中介/结果条件模型的学习依据。20轮更新数很少，不能据此断言增加更新无效；也不应未经能力检查就盲目继续200轮或LOSO。

## 3. 下一项检查的目标与边界

只先检查最弱的DEAP：同一训练被试2、固定16窗口，两类各8个；对seed42新初始化及epoch20权重两个临时模型，各最多300个更新。复用原train_one_epoch、原事实CE＋KL＋fire、lr1e-4、wd0、clip=1，不手写另一套优化目标。

固定loss_epoch=20，使用原warmup公式对应的有效beta=0.001，不执行正式从头warmup；这是当前有效监督目标的局部记忆能力检查。两组都新建Adam、不继承原moment，明确不等同于原训练续训。诊断batch16不替换用户已选的正式DEAP400/HCI100。

本次与E1/E2不同：临时模型确实backward/step，但不保存任何权重、不调用run_fold，不覆盖原checkpoint或记录。只计算事实监督，不执行未被真实标签监督的do/K参考积分；fd_mode=none仅用于略过该计算，不是完整前门训练/消融成绩。它省去了不参与本次损失的参考库计算，改变了随机数消费，因此不与原完整训练轨迹作逐位比较。

诊断样本来自训练总体，可能不全在旧标签下采样优化子集中；不读held-out输入/标签。检查完全相同输入是否有冲突标签，存在时直接停止。没有oracle中介或干预概率。

每50步只读检查训练模式和eval模式，各4个固定随机流。前者保留Gumbel/dropout，后者使用原评估图及采样数；所有BN保持原冻结。分别记录单次CE/ACC和4次概率平均矩阵，检查评估没有改变参数/buffer、恢复训练随机流。模式差异也包含采样设置差异，不能只归因于Gumbel。

最后一次检查若两种模式的4次单次ACC均≥0.95、平均CE≤0.1，记为本次预声明记忆阈值达到；不提前停止、不挑最好步骤、不把阈值称为因果识别/正式选模标准。未达阈值还要看曲线、梯度和参数更新，不能自动证明网络永远不可训练。程序退出0只代表诊断完成，不代表记忆阈值通过。

即使记忆成功，也可能主要靠g(X)的条件信息，不能证明Ma保留了因果识别所需信息。真实前门仍依赖联合SCM假设、可识别的中介机制与足够条件分布支持；事实CE/KL并不自动完成潜在中介的非参数识别。本检查没有改变研究目标或把方法降级成普通分类器。

## 4. 服务器完整命令

复用dslenv，原项目分支和所有旧结果保持。这里只运行独立诊断脚本，不执行mainNeuroFDAN.py，也不重新跑20轮。

```bash
cd /home/fss/NeuroFDAN
export CAPACITY_TOOL_DIR=/home/fss/NeuroFDAN-capacity-tools
export CAPACITY_RESULT_DIR=/home/fss/NeuroFDAN/outputs/phase5_probes/deap_arousal_fold01_capacity_001
(
set -euo pipefail
git status --short
git worktree list
git fetch origin refs/heads/feat/neurofdan-fixed-batch-capacity:refs/remotes/origin/feat/neurofdan-fixed-batch-capacity
if [ ! -e "$CAPACITY_TOOL_DIR" ]; then
    git worktree add --detach "$CAPACITY_TOOL_DIR" origin/feat/neurofdan-fixed-batch-capacity
fi
test "$(git -C "$CAPACITY_TOOL_DIR" rev-parse HEAD)" = "$(git rev-parse origin/feat/neurofdan-fixed-batch-capacity)" || {
    echo '版本不符，停止，不覆盖。'; exit 1;
}
test -z "$(git -C "$CAPACITY_TOOL_DIR" status --porcelain)" || {
    echo '工具有修改，停止，不覆盖。'; exit 1;
}
test -f outputs/phase5_large_batch/DEAP/arousal/seed42/deap_arousal_fold01_b400_20ep_001/fold01/last.pt
mkdir -p outputs/phase5_probes
mkdir "$CAPACITY_RESULT_DIR"
bash "$CAPACITY_TOOL_DIR/scripts/phase3_run_logged.sh" "$CAPACITY_RESULT_DIR/train.log" \
  python -u "$CAPACITY_TOOL_DIR/scripts/check_fixed_batch_capacity.py" \
  --root_dir /home/fss \
  --fold_dir /home/fss/NeuroFDAN/outputs/phase5_large_batch/DEAP/arousal/seed42/deap_arousal_fold01_b400_20ep_001/fold01 \
  --probe_subject 2 --samples 16 --steps 300 --check_every 50 \
  --seed 42 --device cpu --output_dir "$CAPACITY_RESULT_DIR"
cat "$CAPACITY_RESULT_DIR/train.log.exit"
)
```

两组临时模型合计最多600个小批更新，CPU耗时以服务器实测为准，不套用原GPU整epoch耗时。每50步打印进度，不自动延长上限。原checkpoint只读，GitHub不含权重，缺失时不随机替代epoch20组。输出probe_summary.json、train.log、train.log.exit；不生成pt/pth权重。重跑须换新编号。

## 5. dry-run和上传

```bash
cd /home/fss/NeuroFDAN
bash /home/fss/NeuroFDAN-capacity-tools/scripts/upload_run_diagnostics.sh \
  /home/fss/NeuroFDAN/outputs/phase5_probes/deap_arousal_fold01_capacity_001 \
  --log /home/fss/NeuroFDAN/outputs/phase5_probes/deap_arousal_fold01_capacity_001/train.log \
  --dry-run
```

检查manifest后：

```bash
cd /home/fss/NeuroFDAN
bash /home/fss/NeuroFDAN-capacity-tools/scripts/upload_run_diagnostics.sh \
  /home/fss/NeuroFDAN/outputs/phase5_probes/deap_arousal_fold01_capacity_001 \
  --log /home/fss/NeuroFDAN/outputs/phase5_probes/deap_arousal_fold01_capacity_001/train.log
```

回传固定SHA/链接。失败也上传日志，不补造成功JSON、不覆盖大batch实验。纯探针命名unknown不代表失败。

## 6. 后续按结果分支，不再串行盲调多个超参数

| 本次观察 | 可支持的下一方向 | 不能声称 |
|---|---|---|
| 两个起点、两种模式都可记忆 | 基本拟合可行，审查全数据异质性、更新预算和潜在中介学习依据 | 前门或真实中介已经被识别 |
| 训练模式能记忆，eval明显不能 | 定位模式/采样差异，结合梯度与当前图做最小验证 | 直接改BN或宣布Gumbel是唯一根因 |
| 新初始化能记忆、epoch20起点不能 | 参数状态/局部优化问题候选；已明确重置moment | 原Adam历史moment一定有错 |
| 两个起点均未形成明显拟合 | 根据CE、逐模块梯度、参数更新和输入统计排查基础学习路径 | 仅凭300步证明模型理论上不可能训练 |

这是对诊断的收敛，不同时再加多种子/LOSO、KL消融、学习率或结构改动。若基础拟合可行但中介仍弱，下一份研究方案必须明确潜在中介和条件结果模型的学习依据，并在与真实训练信息条件一致的合成设置中验证；不能偷偷使用真实数据不存在的中介标签或do真值。任何模型/目标修订先明确数学定义与影响。

## 7. 实际工作状态

本轮核验的是用户服务器已执行的四组结果。云端没有执行模型、CLI、单元测试或训练；新容量检查尚待服务器运行。不修改NeuroFDAN、frontdoor_scm、ModelTrain、Datasets、loss或原主入口；只新增独立诊断与说明，不改main、不创建PR或合并。源代码审查/语法通过不等于服务器检查已通过。

实际静态检查：新脚本AST解析、三个文档bash块的bash -n、参数名核对、git diff --check均通过。新脚本复用train_one_epoch，未调用torch.save/run_fold/run_loso；核心文件相对446b0b5无差异。没有把静态检查写成单元测试通过。
