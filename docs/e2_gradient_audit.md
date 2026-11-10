# E1b结果核验与E2：分项梯度及Adam耦合L2只读审计

起点feat/neurofdan-e1b-backbone-probe / 8484e001035d8e3563a4630d04a4c73ac321f4b6。独立分支feat/neurofdan-e2-gradient-audit。不改模型、SCM、数据、损失定义、优化器、BN或checkpoint规则；本轮不在云端执行模型/CLI/单元测试。

## 1. 服务器E1b证据

读取固定提交8dd2411b6c78b64254fd27b395cb661706dad5a1，目录runs/2026-11-10_215746_unknown_unknown_run_270757d8。manifest所列4个载荷字节数和SHA256均匹配，skipped为空；日志没有Traceback，三个模型均完成。E1/E1b的输入与最终输出统计差为0，last.pt哈希相同，best实际epoch=1、last实际epoch=37；模型和原文件不变，没有创建优化器。

| 条件 | CNS输出跨样本std | ANS输出跨样本std |
|---|---:|---:|
| seed42参考初始化 | 0.0238175 | 0.0234823 |
| best，实际epoch1 | 0.0157198 | 0.0172869 |
| last，实际epoch37 | 0 | 0 |

这是训练后出现的表示退化，不能解释为输入本身全常量或该构造下初始化必然无变化；但参考初始化不保证与历史启动逐位一致。best只代表原测试选模规则的产物，不表示分类已经有效。

两支骨干按旧std==0准则首次精确常量均在第三个卷积（cns_enc.6、ans_enc.6），不是它们后面的BN或SiLU首次归零。但更早的层已严重衰减：

| 卷积权重RMS | CNS初始化 | CNS epoch37 | ANS初始化 | ANS epoch37 |
|---|---:|---:|---:|---:|
| 第一层 .0 | 0.2601 | 4.4049e-8 | 0.2462 | 3.8099e-8 |
| 第二层 .3 | 0.1628 | 6.6753e-14 | 0.1694 | 1.0373e-12 |
| 第三层 .6 | 0.1664 | 2.6460e-9 | 0.1722 | 5.3698e-14 |

epoch37第三层卷积输出RMS分别为0.0062555、0.0070206，接近对应bias的RMS；**跨样本常量不等于激活全零，卷积权重也没有全部精确归零**。前层差异极小，叠加非零偏置后在实际float32前向中无法保留可辨别差异，与当前统计一致，不据此宣布PyTorch算错。

BN running_mean均为0、running_var均为1、num_batches_tracked均为0，符合原冻结策略。gamma约0.177–0.188而不是0，不能解释成BN全部关闭；它仍进一步缩小尺度。CNS图不是两支共同退化的唯一解释，ANS无该图仍退化。

旧“精确常量”使用std==0作判据。1e-30等极小std不能作为有效信号证据，浮点归约也不适合定义位级相等。本分支只改诊断统计：增加max-min，直接比较每个样本与首样本是否逐元素相同；std数值照常保留。**不改变任何前向或精度策略**，也不声称旧结果一定出错。E2在新的训练被试窗口上读取此判据，不把两个不同被试的差异归因于统计修订。

## 2. 证据等级与需要验证的假设

A已确认：训练后的卷积核多数量级收缩；输入差异在骨干内丢失；BN按原策略冻结；当前激活不是简单全零；Adam记录lr=1e-4、weight_decay=0.01。

B有依据但待检验的解释：任务信号与既有正则/参数衰减的量级失衡可能参与退化。原代码确实对所有参数使用单组Adam耦合L2，包含卷积、偏置及BN仿射参数；这项实现事实不能单独证明它导致了退化。

C未确认：哪项在训练早期占主导，以及哪个机制先退化。不能只凭权重收缩证明weight_decay是唯一原因；也不能只看late checkpoint的零任务梯度倒推早期原因。

原顺序是：事实CE+warmup KL+原发放率项 → backward → max_norm=1裁剪 → Adam.step内部加入weight_decay×参数。已有裁剪前梯度日志**不包含内部L2项**；Adam保存的exp_avg/exp_avg_sq包含历史有效梯度，不能从其中直接恢复每一步CE/KL各自贡献。

所以本轮不把weight_decay改成0，不改KL，也不重跑100轮。E2只比较同一个训练被试局部批次上，各项梯度和数学L2方向的相对量级，分别读取best/last的实际参数与moment统计。

## 3. E2的含义和边界

使用原训练被试2的16个窗口，不用held-out被试1标签求梯度。默认仅对这个诊断批次每类选8窗口，控制局部类别先验；这是探针取样，不修改原训练集、标签或参考总体。若该被试没有足够两类窗口，明确停止，不自动换样本。此批来自训练总体，未保证恰好都在当时trial平衡选中的优化子集中；不是完整训练集梯度估计。两份checkpoint使用同一批、seed42。

原优化器batch由trial窗口组构成，不保证每batch类别均衡；受控均衡探针不能直接替代原历史batch，也不重放其随机状态。范数结果只用于定位量级和路径，不能从一个局部批次推出所有训练批次的衰减占比。

默认mode=train_frozen_bn：使用原训练模式的Gumbel和dropout，但沿用原BN冻结，代码显式核查所有BN为eval。**训练模式前向不等于训练更新**；仅autograd.grad返回梯度，不调用backward、不写parameter.grad、不创建优化器、不step。原模型state_dict和文件哈希均检查不变。

前向用fd_mode=none跳过不参与当前真实损失的do参考积分，显式用原S_train。原事实CE/KL/发放率在full中均于参考积分前产生，这个单批探针不改变其定义；不把它当作重训消融，也不把观察标签用于do监督。跨批训练的RNG消耗差异不是本探针在复现的对象。

分开记录原权重下的CE、CNS KL、ANS KL、发放率梯度，及原总loss梯度。DEAP默认fire_weight=0，因此加权发放率梯度为0是预期。总梯度裁剪系数只按原公式以float64估计，不实际裁剪，也不承诺与PyTorch foreach逐位相同。之后比较现有Adam的weight_decay×参数方向；这不是新增训练损失。

逐参数与逐模块区分None、精确零与非零梯度。实际Adam会跳过grad=None的参数，不能把这种参数的数学L2方向当成实际更新。eval模式下图argmax没有任务梯度是预期；默认训练模式则沿用原可微Gumbel。

范数比不等于真实参数更新比：还受方向、历史moment、bias correction、eps影响；记录moment范数和step但不计算或执行新的参数更新。某项CE范数为0时比值为null并有明确标记，不制造无穷大数值。

## 4. 服务器完整执行命令

使用dslenv（或E1b实际使用的同一环境），不要重建环境。独立工具worktree，不切换原项目分支，不覆盖E1/E1b。以下在同一Xshell会话运行：

```bash
cd /home/fss/NeuroFDAN
python -c "import sys, torch, numpy; print('解释器：', sys.executable); print('PyTorch：', torch.__version__)"

export E2_TOOL_DIR=/home/fss/NeuroFDAN-e2-tools
export E2_RESULT_DIR=/home/fss/NeuroFDAN/outputs/phase5_probes/deap_arousal_fold01_epoch37_e2_001

(
set -euo pipefail
cd /home/fss/NeuroFDAN
git status --short
git branch --show-current
git worktree list
git fetch origin refs/heads/feat/neurofdan-e2-gradient-audit:refs/remotes/origin/feat/neurofdan-e2-gradient-audit
if [ ! -e "$E2_TOOL_DIR" ]; then
    git worktree add --detach "$E2_TOOL_DIR" origin/feat/neurofdan-e2-gradient-audit
fi
test "$(git -C "$E2_TOOL_DIR" rev-parse HEAD)" = "$(git rev-parse origin/feat/neurofdan-e2-gradient-audit)" || {
    echo '工具版本与远端不同；停止，不强制覆盖。'; exit 1;
}
test -z "$(git -C "$E2_TOOL_DIR" status --porcelain)" || {
    echo '工具有本地修改；停止，不覆盖。'; exit 1;
}
mkdir -p outputs/phase5_probes
mkdir "$E2_RESULT_DIR"
bash "$E2_TOOL_DIR/scripts/phase3_run_logged.sh" "$E2_RESULT_DIR/train.log" \
  python -u "$E2_TOOL_DIR/scripts/probe_gradient_components.py" \
  --root_dir /home/fss --dataset DEAP --task arousal \
  --held_out_subject 1 --probe_subject 2 \
  --seed 42 --device cpu --samples 16 --sampling balanced --expected_epoch 37 \
  --mode train_frozen_bn --checkpoints best last \
  --fold_dir /home/fss/NeuroFDAN/outputs/phase4/DEAP/arousal/seed42/deap_arousal_fold01_full/fold01 \
  --e1b_summary /home/fss/NeuroFDAN/outputs/phase5_probes/deap_arousal_fold01_epoch37_e1b_001/probe_summary.json \
  --output_dir "$E2_RESULT_DIR"
cat "$E2_RESULT_DIR/train.log.exit"
)
```

当前终端已显示(dslenv)时直接复用。如果换了Xshell窗口，先用原来的方式激活dslenv；不知道其环境路径时不猜测venv或重装依赖。缺权重/身份或哈希不符时停止，不重训或用随机模型代替。再次运行须统一改e2_001为新编号。

这次不运行mainNeuroFDAN.py，没有训练epoch循环、GPU实验或完整LOSO。结果准确位置：

```text
/home/fss/NeuroFDAN/outputs/phase5_probes/deap_arousal_fold01_epoch37_e2_001/
  probe_summary.json
  train.log
  train.log.exit
```

正常应显示python_exit=0和tee_exit=0；JSON内每个条件模型状态及checkpoint文件不变。全部只保存统计，没有权重、梯度数组或原信号。

## 5. dry-run和正式上传

train.log会自动识别，下方显式给出；源目录独立，server-diagnostics追加唯一记录：

```bash
cd /home/fss/NeuroFDAN
bash /home/fss/NeuroFDAN-e2-tools/scripts/upload_run_diagnostics.sh \
  /home/fss/NeuroFDAN/outputs/phase5_probes/deap_arousal_fold01_epoch37_e2_001 \
  --log /home/fss/NeuroFDAN/outputs/phase5_probes/deap_arousal_fold01_epoch37_e2_001/train.log \
  --dry-run
```

查看manifest及脱敏文件后：

```bash
cd /home/fss/NeuroFDAN
bash /home/fss/NeuroFDAN-e2-tools/scripts/upload_run_diagnostics.sh \
  /home/fss/NeuroFDAN/outputs/phase5_probes/deap_arousal_fold01_epoch37_e2_001 \
  --log /home/fss/NeuroFDAN/outputs/phase5_probes/deap_arousal_fold01_epoch37_e2_001/train.log
```

回传工具打印的固定commit链接。unknown命名仍是当前上传器的纯探针元数据限制，不需要创建虚假的训练fold.json。

## 6. E2之后的决策

若早期仍有输入变化的参数上，L2方向已显著超过任务梯度，并伴随卷积/中介权重收缩，再提出仅改变weight_decay的短程单变量对照；必须明确修改位置、默认值和预期影响，不直接实施。若任务梯度本身先断开或正则不占主导，优先查对应梯度路径，不用盲调衰减解释所有问题。没有证据支持删除CNS→ANS、添加Mc→Y或更换SCM。

仅best/last两个时间点和一个局部批不能证明整个训练历史的唯一因果原因；若需要更多窗口或另一训练被试，要根据这次结果有针对性地增加，不先启动大规模重复。

## 7. 静态审查与已知边界

本轮仅读取已上传的E1b数据并做代码/语法/Git差异审查。没有云端运行E2、CLI、模型或单元测试，没有服务器E2数值结果，也不声称已修复模型或提升ACC。没有修改ModelTrain、NeuroFDAN、frontdoor_scm、Datasets、loss或主训练CLI。

实际静态检查：两个探针Python文件通过AST解析；本说明的三个bash代码块通过bash -n；git diff --check通过；核心模型、训练器、数据集和上传器相对E1b基线无差异。上述检查不等于E2运行测试，E2的运行兼容性和数值结果待服务器验证。
