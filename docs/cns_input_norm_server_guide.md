# CNS入口LayerNorm实现与N0服务器验证

用户已同意本试验；正式分类对比优先ACC，保留F1和混淆矩阵诊断。Balanced Accuracy可作为旧产物字段保留，不作为新试验的主报告或排名标准。原test-selected checkpoint规则（ACC、同分F1）没有改变。

## 实际改动

- frontdoor_scm.py：新增FrontDoorConfig.cns_input_norm=none|layernorm，默认none；CNSMediator.distribution在原encoder前可选nn.LayerNorm(input_dim,eps=1e-5,elementwise_affine=False)。forward及oracle条件似然共用该入口。ANS、参考编码、OutcomeHead和积分公式不变；一维输入拒绝开启，避免常数退化。
- mainNeuroFDAN.py、ModelTrain.frontdoor_config：新增真实CLI开关`--cns_input_norm layernorm`；默认none。加载旧checkpoint时缺少字段解释为none；none权重不能静默按layernorm续训。
- scheme_a_fit_check.py：新增`--fit_gate_metric acc`，新归一化拟合以ACC≥0.8、两类最小Recall≥0.6、积分观察NLL低于该批最优常数至少0.05、3次MC均满足为工程验收。两类Recall防止多数类预测伪装高ACC，不要求论文汇报Balanced Accuracy。旧调用默认门槛保留，不改历史结论，也不改变正式选模。
- 原KL对照的只读兼容守卫对缺省norm=none做规范化；摘要显示新开关。旧模型参数键及参数个数不增加。
- NeuroFDAN.py、Datasets.py、loss.py、classification_objectives.py不修改；原始数据处理、BN、CNS→ANS、随机采样、损失权重和正式选模不变。

令每窗z_c为展平的CNS骨干表示，\(\tilde z_c=(z_c-\bar z_c)/\sqrt{\operatorname{var}(z_c)+10^{-5}}\)，只将它送入原CNS encoder。这是数值参数化候选，不是新增因果识别条件。不强制posterior mu分离或sigma缩小，不新增真实数据loss、教师或辅助头。

## 本次先执行N0

遵守合成验证在真实数据修改试验之前的顺序。本轮交付模型实现及可选真实CLI，但先只运行下述CPU合成试验，不自动运行DEAP/HCI或LOSO。合成结果需审查，不能因程序退出0便宣称方法性能有效。

小规模配对Gaussian SCM：U为2维、Xc为16维、Xa为8维、Mc/Ma均4维；独立标准Gaussian外生量，U→X/Y，Xc→Mc，Xa/Mc→Ma，Ma/U→Y，Bernoulli logistic结果。全部实际矩阵记录在probe_summary。无tanh生成/normal拟合的分布错配。训练256、独立测试128、batch64、seed42、Adam lr0.001、每组200步、CPU单线程，固定最后模型不选测试最佳。合成超参不改真实数据lr2e-5/batch400等配置。

三组共用同一数据、初始化、批次及评估随机流：

1. observation_only：none；只能接收Xc/Xa/Y，用现有积分观察NLL+KL+fire。
2. observation_norm：layernorm；相同观察监督，仅CNS入口不同。
3. oracle_mediator：none；额外允许真实Mc/Ma条件似然和真实Ma事实CE，作为额外信息参考，有限训练下不保证数值上界。它不进入真实数据方法。

fit严格检查字典字段；U、do概率和独立测试数据均不传入fit。三组X/Y训练指纹相同，参数初始化指纹需相同；oracle的中介真值只在oracle分支访问。所有组评价时才使用真实中介/条件均值/干预概率，坐标恢复误差不等于一般潜变量可识别证明。

记录ACC、正类F1、两类矩阵、观察NLL/Brier、CNS条件均值/条件NLL恢复、给定真实Mc的ANS恢复，以及16点已知do概率MAE/MSE和3次MC波动。do真值由独立8192次直接生成积分并记录SE，不将观察分类提高等同干预误差下降。Oracle多出的监督项只属于合成参考组，真实总损失没有增加项。

另记录幅度丢失反例：Xc=pattern+a*ones，LayerNorm抹去a，而真实Mc依赖mean(Xc)，两个do点的真值概率不同。概率差给出的误差下界是MC估计，不能当作严格数值证书。再记录既有“同观察分布、不同干预分布”的解析反例，保留一般潜中介非识别边界。

## 直接执行：测试→N0→自动上传

```bash
cd /home/fss/NeuroFDAN
conda activate dslenv
(
set -euo pipefail
if [ -z "${GIT_SSH_COMMAND:-}" ]; then
    export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
fi
git status --short
timeout 600 git fetch --no-tags origin feat/neurofdan-cns-input-norm
NORM_SHA="$(git rev-parse FETCH_HEAD)"
NORM_TOOLS=/home/fss/NeuroFDAN-cns-input-norm-tools
if [ ! -e "$NORM_TOOLS" ]; then git worktree add --detach "$NORM_TOOLS" "$NORM_SHA"; fi
test "$(git -C "$NORM_TOOLS" rev-parse HEAD)" = "$NORM_SHA"
test -z "$(git -C "$NORM_TOOLS" status --porcelain)"
git -C "$NORM_TOOLS" log -1 --oneline
bash "$NORM_TOOLS/scripts/run_cns_norm_n0_and_upload.sh" \
    /home/fss/NeuroFDAN/outputs/phase6/cns_input_norm/n0_gaussian_001
)
```

runner先以CUDA_VISIBLE_DEVICES=''执行完整已有轻量单测（新增7项归一化/字段/ACC门槛测试），通过才执行：

```bash
# 已由runner调用，不要重复运行同名目录
python -u /home/fss/NeuroFDAN-cns-input-norm-tools/scripts/cns_norm_synthetic.py \
    --output_dir /home/fss/NeuroFDAN/outputs/phase6/cns_input_norm/n0_gaussian_001 \
    --seed 42 --steps 200
```

这是CPU合成训练，不读取DEAP/HCI、旧checkpoint或服务器数据路径；dataset=synthetic、device=cpu、batch=64及诊断内容固定记录，不编造此脚本没有的dataset/label/workers CLI。真实DEAP/HCI阶段尚未启动，本轮无需mainNeuroFDAN.py。

结果在n0_gaussian_001/probe_summary.json；测试及训练日志、.exit在同路径后缀_logs目录。相同结果目录拒绝覆盖。自动使用当前已验证的上传器，先dry-run、后正式上传，600秒超时、16MiB/64MiB，结果追加到server-diagnostics独立run目录，不上传模型或原始数据。独立bundle和upload_state.json位于结果路径后缀_bundle_日期_PID目录，上传日志在_logs中，失败也尝试上传已有测试日志。不需要手工附日志或检查manifest。

## 回传及后续

成功只回传最后SHA/链接即可，失败回传终端输出。N0不会自动触发真实训练。下一步审查结果后，先做已同意的同400窗N1（beta0，只有CNS norm变化），用ACC/F1、两类Recall、NLL和MC稳定性判断；再验证正KL条件。N1必须重置seed42重新训练，不热插旧权重；旧的“只能改beta、首步NLL相同”守卫不适用，需要核对相同参数初始化指纹/来源并明确仅norm不同。随后才恢复完整单fold及LOSO，不用本次合成ACC作为论文真实数据成绩。

云端只做AST、bash -n和Git差异检查，没有运行模型、CLI、单元测试、合成/真实训练或GPU。新增7项及已有单测由上方服务器命令验证。独立分支交付，不修改main，不创建PR或合并；旧实现默认none保留，所有旧结果保留。
