# KL关闭负结果与私有权重只读审查

结果SHA：a334d86390f1a37076e738e84b14ba53040531d2，目录runs/2026-11-13_114556_unknown_unknown_run_41890927。9项载荷长度/SHA256全部核验一致，skipped为空，missing仅run_report.txt。训练源码bfe108e；300步完成、原摘要未改、查询/参考来源一致、首步主NLL一致；服务器16项测试通过，比较报告完整，无训练异常。没有执行20轮或LOSO。

## 判断

beta=0后21次MC评价仍全部预测1，训练子集矩阵[[0,153],[0,247]]，BA=0.5。末端三次NLL为0.6652335525、0.6652796865、0.6654849052，仍接近原400窗常数基线0.6652747443。关闭KL不足以解决当前300步预算的判别问题，停止继续降低beta或增加相同配置轮数；不能据此声称KL在所有配置下都无影响。

| CNS统计 | beta=0.001 | beta=0 |
|---|---:|---:|
| posterior均值跨输入std | 0.001159624 | 0.003696448 |
| sigma跨输入std | 0.000910235 | 0.002621138 |
| 平均posterior sigma | 0.974309905 | 0.938887078 |
| 平均条件KL | 0.048247736 | 0.650649071 |

信息相关统计确实变化，却未转化为分类收益；这进一步说明均值离散程度和KL单项都不能作为有效中介或分类能力的充分证据。ANS均值变化约0.06462，仍含有限Mc抽样误差。CNS发放率为0，但posterior仍通过平均current接收输入，不能称整条梯度断开。

末端CNS任务梯度0.000367465、fire梯度0.000690638，余弦-0.0224，总梯度与任务余弦0.454。fire仍可能影响训练方向；本结果没有排除这一因素，也没有证明输入尺度是唯一原因。ANS、CNS→ANS和Head任务梯度均非零。不再凭一个范数选择另一轮长训。

## 本次操作

利用已保存private_fit_last.pt，不重新拟合。只取原400窗来源的前32个，不按类别筛选；保留原2048参考成员，只读取训练被试。使用原eval S=16/K=64，seed42起的4次MC；不是原400窗整体成绩的重复评价。

- 记录原始输入、骨干输出、CNS驱动、LIF平均current/rates及posterior的RMS、跨样本std和样本内维度std。
- 对4次完整原前门预测做有限R方差分解：Var_(r,i)(p)=Var_i(mean_r p)+mean_i Var_r(p)。第一项仍含有限R误差，不能直接叫真实输入信号；不把这些重复构成的方差当作可靠置信区间。
- ANS分别使用共同CNS标准噪声和固定零Mc，读取原conditional distribution的mu/sigma；区分输入条件差异与独立Mc噪声影响。零Mc可能偏离支持域，只用于函数响应检查。
- 固定实际抽样Ma/参考组合，先重现原概率积分，再分别循环移位Ma和参考，记录概率L1变化；不修改阈值、不将置换结果当作物理干预。Ma移位同时搬移采样噪声，不能独立证明类别信息或因果依赖。
- model.eval、no_grad、无optimizer；核对模型参数/BN缓冲区、原checkpoint文件哈希不变，移除hook。只写新目录。

若骨干已有变化而CNS drive/current/后验明显收缩，优先讨论中介入口尺度校准（例如LayerNorm）这个最小参数化候选；若共同噪声下ANS变化消失，优先讨论层级噪声传播；若中介变化能到达Head而积分输出仍弱，则聚焦条件结果学习和边缘化。候选都需看本次结果，涉及frontdoor_scm.py、NeuroFDAN.py等关键文件先向用户说明并获准。本轮不实施结构修改，不新增损失，不永久删除KL或fire，不启动正式LOSO。

## 服务器命令：一次执行只读操作及自动上传

```bash
cd /home/fss/NeuroFDAN
conda activate dslenv
(
set -euo pipefail
if [ -z "${GIT_SSH_COMMAND:-}" ]; then
    export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
fi
git status --short
timeout 600 git fetch --no-tags origin feat/neurofdan-private-signal-audit
SIGNAL_SHA="$(git rev-parse FETCH_HEAD)"
SIGNAL_TOOLS=/home/fss/NeuroFDAN-private-signal-tools
if [ ! -e "$SIGNAL_TOOLS" ]; then git worktree add --detach "$SIGNAL_TOOLS" "$SIGNAL_SHA"; fi
test "$(git -C "$SIGNAL_TOOLS" rev-parse HEAD)" = "$SIGNAL_SHA"
test -z "$(git -C "$SIGNAL_TOOLS" status --porcelain)"
bash "$SIGNAL_TOOLS/scripts/run_private_signal_audit_and_upload.sh" \
    /home/fss \
    /home/fss/NeuroFDAN/outputs/phase6/scheme_a_kl_control/DEAP/arousal/seed42/deap_arousal_fold02_scheme_a_beta0_fit300_001/private_fit_last.pt \
    /home/fss/NeuroFDAN/outputs/phase6/private_signal_audit/deap_arousal_fold02_beta0_001
)
```

无需mainNeuroFDAN.py或scheme_a_fit_check.py。脚本先运行4项CPU小测试，再调用audit_private_signal.py，参数完整为--root_dir、--checkpoint、--output_dir、--device cuda:0。dataset/label/fold/seed和实际300步来自强制检查的checkpoint身份；本次epochs=0、optimizer step=0；32窗批量推理、4次重复，S/K沿用权重配置，无DataLoader worker或训练日志等级参数，不编造其CLI开关。

结果源目录为命令最后一个参数，包含config.json、probe_summary.json。日志在同路径后缀_logs目录，包含tests.log、train.log及各自.exit，train.log只是沿用日志名称，实际没有训练。自动使用与57d63d1相同上传器，先dry-run、再正式上传，--git_timeout 600、单文件16MiB/总64MiB；唯一bundle在结果路径后缀_bundle_日期_PID目录，保留upload_state.json，旧结果不覆盖、权重不上传。成功只回传SHA和GitHub链接；失败回传终端日志，不重训。

## 交付边界

只新增诊断脚本、自动上传包装、4项测试及本文；核心模型/训练/损失/数据文件不变。云端仅读取既有服务器JSON、校验哈希及AST/bash/Git静态检查，不运行模型、CLI、单测或GPU。新测试与推理由用户服务器执行，不能把旧16项通过当作新4项已通过。没有main修改、PR、合并。研究主线保持CNS→ANS层级概率中介和probs_do主输出；局部功能响应不证明真实生理中介已识别。
