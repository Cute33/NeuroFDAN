# 方案A单类结果审查与KL单因素验证

## 已核验的服务器证据

结果提交：`bb1c362110f038a1df500f339a5632ec5b454cff`。
目录：`runs/2026-11-12_230008_unknown_unknown_run_33be8365`。
8项载荷的长度及SHA256全部一致，skipped为空。missing中的run_summary.json、run_report.txt是未产生fold时的通用根报告，实际config、probe_summary和日志齐全。

原拟合源码是30152089，不是上传时主项目worktree的9ef2b05；上传工具源码为57d63d1。原181项服务器测试通过；本次上传恢复19项测试在1.101秒通过，Git fetch为6.9秒，push和远端核验成功。本次不用重复上传旧结果。

这是排除被试2后，从自然训练总体固定抽取400窗的**训练子集拟合**，不是held-out测试或LOSO成绩。标签计数[153,247]，正类比例247/400=0.6175。所有7个检查点、每点3次MC均为：

```text
真实行、预测列，顺序[0,1]
[[0,153],
 [0,247]]
ACC=0.6175，Balanced Accuracy=0.5，Macro-F1=0.381762
```

末端事实路径也全部预测1。正类F1=0.763524不能表示两类判别成功。

最优常数预测NLL为0.6652747443；第300步三次评价NLL为0.6652688384、0.6652835011、0.6654773951，均值0.6653432449。正类概率均值约0.6173，跨窗std约0.00164–0.00169。训练态主NLL=0.6653127074。这支持“本次主要学到类别先验”，没有显示出足以区分类别的有效收益；不是单凭ACC推测单类。不能从这三次有限MC的差异推断逐窗口MC方差或置信区间。

原流程完成300步，未通过固定门槛，退出3；没有执行20轮主fold。

## 中介与梯度：证据和边界

第300步eval的CNS均值跨输入std=0.001159624，sigma跨输入std=0.000910235，posterior sigma均值=0.974309905；CNS输入条件变化相对采样尺度很小，但不严格为0。CNS KL=0.048247736，也没有完全趋于零。其均值变化从第1步0.000739651略有上升，不能描述成所有统计量都持续下降。

ANS对应统计约0.045648613、0.023290416、0.993443444；ANS统计先对有限S平均，仍含Mc采样误差，不等于全部来自稳定Xa依赖。不能把flatten(B,S)的mu_a_spread=0.17827当作纯跨输入变化。

CNS平均发放率约2.44e-6，ANS约0.0956。源码GaussianMediator.distribution仍拼接rates与平均current，所以低发放率不等于整条posterior路径断开；不能凭它擅自修改LIF阈值。

第300步实际训练前向、裁剪前的CNS分项梯度：

| 分项（已乘实际权重） | L2 | 相对主NLL梯度 | 与主NLL梯度余弦 |
|---|---:|---:|---:|
| 主积分观察NLL | 0.000208895 | 1 | 1 |
| CNS KL | 0.000745848 | 3.57 | -0.008 |
| ANS KL经过CNS路径 | 0.000580521 | 2.78 | 0.049 |
| fire | 0.000512388 | 2.45 | 0.112 |
| 总loss | 0.000628394 | 3.01 | 0.459 |

这说明CNS更新方向受到正则项明显影响；方向并非一致反向，不能说“KL直接抵消任务梯度”或凭范数断言Adam实际更新比例。ANS主任务梯度约0.003134、CNS→ANS约0.001177、Head约0.047699，均非零，不支持完全断图。首步CNS任务梯度约0.000739，末端降低，但有限检查点不构成长期渐近结论。

梯度来自更新前train模式，posterior/矩阵来自更新后eval模式，不能当作严格对齐测量。现有证据支持验证KL竞争，不足以证明它是唯一原因；fire、输入尺度、Gumbel图采样、潜变量采样与Head学习仍可能共同作用。

## 下一步：只做一次KL关闭诊断

基线目标：

\[
L_A=-B^{-1}\sum_i\log\hat p_{do,i}(y_i)+\beta_t(KL_c+KL_a)+0.028L_{fire},
\quad\beta_t=0.001\min(e/10,1).
\]

对照目标只将beta设为0：

\[
L_{A,KL0}=-B^{-1}\sum_i\log\hat p_{do,i}(y_i)+0.028L_{fire}.
\]

KL数值仍计算并记录，原损失代码和默认beta不变。零权重只用于这份独立诊断，不作为正式方法的最终配置；不增加辅助头、标签平衡、教师或中介真值。监督仍是观察标签，不是无偏do真值监督，也不解决一般潜在中介不可识别。

同一fold2、seed42、400个窗口、2048参考成员、初始化、lr2e-5、LIF256、fire0.028、S/K=4/16和16/64、BN策略及裁剪规则。只比较一次最多300步的私有拟合，不启动主fold或LOSO，也不延长到500步。固定批次诊断是反复拟合同400窗；不能称为全训练总体的epoch或全局shuffle遍历。

新增守卫逐项核对原参数，只允许beta和输出位置不同；核对相同查询/参考来源ID、标签数、完整总体步数、模型配置和门槛。首步主NLL须与原0.6838082075相差不超过2e-6，失败立即停，不消耗300步。ID和首步NLL匹配是复核证据，不是对所有原始数据文件逐字节不变的证明。

原3次MC门槛保持BA≥0.8、最小Recall≥0.6、NLL低于常数基线至少0.05；通过可提前结束，但不会自动运行20轮。

## 结果如何决定后续

1. 若KL关闭后突破先验、两类Recall及NLL显著改善，说明当前预算下去掉KL具有正面作用；不证明KL是全部问题。下一阶段研究保留正KL的权重/启动时机，再做独立完整单fold验证，不直接将beta0当作最终论文方法。
2. 若仍全部预测1、NLL接近0.665275，说明关闭KL不足以解决当前问题。停止本轮beta扫描及长训；下一步先讨论中介输入尺度/驱动表征的最小修改，再经用户确认实施关键模型文件。不能把本对照失败解释成所有正则、所有参数或框架理论上均不可行。
3. 若有概率/NLL收益但未通过固定门槛，如实记录部分改善，不切换主输出、不更换argmax、不自动降门槛放行。

当前研究主线仍为CNS→ANS层级概率中介及前门形式主输出。分类能力是必要工程基础；重复训练集拟合成功不等于跨被试泛化或真实生理机制被识别。可用于论文的主分类表仍需冻结配置后的完整LOSO；此处只能标注为开发期训练子集诊断。

## 代码及服务器运行

核心NeuroFDAN.py、frontdoor_scm.py、ModelTrain.py、mainNeuroFDAN.py、loss.py、Datasets.py和classification_objectives.py均不修改。

- scheme_a_fit_check.py只增加可选基线核对和末端私有权重保存；默认优化/评价路径不变。
- scheme_a_control_evidence.py执行协议守卫和配对比较。
- run_scheme_a_kl_control_and_upload.sh：16项CPU接口测试→单次私有拟合→摘要/比较→上传；没有mainNeuroFDAN.py调用。
- 新增7项测试覆盖只改beta、预算/来源/初始化守卫、负结果原样保留、权重原子创建拒绝覆盖以及beta0仍保留任务与fire梯度；连同原方案A9项共16项，待服务器执行。云端不运行测试、CLI、模型或训练。

只执行这一块，沿用dslenv和原数据：

```bash
cd /home/fss/NeuroFDAN
conda activate dslenv
(
set -euo pipefail
if [ -z "${GIT_SSH_COMMAND:-}" ]; then
    export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
fi
git status --short
timeout 600 git fetch --no-tags origin feat/neurofdan-scheme-a-kl-control
KLC_SHA="$(git rev-parse FETCH_HEAD)"
KLC_TOOLS=/home/fss/NeuroFDAN-scheme-a-kl-tools
if [ ! -e "$KLC_TOOLS" ]; then git worktree add --detach "$KLC_TOOLS" "$KLC_SHA"; fi
test "$(git -C "$KLC_TOOLS" rev-parse HEAD)" = "$KLC_SHA"
test -z "$(git -C "$KLC_TOOLS" status --porcelain)"
git -C "$KLC_TOOLS" log -1 --oneline
bash "$KLC_TOOLS/scripts/run_scheme_a_kl_control_and_upload.sh" \
    /home/fss \
    /home/fss/NeuroFDAN/outputs/phase6/scheme_a_kl_control \
    deap_arousal_fold02_scheme_a_beta0_fit300_001 \
    /home/fss/NeuroFDAN/outputs/phase6/scheme_a/DEAP/arousal/seed42/deap_arousal_fold02_scheme_a_001/probe_summary.json
)
```

完整启动参数已写入runner的显式python命令，包含dataset/label/root/output/run/seed/device/fold/epochs/batch/workers/诊断等级和所有原采样及损失参数。`--epochs 20`仅保留原拟合入口的元数据，实际预算由`--fit_max_steps 300`控制，不执行20轮。无需另跑mainNeuroFDAN.py。

新结果根：`outputs/phase6/scheme_a_kl_control/DEAP/arousal/seed42/deap_arousal_fold02_scheme_a_beta0_fit300_001`。

probe_summary.json保存实际步数、矩阵、分项梯度和posterior；run_summary.json为标注独立格式的配对比较，不是LOSO报告。日志为train.log、interface_tests.log、summary.log、comparison.log及各自.exit。config.json记录流程状态。独立private_fit_last.pt仅留服务器，格式neurofdan_private_fit_v1，不含原始信号、不含正式epoch或Adam续训状态，不能作为正式format_version=2 checkpoint续训。保存只在新目录原子创建，已有目标拒绝覆盖；后续只读探针可直接读取它，避免为补诊断重复拟合。

## 自动上传及回传

上述runner同时执行服务器操作和上传两组动作，无需用户再抄上传命令。使用与已通过19项测试/实际上传成功的57d63d1逐字节相同的上传器，先`--dry-run`，再正式上传；600秒网络超时，16MiB单文件/64MiB总额。自动附带train.log及其余已生成日志和退出收据，白名单不会上传private_fit_last.pt。

唯一上传日志与bundle位于新输出根的upload_logs和upload_bundles，保留upload_state.json并打印准确路径。远端追加新的server-diagnostics/runs目录，不覆盖旧结果。测试或拟合报错时也尝试回传已有文件；上传成功与拟合通过分别判断。流程退出3仍表示预算内未通过，不自动重跑。

成功回传最后打印的SHA及GitHub链接即可。网络失败直接回传终端摘要和收据，不重训。无需手工核对manifest或摘抄轮数。GitHub无需切换main、创建PR或合并。
