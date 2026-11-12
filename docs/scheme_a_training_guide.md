# 方案A：积分预测的观察标签风险与分阶段服务器运行

## 1. 为什么替换主损失

原事实主项为单次或S次平均的

\[
L_{obs}=\frac1{BS}\sum_{i,s}-\log h_\theta(y_i\mid m_{a,i}^{(s)},g(x_i)).
\]

这里的m_a来自模型按当前骨干表示Z独立重采样。对固定编码器，训练增广分布为P(Z,Y)q_phi(M|Z)，所以Head可以通过自身参考Z预测Y，不必使重采样中介有用。在这个允许的解下，替换为训练总体参考并边缘化可能产生接近训练类别比例的常数预测。它是监督设计风险，不证明当前所有失败的唯一原因。

此前fire、lr、训练预算和CE采样对照没有改变该期望目标。本轮用户明确批准方案A，改变此前“观察标签只监督事实路径”的约定。新的主项为

\[
\hat p_i(y)=\frac1{SK}\sum_{s,k}\mathrm{softmax}
h_\theta(m_{a,i}^{(s)},g(x'_k))_y,
\quad L_{pred}=-\frac1B\sum_i\log\hat p_i(y_i).
\]

分类项在参考及中介概率平均之后计算。它不同于平均每个交叉配对的CE，也不同于CE(probs_do,Y)。实现直接复用已有logits_do=log(probs_do)做NLL，不另写前门估计器。

如果只有参考分支提供总体先验，最终预测接近常数，L_pred就接近H(Y)；当当前输入存在可学习标签信息、模型能利用它时，分类风险可以低于这个常数基线。对理想不限函数类，最优观察风险为H(Y|X)，两者差为I(X;Y)。这说明让最终预测器直接承担分类误差具有依据，不是保证有限网络、有限数据或跨被试ACC一定改善。

令p_j为某个(s,k)组合对真实类别的概率，积分NLL的梯度可写成

\[
\nabla[-\log\bar p_y]= -\sum_j\frac{p_j}{\sum_l p_l}\nabla\log p_j.
\]

梯度经现有积分图传入OutcomeHead、ANS、CNS→ANS及CNS。原先分类项只评分自身参考配对；新项直接评分最终部署的组合预测。梯度非零、训练NLL下降与BA提高仍须分别验证。

## 2. 总损失及科学边界

\[
L_A=L_{pred}+\beta_t(KL_c+KL_a)+0.028L_{fire},
\qquad\beta_t=0.001\min(e/10,1).
\]

分类主项系数固定为1，显式lambda_obs=0，原事实CE仅诊断；不同时叠加两个分类任务。仍是分类、KL、发放率三类项，没有教师、rec/div、锚点或新辅助头。

**标签是观察情绪标签，不是真实do标签。** 新目标优化有限S/K积分形式预测器的观察风险，不构成无偏干预概率监督。有限MC后取log还存在估计误差。真实潜在中介不可观测、条件分布及参考表示的识别假设仍未验证。不能仅凭分类提高宣称消除了混杂或识别真实生理因果机制；相关合成干预误差验证仍需独立完成。

保留SCM、联合处理、CNS→ANS电流、OutcomeHead不接收Mc、softmax后两级平均。NeuroFDAN.py、frontdoor_scm.py、Datasets.py不修改。旧factual模式默认保留，旧损失文件和分支保留。方案A只新增独立目标模块及训练开关，不改骨干或中介层。

## 3. 配置与兼容

新增--classification_objective factual|integrated_observational，默认factual。方案A要求--fd_mode full --lambda_obs 0，防止误叠加旧CE或在none模式中伪造积分损失。新目标必须新run从头训练；resume校验主目标，不能从旧factual权重及Adam状态静默续训。

当前受控配置为DEAP/arousal/fold2，seed42，lr2e-5，Adam/wd0，train/eval400，LIF256、hidden128、中介64，KL0.001/10轮，fire0.028/目标0.05，S/K=4/16与16/64，N_ref2048，ref_dim0，dropout0.3，4个内部驱动步；全局窗口shuffle，自然训练总体，不做类别平衡。factual_ce_mode=all_samples只决定事实CE诊断的计算，方案A不将它加入总loss。

probs_do仍是主要评价及选模输出；probs_obs仍取原首样本路径，仅作对照。没有validation，test-selected ACC/F1规则不变，存在测试集选模偏差。不能把旧none模式与新full模式的差异直接称为纯前门消融，训练目标不同还需单独控制。

## 4. 分阶段预算与预定门槛

1. 先在服务器执行全部轻量单元测试。新增9项覆盖NLL语义、替换而非叠加、旧目标逐位兼容、真实集成模型梯度、CLI/续训守卫、拟合门槛和训练入口/保存记录。
2. 固定自然训练总体中seed42随机选400窗，不读取测试数据、不依标签挑选或平衡；参考仍为全部未平衡训练总体构建的2048库。
3. 第1步做真正的batch400完整反传和Adam更新，检查任务NLL到达CNS、ANS、CNS→ANS、Head，记录显存和状态。OOM/NaN/无任务梯度立即停止，不静默缩batch或S/K。
4. 私有拟合最多300步，每50步检查；KL等效epoch=1+(step-1)//31，沿用原每轮31步的系数映射，不把私有step当成LOSO epoch。每次评估使用原S_eval16/K_eval64，训练seed42不变，隔离随机流后从seed42连续作3次MC抽样。
5. 至少50步后，3次MC都满足训练子集BA≥0.8、两类最小Recall≥0.6、积分NLL低于该子集最优常数NLL至少0.05，才通过。门槛是工程拟合验收，不是因果证明或正式测试成绩。预算内不通过不代表理论上永远学不会，但不晋级长训。
6. 通过后独立新进程、seed42从头运行完整fold20轮，不读取旧checkpoint或私有拟合权重。仍按held-out规则保存best，拟合门槛不参与checkpoint选择。

只有20轮主输出表现出持续训练判别后，才考虑其他固定fold或扩展轮数。先不运行完整LOSO，不再自动进行100/300/500轮循环。

## 5. 服务器一条工作流：测试→验收→通过才训练→自动上传

继续已激活dslenv，不重装环境。新worktree不切换/home/fss/NeuroFDAN原分支，输出独立。下面脚本完整执行两组动作：服务器运行与GitHub结果上传。测试、拟合或训练失败时也会尝试上传已有日志/退出收据。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
if [ -z "${GIT_SSH_COMMAND:-}" ]; then
    export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
fi
git status --short
git worktree list
timeout 600 git fetch origin feat/neurofdan-scheme-a
SCHEME_SHA="$(git rev-parse FETCH_HEAD)"
SCHEME_TOOLS=/home/fss/NeuroFDAN-scheme-a-tools
if [ ! -e "$SCHEME_TOOLS" ]; then git worktree add --detach "$SCHEME_TOOLS" "$SCHEME_SHA"; fi
test "$(git -C "$SCHEME_TOOLS" rev-parse HEAD)" = "$SCHEME_SHA"
test -z "$(git -C "$SCHEME_TOOLS" status --porcelain)"
git -C "$SCHEME_TOOLS" log -1 --oneline
command -v python
python --version
bash "$SCHEME_TOOLS/scripts/run_scheme_a_and_upload.sh" \
    /home/fss /home/fss/NeuroFDAN/outputs/phase6/scheme_a \
    deap_arousal_fold02_scheme_a_001
)
```

本次只需执行这一块。正常最后回传一个SHA和GitHub链接即可，无需手抄轮数或manifest。

完整主函数命令如下，已由上方脚本在验收通过后调用，**不要绕过验收或重复执行同名run**：

```bash
cd /home/fss/NeuroFDAN
SCHEME_TOOLS=/home/fss/NeuroFDAN-scheme-a-tools
SCHEME_OUTPUT=/home/fss/NeuroFDAN/outputs/phase6/scheme_a
SCHEME_NAME=deap_arousal_fold02_scheme_a_001
SCHEME_RESULT="$SCHEME_OUTPUT/DEAP/arousal/seed42/$SCHEME_NAME"
bash "$SCHEME_TOOLS/scripts/phase3_run_logged.sh" "$SCHEME_RESULT/train.log" \
    python -u "$SCHEME_TOOLS/mainNeuroFDAN.py" \
    --dataset DEAP --label arousal --root_dir /home/fss \
    --output_dir "$SCHEME_OUTPUT" --run_name "$SCHEME_NAME" \
    --seed 42 --device cuda:0 --held_out_subjects 2 --epochs 20 \
    --lr 0.00002 --weight_decay 0 --batch_size 400 --eval_batch_size 400 --workers 0 \
    --sampling_mode window --fd_mode full --no_balance \
    --mediator_dim 64 --hidden_dim 128 --lif_neurons 256 --dropout 0.3 --ref_dim 0 \
    --time_steps 4 --logvar_min -6 --logvar_max 2 \
    --S_train 4 --K_train 16 --S_eval 16 --K_eval 64 --N_ref 2048 --reference_chunk_size 4 \
    --classification_objective integrated_observational --lambda_obs 0 --factual_ce_mode all_samples \
    --beta 0.001 --beta_warmup_epochs 10 --fire_weight 0.028 \
    --diagnostic_level full --log_detail_interval 5 --loss_gradient_interval 5 \
    --max_train_batches 0 --max_test_batches 0
```

拟合脚本接受同一套主函数参数，额外--fit_max_steps 300 --fit_check_every 50。epochs20是后续主fold配置；私有拟合实际执行步数单独记录，不将其称为完成20轮。

## 6. 输出与自动上传

run目录：`/home/fss/NeuroFDAN/outputs/phase6/scheme_a/DEAP/arousal/seed42/deap_arousal_fold02_scheme_a_001`。

- config.json：工作流各阶段状态及退出码；未通过会明确标记fold20=not_started。
- probe_summary.json：私有拟合实际step、门槛、3次MC、矩阵、NLL、posterior摘要、首batch任务梯度、完整反传及显存；明确不是正式fold成绩。
- interface_tests.log/.exit、fit_check.log/.exit、train.log/.exit：自动收集，train仅在晋级后产生。
- fold02/：通过后产生完整20轮日志、checkpoint、矩阵、中介/塌缩诊断。新加classification_loss/weighted_classification_loss、weighted_obs_loss(应为0)、integrated_observational_nll、BA及Macro-F1；事实CE保留为诊断。
- 分项梯度参考项为integrated_observational_nll，余弦字段为cosine_to_classification，避免仍称相对事实CE。

一份快照同时上传拟合、工作流和主fold的白名单产物，减少重复Git fetch。工具沿用已验证的增强上传器，自动读取各.log.exit，先dry-run再正式上传，600秒网络超时、16MiB单文件/64MiB总量；唯一bundle/上传日志位于输出根的upload_bundles/upload_logs。不上传权重或原信号。

工作流退出码3表示拟合预算内未通过，可能同时已经成功上传；不是自动认定软件错误。若没有fold02无需手工补文件。上传成功和训练成功由不同证据分别判断。

仅网络上传失败且已确认未成功时，独立重传如下；不要重新训练。若push结果未知，先回传候选SHA，由Codex核验远端，避免重复快照。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
SCHEME_TOOLS=/home/fss/NeuroFDAN-scheme-a-tools
SCHEME_OUTPUT=/home/fss/NeuroFDAN/outputs/phase6/scheme_a
SCHEME_RESULT="$SCHEME_OUTPUT/DEAP/arousal/seed42/deap_arousal_fold02_scheme_a_001"
if [ -z "${GIT_SSH_COMMAND:-}" ]; then
    export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
fi
SCHEME_UPLOAD_ARGS=("$SCHEME_RESULT" --log "$SCHEME_RESULT/interface_tests.log" --git_timeout 600 --max_file_mib 16 --max_total_mib 64)
if [ -f "$SCHEME_RESULT/fit_check.log" ]; then SCHEME_UPLOAD_ARGS+=(--log "$SCHEME_RESULT/fit_check.log"); fi
bash "$SCHEME_TOOLS/scripts/upload_run_diagnostics.sh" "${SCHEME_UPLOAD_ARGS[@]}" --dry-run
SCHEME_UPLOAD_ID="$(date +%Y%m%d_%H%M%S)_$$"
SCHEME_UPLOAD_LOG="$SCHEME_OUTPUT/upload_logs/retry_${SCHEME_UPLOAD_ID}.log"
SCHEME_BUNDLE="$SCHEME_OUTPUT/upload_bundles/retry_${SCHEME_UPLOAD_ID}"
test ! -e "$SCHEME_BUNDLE"
bash "$SCHEME_TOOLS/scripts/phase3_run_logged.sh" "$SCHEME_UPLOAD_LOG" \
    bash "$SCHEME_TOOLS/scripts/upload_run_diagnostics.sh" "${SCHEME_UPLOAD_ARGS[@]}" --bundle_dir "$SCHEME_BUNDLE"
cat "$SCHEME_BUNDLE/upload_state.json"
)
```

## 7. 验证与Git边界

用户授权后只在feat/neurofdan-scheme-a实施可选模式。原模型、核心估计器、数据源文件保持逐字节一致；默认事实训练保留，不修改main、不创建PR或合并。所有新输出独立。

云端只源码/静态AST、Shell和Git检查，未运行模型、CLI、单元测试、真实数据或GPU。新增9项及全部旧测试均由上方服务器流程执行。不能将此前172项服务器测试通过写成本次新增测试通过，也不能保证验收或20轮性能。判别改善是实验问题，真实因果识别是另一个需要证据的问题。
