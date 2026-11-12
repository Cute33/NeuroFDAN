# fire对照负结果与同一期望目标的事实CE多采样对照

## 1. 先核验本轮，不把上传成功当成学习成功

固定快照f85389383113dd16d90e597c39e326fe168b158f，runs/2026-11-12_195926_DEAP_arousal_fold02_eb88fe5a。14项载荷长度及SHA256全部一致，skipped为空，两个日志的python_exit/tee_exit均0。epoch1–100连续，全部complete=true、smoke=false，best仍epoch1的ACC0.6/F1(类1)0.75。服务器165项测试、15.102秒、OK。缺少5项为通用模板中的可选文件，不需要用户手工补传。

参考来源文件与原fire2.8/300轮快照0cdd7392cd021022b2610397fc62a20702e91678逐字节相同。可读arguments差异为fire2.8→0.028、总预算300→100、分项记录开关及运行名。路径已脱敏，不从相同占位符推断绝对路径一致；实际命令明确复用/home/fss。比较原实验前100轮，原KL warmup固定10轮，无scheduler，不因总预算变100而改变前100轮系数。

第100轮：

| 指标 | 原fire2.8 | fire0.028 |
|---|---|---|
| 优化事实CE | 0.679823 | 0.679648 |
| 完整训练eval事实CE | 0.679238 | 0.679053 |
| Train前门CM | [[0,5270],[0,7130]] | 相同 |
| Test前门CM | [[0,160],[0,240]] | 相同 |
| Train事实CM | [[3,5267],[2,7128]] | 相同 |
| Test事实CM | [[0,160],[0,240]] | 相同 |
| 测试前门P1跨窗std | 0.003614 | 0.003711 |
| 完整训练CNS KL | 0.008339 | 0.054422 |

新实验前门Train/Test100轮全部预测1，ACC0.575/0.6，BA均0.5。事实训练末轮ACC0.575081、BA0.500144，仍接近多数类。提高CNS后验变化或KL不自动变成有用类别信息。降低fire没有解决判别退化，不支持立即延续300/500或完整LOSO，更不能把fire称为唯一根因。

## 2. 实际batch400的梯度比，而非局部16样本外推

以下为每个指定epoch首个优化batch的裁剪前梯度，不是epoch平均或Adam更新量：

| epoch | CNS fire/CE | CNS KLc/CE | CNS KLa/CE | 总梯度与CE余弦 |
|---|---|---|---|---|
| 1 | 0.369 | 0.047 | 0.119 | 0.903 |
| 10 | 1.166 | 3.380 | 2.514 | 0.700 |
| 20 | 1.586 | 1.293 | 1.144 | 0.482 |
| 50 | 8.748 | 1.603 | 0.797 | 0.105 |
| 100 | 2.988 | 3.785 | 0.253 | 0.218 |

证明“降100倍fire就会在新轨迹中始终与CE相当”的预测没有成立。梯度比依赖参数状态和采样，不能把旧checkpoint上的线性缩放当作整个训练的保证。ANS和CNS→ANS映射仍有任务梯度，不存在已证实的全链路断梯度。也不因某轮KLc/CE较高就自动删KL或继续扫描权重。

## 3. 推荐的下一步：先改善同一CE期望的估计器

实际路径FrontDoorEstimator.forward采样S_train=4组Mc/Ma，而原logits_obs仅使用Ma[:,0]；frontdoor_loss据此计算单次CE。其余样本参与KL及前门计算，但不进入事实CE。这个选择此前已记录，是有效的一样本MC估计，不是新发现的标签bug。

本次增加可选factual_ce_mode=all_samples，用已经存在的4组Ma和同一当前样本参考，复用原OutcomeHead。数学上，令

\[
\ell_i(m_a)=-\log p_\theta(y_i\mid m_a,g(x_i)),\quad
M_c^{(s)}\sim q_c(\cdot\mid x_{c,i}),\quad
M_a^{(s)}\sim q_a(\cdot\mid x_{a,i},M_c^{(s)}).
\]

原实现和新实现分别为

\[
\widehat L_{obs}^{(1)}=\frac1B\sum_i\ell_i(M_{a,i}^{(1)}),\qquad
\widehat L_{obs}^{(S)}=\frac1{BS}\sum_{i,s}\ell_i(M_{a,i}^{(s)}).
\]

在相同分布和独立重参数化噪声下，两者期望相同。条件于共享输入和图等随机量，独立样本的梯度均值可降低中介采样噪声；共享图、有限batch及优化轨迹使总体方差未必精确降为1/4，不预设真实数据上一定改善。

关键是**平均各样本CE**，不是对平均概率再取CE，也不平均logits。后两者通常是不同目标。目标仍为

\[
L=\lambda_{obs}\widehat L_{obs}+\beta_t(KL_c+KL_a)+0.028L_{fire}.
\]

没有新增损失、教师、辅助头、oracle标签、观测标签的do监督、类别平衡或新SCM边；不改变S_train=4、S_eval/K/N_ref或模型宽度。KL/fire仍按原定义归约，不随CE样本数额外乘4。

默认single保留原调用及随机流。all_samples批量执行4组Head，会改变Dropout及后续随机流，不能承诺与single逐位对应；同seed/相同初始结构只是可复现配置，不代表相同每步噪声。probs_obs及其诊断仍取第1组，评估及选模仍使用原probs_do路径，不事后改报较高ACC。

## 4. 识别边界与停止条件

此操作不解决潜在中介不可识别，也不消除事实监督的参考捷径。对固定编码器表示Z，按q(M|Z)独立重采样的增广分布满足Q(Y|M,Z)=P(Y|Z)；有能力的Head仍可能只用自身参考。平均4组CE仍估计同一目标，不会凭空强迫中介具有因果意义。

因此这是一个限定的数值优化对照，不包装成新的因果学习方法。先做同预算100轮，与fire0.028/single快照比较。看训练CE、训练BA/Macro-F1、矩阵、CNS任务梯度与正则比例、posterior参数以及概率变化，而非仅看best测试ACC。test-selected偏差继续披露。

若仍接近多数类，应停止在当前目标上继续lr/epoch/fire扫描；下一阶段需要明确可检验的中介信息或测量约束，并先合成验证，不能继续把增加轮数当作识别方案。即便改善，也要通过更多固定fold和消融后才冻结正式LOSO配置，不能凭一折证明因果机制。

## 5. 代码与兼容

- frontdoor_scm.py：可选[B,S,C]事实logits，frontdoor_loss按B×S对齐标签并均值CE，唯一前门积分不改。
- NeuroFDAN.py：转发采样模式，没有新参数层或路径。
- ModelTrain.py：仅训练阶段传模式，记录factual_ce_mode/factual_ce_samples；分项梯度记录真实CE样本数。切换模式禁止续训混用旧轨迹；必须新run从头。
- mainNeuroFDAN.py：--factual_ce_mode single|all_samples，默认single。
- phase6_gradient_audit.py：读取checkpoint原模式，避免后续复核时把MC训练误记为单样本分项。
- 原自动脚本增加可选第4参数；省略时仍single。新实验通过all_samples启动，测试/失败日志/退出收据/上传自动记录。
- 新7项测试：CE归约与Jensen区别、每组样本梯度、默认随机流兼容、S=1等价、前门两次概率平均、模式续训守卫、集成CPU前反向。云端未执行。

## 6. 服务器只需运行这一块

继续已激活dslenv。新worktree不会切换主项目分支，不覆盖原输出。脚本先跑服务器测试，通过才运行训练；出错也尝试上传日志。无需额外手工整理epoch或manifest。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
if [ -z "${GIT_SSH_COMMAND:-}" ]; then
    export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
fi
git status --short
git worktree list
timeout 600 git fetch origin feat/neurofdan-factual-mc-control
MC_SHA="$(git rev-parse FETCH_HEAD)"
MC_TOOLS=/home/fss/NeuroFDAN-factual-mc-tools
if [ ! -e "$MC_TOOLS" ]; then git worktree add --detach "$MC_TOOLS" "$MC_SHA"; fi
test "$(git -C "$MC_TOOLS" rev-parse HEAD)" = "$MC_SHA"
test -z "$(git -C "$MC_TOOLS" status --porcelain)"
git -C "$MC_TOOLS" log -1 --oneline
bash "$MC_TOOLS/scripts/run_fire_weight_control_and_upload.sh" \
    /home/fss /home/fss/NeuroFDAN/outputs/phase6/factual_mc_control \
    deap_arousal_fold02_ce4_fire0p028_100ep_001 all_samples
)
```

脚本内实际主函数命令如下，已由上块调用，不要重复执行同名run：

```bash
cd /home/fss/NeuroFDAN
MC_TOOLS=/home/fss/NeuroFDAN-factual-mc-tools
MC_OUTPUT=/home/fss/NeuroFDAN/outputs/phase6/factual_mc_control
MC_NAME=deap_arousal_fold02_ce4_fire0p028_100ep_001
MC_RESULT="$MC_OUTPUT/DEAP/arousal/seed42/$MC_NAME"
bash "$MC_TOOLS/scripts/phase3_run_logged.sh" "$MC_RESULT/train.log" \
    python -u "$MC_TOOLS/mainNeuroFDAN.py" \
    --dataset DEAP --label arousal --root_dir /home/fss \
    --output_dir "$MC_OUTPUT" --run_name "$MC_NAME" \
    --seed 42 --device cuda:0 --held_out_subjects 2 --epochs 100 \
    --lr 0.00002 --weight_decay 0 \
    --batch_size 400 --eval_batch_size 400 --workers 0 --sampling_mode window \
    --fd_mode full --no_balance --mediator_dim 64 --hidden_dim 128 --lif_neurons 256 \
    --dropout 0.3 --ref_dim 0 --time_steps 4 --logvar_min -6 --logvar_max 2 \
    --S_train 4 --K_train 16 --S_eval 16 --K_eval 64 --N_ref 2048 --reference_chunk_size 4 \
    --lambda_obs 1 --beta 0.001 --beta_warmup_epochs 10 --fire_weight 0.028 \
    --factual_ce_mode all_samples --diagnostic_level full \
    --log_detail_interval 10 --loss_gradient_interval 10 \
    --max_train_batches 0 --max_test_batches 0
```

结果在outputs/phase6/factual_mc_control/DEAP/arousal/seed42/deap_arousal_fold02_ce4_fire0p028_100ep_001；train.log/.exit和interface_tests.log/.exit自动收集；fold02的epoch_metrics.jsonl中optimization.factual_ce_samples及loss_component_gradients.factual_ce_samples应为4。测试及普通事实诊断仍为原首样本路径，不将优化MC损失与其数值直接混算。

## 7. 自动上传及网络失败后独立重传

正常上块已自动执行预览及正式上传，600秒超时、16/64MiB、唯一bundle和上传日志，沿用上轮服务器验证成功的增强上传器。回传最终SHA/链接即可。若只有上传失败，不重新训练；push结果未知先由Codex核验候选SHA。以下是明确未成功上传时的重传命令：

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
MC_TOOLS=/home/fss/NeuroFDAN-factual-mc-tools
MC_OUTPUT=/home/fss/NeuroFDAN/outputs/phase6/factual_mc_control
MC_RESULT="$MC_OUTPUT/DEAP/arousal/seed42/deap_arousal_fold02_ce4_fire0p028_100ep_001"
if [ -z "${GIT_SSH_COMMAND:-}" ]; then
    export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
fi
bash "$MC_TOOLS/scripts/upload_run_diagnostics.sh" "$MC_RESULT" \
    --log "$MC_RESULT/interface_tests.log" --git_timeout 600 \
    --max_file_mib 16 --max_total_mib 64 --dry-run
MC_UPLOAD_ID="$(date +%Y%m%d_%H%M%S)_$$"
MC_UPLOAD_LOG="$MC_OUTPUT/upload_logs/retry_${MC_UPLOAD_ID}.log"
MC_BUNDLE="$MC_OUTPUT/upload_bundles/retry_${MC_UPLOAD_ID}"
test ! -e "$MC_BUNDLE"
bash "$MC_TOOLS/scripts/phase3_run_logged.sh" "$MC_UPLOAD_LOG" \
    bash "$MC_TOOLS/scripts/upload_run_diagnostics.sh" "$MC_RESULT" \
    --log "$MC_RESULT/interface_tests.log" --git_timeout 600 \
    --max_file_mib 16 --max_total_mib 64 --bundle_dir "$MC_BUNDLE"
cat "$MC_BUNDLE/upload_state.json"
)
```

本轮云端仅解析已上传数值和源码、执行AST/Shell/Git静态检查；未运行模型、CLI、单元测试或GPU。165项通过是旧服务器证据，新7项等待上方服务器流程执行。所有改动在独立分支，不修改main、不创建PR或合并。
