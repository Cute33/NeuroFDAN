# Epoch300分项证据与fire单因素对照

## 1. 已核验结果

快照SHA：9bab3f5854d31304469f96c4dac995e5414eb864；目录：runs/2026-11-12_190324_unknown_unknown_run_1180c4d7。6项载荷逐项长度/SHA256一致，skipped为空。missing中的run_summary.json/run_report.txt是无fold探针未生成的旧根摘要；probe_summary、config、日志及上传摘要都已上传。manifest自动包含两个日志的python_exit=0/tee_exit=0。服务器日志163项测试、14.886秒、OK。模型state、原文件哈希保持不变，无优化器和训练。

原训练源为e4c5f049e38a5d88cefd9055d8dc78acf4782d81，探针源3a448e85d15320190f447d5b873eb5f6ebb3f101，checkpoint为DEAP/arousal/fold2/epoch300。查询被试1/17/32分别16个库外窗口，类计数[6,10]/[5,11]/[6,10]；参考64个原库成员，标签[19,45]。是固定小规模局部诊断，不是整个训练总体的梯度。

## 2. 确认了什么，没有确认什么

训练模式且BN冻结的CNS中介分项（包含256神经元分支，不含单独分组的ANS电流映射）：

| 查询被试 | CE梯度L2 | 加权fire梯度L2 | fire/CE | fire与CE余弦 | 总梯度与CE余弦 |
|---|---|---|---|---|---|
| 1 | 0.000581062 | 0.062679075 | 107.87 | -0.001 | 0.009 |
| 17 | 0.000860709 | 0.055554415 | 64.55 | -0.020 | -0.004 |
| 32 | 0.000633868 | 0.034867520 | 55.01 | 0.009 | 0.028 |

eval模式fire/CE仍约61.69–68.65。方向大体近正交，而非一致强烈反向，不能简化成“fire始终抵消CE”。这些证据支持局部CNS原始梯度受fire主导，不证明fire是全部性能问题的唯一根因，也不等于Adam实际参数更新受其同比例主导。Adam预条件、动量及裁剪仍影响更新。

同一训练模式下，ANS分支fire/CE约0.287–0.308，CNS→ANS电流映射约0.310–0.345；OutcomeHead只有任务梯度，KL/fire不直接作用它。CNS的加权KLc/CE约0.434–0.687、KLa/CE约0.047–0.065，不支持把所有问题笼统归因KL，更不应立即删KL。

CNS均值跨输入std约0.0220–0.0260、sigma跨输入std约0.0080–0.0089；ANS对应约0.0160–0.0166和0.0082–0.0083。均值及方差都有非零输入变化，posterior sigma约1，不能称所有条件信息完全为零。但小样本类别均值距离非零不证明可泛化类别可分性或因果识别。

do-P1局部导数非零不表示有do标签监督；三个批次的平均do-P1近似相同，还受共同随机数、噪声及参考边缘化影响，不能单凭该均值下结论。训练事实CE仍是观察监督，主预测仍为probs_do。

## 3. 下一步只改变一个训练因素

推荐独立DEAP fold2对照：fire_weight=2.8→0.028，保留正的fire项及0.05目标率，其他训练配置与原300轮一致。默认CLI仍fire=2.8，不替换baseline或覆盖原结果。

在当前局部checkpoint上，由于fire项对权重线性，降100倍预计把其CNS梯度/CE比从55–108改为约0.55–1.08。这只是选择一个预先明确的验证量级，不是最优权重证明；新轨迹的实际比值会变。16样本单被试局部比值也不能直接代表400窗口混合被试batch，因此新增真实训练首batch的分项记录。

先跑100轮，与已有2.8实验的**前100轮**比较，而不是与其300轮best做不等预算比较。既有无scheduler、KL warmup为固定10轮；把总预算设100不会改变前100轮的时间权重定义。没有resume旧epoch300权重，因为那会同时改变初始化及历史轨迹，且已有续训守卫禁止静默换fire。

固定：lr2e-5，LIF256，hidden128，中介64，train/eval400，原始窗口分布，全局窗口shuffle，seed42，Adam/wd0，KL0.001/10轮，S/K=4/16和16/64，N_ref2048，ref_dim0，dropout0.3，4个内部驱动步，原BN冻结、原test-selected规则。没有新模块、新辅助分类头、标签权重、rec或中介真值。

判断以训练判别、两路径矩阵、BA/Macro-F1、CE及中介输入依赖为主，不仅看最高测试ACC。若正则比例下降且训练判别改善，支持这一优化瓶颈具有作用，但不证明因果机制正确；若比例下降但仍单类，则fire不是充分解释，应转向条件中介学习和OutcomeHead监督缺口，不自动继续减小lr或加到500轮。真实潜在中介不可观测与test-selected偏差仍保留。

## 4. 新增可选记录，不额外训练或采样

mainNeuroFDAN新增--loss_gradient_interval，默认0。设10时，在epoch1、10、20…100的**首个实际优化batch400**上复用原前向图，autograd.grad依次记录加权CE、KLc、KLa、fire、总loss的模块范数和与CE的夹角。随后只执行原有一次backward/裁剪/Adam step。没有第二次前向、额外随机采样或动态改权重。

ModelTrain.loss_component_gradient_report复用既有gradient_components；数值记录进入history及epoch_metrics中的optimization.loss_component_gradients。这里是首batch裁剪前梯度，不是epoch平均或Adam更新范数。新增测试比较开关前后的RNG、.grad及单步Adam后的全部参数/BN状态，确认记录不改变更新；这些测试由服务器执行，云端未运行。

## 5. 服务器一次运行并自动上传

继续当前dslenv。新worktree不切换主项目分支，旧结果不改。以下脚本先运行服务器测试，成功才运行100轮；测试或训练失败时也尝试上传已有日志和退出收据。正常只需回传SHA和链接。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
if [ -z "${GIT_SSH_COMMAND:-}" ]; then
    export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
fi
git status --short
git worktree list
timeout 600 git fetch origin feat/neurofdan-fire-gradient-control
FIRE_SHA="$(git rev-parse FETCH_HEAD)"
FIRE_TOOLS=/home/fss/NeuroFDAN-fire-control-tools
if [ ! -e "$FIRE_TOOLS" ]; then git worktree add --detach "$FIRE_TOOLS" "$FIRE_SHA"; fi
test "$(git -C "$FIRE_TOOLS" rev-parse HEAD)" = "$FIRE_SHA"
test -z "$(git -C "$FIRE_TOOLS" status --porcelain)"
git -C "$FIRE_TOOLS" log -1 --oneline
bash "$FIRE_TOOLS/scripts/run_fire_weight_control_and_upload.sh" \
    /home/fss /home/fss/NeuroFDAN/outputs/phase6/fire_weight_control \
    deap_arousal_fold02_fire0p028_100ep_001
)
```

完整主函数如下，已由上块自动调用，不重复执行：

```bash
cd /home/fss/NeuroFDAN
FIRE_TOOLS=/home/fss/NeuroFDAN-fire-control-tools
FIRE_OUTPUT=/home/fss/NeuroFDAN/outputs/phase6/fire_weight_control
FIRE_NAME=deap_arousal_fold02_fire0p028_100ep_001
FIRE_RESULT="$FIRE_OUTPUT/DEAP/arousal/seed42/$FIRE_NAME"
bash "$FIRE_TOOLS/scripts/phase3_run_logged.sh" "$FIRE_RESULT/train.log" \
    python -u "$FIRE_TOOLS/mainNeuroFDAN.py" \
    --dataset DEAP --label arousal --root_dir /home/fss \
    --output_dir "$FIRE_OUTPUT" --run_name "$FIRE_NAME" \
    --seed 42 --device cuda:0 --held_out_subjects 2 --epochs 100 \
    --lr 0.00002 --weight_decay 0 \
    --batch_size 400 --eval_batch_size 400 --workers 0 --sampling_mode window \
    --fd_mode full --no_balance --mediator_dim 64 --hidden_dim 128 --lif_neurons 256 \
    --dropout 0.3 --ref_dim 0 --time_steps 4 --logvar_min -6 --logvar_max 2 \
    --S_train 4 --K_train 16 --S_eval 16 --K_eval 64 --N_ref 2048 --reference_chunk_size 4 \
    --lambda_obs 1 --beta 0.001 --beta_warmup_epochs 10 --fire_weight 0.028 \
    --diagnostic_level full --log_detail_interval 10 --loss_gradient_interval 10 \
    --max_train_batches 0 --max_test_batches 0
```

输出根为outputs/phase6/fire_weight_control，run下保存train.log/.exit、interface_tests.log/.exit，fold02下保存原全套产物及分项梯度。上传自动调用本分支增强的upload_run_diagnostics.sh，继承3a448e8已在服务器使用成功的退出收据功能和6681381传输流程；旧上传工具不改。dry-run与正式上传都用600秒、单文件16MiB/总64MiB，唯一bundle/日志保存在同输出根upload_bundles/upload_logs，不需要手工检查manifest或抄轮数。

若仅网络上传失败，不重训；保留原上传日志/收据。以下是已确认未上传成功时的独立重传命令；push结果不确定时先回传候选SHA，由Codex核验是否已在远端，避免重复快照。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
FIRE_TOOLS=/home/fss/NeuroFDAN-fire-control-tools
FIRE_OUTPUT=/home/fss/NeuroFDAN/outputs/phase6/fire_weight_control
FIRE_RESULT="$FIRE_OUTPUT/DEAP/arousal/seed42/deap_arousal_fold02_fire0p028_100ep_001"
if [ -z "${GIT_SSH_COMMAND:-}" ]; then
    export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
fi
bash "$FIRE_TOOLS/scripts/upload_run_diagnostics.sh" "$FIRE_RESULT" \
    --log "$FIRE_RESULT/interface_tests.log" \
    --git_timeout 600 --max_file_mib 16 --max_total_mib 64 --dry-run
FIRE_UPLOAD_ID="$(date +%Y%m%d_%H%M%S)_$$"
FIRE_UPLOAD_LOG="$FIRE_OUTPUT/upload_logs/retry_${FIRE_UPLOAD_ID}.log"
FIRE_BUNDLE="$FIRE_OUTPUT/upload_bundles/retry_${FIRE_UPLOAD_ID}"
test ! -e "$FIRE_BUNDLE"
bash "$FIRE_TOOLS/scripts/phase3_run_logged.sh" "$FIRE_UPLOAD_LOG" \
    bash "$FIRE_TOOLS/scripts/upload_run_diagnostics.sh" "$FIRE_RESULT" \
    --log "$FIRE_RESULT/interface_tests.log" \
    --git_timeout 600 --max_file_mib 16 --max_total_mib 64 --bundle_dir "$FIRE_BUNDLE"
cat "$FIRE_BUNDLE/upload_state.json"
)
```

## 6. 交付边界

已执行原快照数值复核、源码审查、AST/Shell/Git静态检查。新增2项测试未在云端运行；163项通过是上轮服务器证据，不能描述成本次新增测试已通过。本轮不在云端训练、不启动GPU、不运行CLI或单元测试；新对照由用户服务器执行。不修改main、原稳定分支，不创建PR或合并。
