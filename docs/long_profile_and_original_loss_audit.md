# 原始损失追溯与低学习率长程联合配置

## 1. 最新lr对照结果

提交344b7e1d551af30a9d10eeef56db48a39317b351，runs/2026-11-12_150401_DEAP_arousal_fold02_b46b6c41。13个载荷长度/SHA256全部一致，无跳过，20轮完整，源码b7532d055bbf4dd74ff647e54a78a6dc9212907a。lr=3e-5最后事实CE=0.681344，原1e-4为0.681276；训练/测试仍全部预测1，ACC=0.575/0.6、BA=0.5，best epoch1。不能说降低lr已有效，也不能由20轮否定长程优化；本次只是相同620步预算的对照。

## 2. 与“最初”具体有什么区别

以用户指定审计基线19bf18f052b5c4b9b7395f99c0e8fff7f3ca4264为依据，不从注释里的旧ACC推断实验成功，也不把未提供版本视为相同代码。

旧loss.py的CausalMultiLoss使用BCEWithLogitsLoss(pos_weight=0.5)，并在最后加0.5×cns_dag_penalty+0.2×ans_dag_penalty。虽然文件也定义CE权重[0.6,1.5]，旧trainer实际选择的是bce分支。后门/前门分类权重配置均为0，也没有传对应预测标签，不能称执行了真实前门监督。

旧HCI的实际调用在ModelTrain.py:591–596：

\[
L_{old,HCI}=BCE_{pos=0.5}+0.5P_c+0.2P_a
+0.5L_{rec}-0.01L_{div}+1.0L_{fire}.
\]

旧NeuroFDAN.py最后一个同名类（:801–919）覆盖前一个类并返回7项，其中两个DAG penalty显式为0。所以在这个有效定义下，名义P_c/P_a没有实质惩罚。

- **rec**（旧:792）：MSE(memory重组特征, SNN rates)，重构的是64维发放率，不是原始EEG/ECG/GSR。
- **div**（旧:793–794）：H(mean_batch attention)，最小化−0.01div鼓励跨样本平均原型使用熵；它不是当前高斯posterior多样性。
- **fire**（旧:904–906）：各神经元在batch和时间上平均发放率与0.05的MSE。旧模型为一个64神经元LIF、96个编码时间步。

旧DEAP路径（旧ModelTrain.py:214）只解包3项，不添加rec/div/fire，但末尾有效模型返回7项；该提交存在接口不一致。因此只能报告其静态意图是BCE及DAG项，不能说该原样DEAP入口已成功训练或与用户其他旧版本结果相同。

当前真实目标（Phase1后版本）为：

\[
L_{current}=\lambda_{obs}CE(logits_{obs},Y)
+\beta_t(KL_c+KL_a)+w_{fire}L_{fire}^{hierarchical}.
\]

近期基线DEAP w_fire=0、HCI=1，beta=0.001/10轮warmup。新联合配置w_fire=2.8，其他损失形式不变。

当前fire为CNS与ANS两分支各自平均发放率MSE的0.5平均；CNS平均B，ANS平均B×S，各自先对4个内部驱动步求rate。因此与旧单分支96步的fire不是逐项相同数值，不能只比较系数就称约束强度相同。2.8是权重，不是目标率、Hz或surrogate斜率；目标仍0.05。

**这是实质性训练目标变更，不只是把损失换个名称。** 原因如下：

1. 用户最终规格要求删除可学习原型库、用真实配对经验参考样本执行前门积分；rec/div所约束的原型模块已不存在，不能机械保留或恢复旧原型注意力来追分。
2. 新中介是分层概率分布，需要明确KL正则；KL只约束分布与先验距离，不等价于重构，也不保证替代rec的输入保真作用。现有弱输入依赖说明这一缺口仍需研究。
3. 双类别输出使用CE；旧BCE的pos_weight=0.5还改变类别代价，与当前无类别权重CE不是只换输出形状。
4. 最初补充规格曾提出观察标签的do-path NLL，随后Phase1第5节明确禁止将真实观察标签直接用于干预分布监督；当前遵循后来的事实监督要求。合成oracle概率监督与真实训练严格分离。

loss.py历史定义仍保留，未被重写；当前入口不调用它。锚点候选已未通过预定合成标准，不将其损失加入此次真实长程配置。提高fire权重也不是“补回rec”。SCM、OutcomeHead无Mc直接输入、经验参考与两层概率积分保持不变。

## 3. 用户新偏好如何落地

| 项目 | 近期基线 | 新联合配置 |
|---|---|---|
| lr | 1e-4或3e-5 | 2e-5恒定；CLI可显式1e-5 |
| epochs | 20 | 总300；同配置可resume至总500 |
| DEAP训练/评估batch | 400/40 | 400/400 |
| HCI训练/评估batch | 100/40 | 100/100 |
| 每分支LIF神经元 | 128 | 256 |
| 编码器/Head hidden_dim | 128 | 仍128，不随LIF翻倍 |
| 中介维度 | 64 | 仍64 |
| fire权重 | DEAP0/HCI1 | 2.8 |
| 目标率、KL、S/K/N_ref、dropout等 | 既有值 | 保持不变 |

新真实CLI默认lr2e-5/epochs300/fire2.8/lif256，batch随DEAP/HCI取400/100、eval未指定时匹配train；命令仍显式给值。全局窗口shuffle，400/100不表示训练每batch恰为一个被试。评估batch不负责优化，原BN已冻结，因此相同batch不保证提高ACC；它会增加显存，也可能改变MC随机样本的分配和有限采样预测。

这是**多个因素共同变化的探索配置**，不是lr单因素实验。若变好，需要相同预算的消融才能归因，不能称“神经元256证明有效”或“fire2.8解决因果识别”。300轮DEAP每轮31步，共9300步，500轮15500步；低lr与多轮不是简单等效。更多测试选模机会本身可抬高best ACC，仍须披露test-selected偏差，看训练完整曲线与矩阵而非只看最高值。

HCI既有局部分项梯度审查中fire已明显强于CE；2.8可能进一步放大竞争，不能预设越大越好。新日志增加weighted_obs_loss/weighted_kl_loss/weighted_fire_loss，保持原总loss计算不变，便于比较实际分项。

## 4. 代码、接口与兼容

FrontDoorConfig新增可选lif_neurons；lif_width在未指定时沿用hidden_dim，以保留旧合成/配置。真实NeuroFDAN默认和CLI为256；原128配置可显式--lif_neurons128。GaussianMediator的encoder第一隐藏层仍128，第二层输出4×256，LIF256，rates/current拼接512后生成2×64参数；ANS电流Linear(64,256)。μ_c[B,64]、m_c[B,S,64]及μ_a/logvar_a/m_a[B,S,64]不变；Head仍128，读Ma及参考。

旧checkpoint缺失lif_neurons时解释为当时hidden_dim；匹配128结构可加载，不能把128权重填充为256。load_checkpoint比较规范化配置并严格加载权重。resume新增lr及分类/fire权重检查，已有batch、模式、参考来源等检查保留，防止optimizer载入后恢复旧lr。旧工具不认识新lif_neurons字段，新模型checkpoint应使用本分支工具读取。

300→500保持同run目录/模型/lr/fire/batch，epochs表示总轮数，不是再加500。既有恢复未保存所有随机流/采样器进度，不承诺与一次不间断500轮逐位一致。旧20轮/不同宽度/不同fire的checkpoint不可用于该配置续训，必须从头新run。

新增6项CPU测试覆盖默认参数/独立LIF宽度、前反向step、旧配置权重一致、checkpoint加载及resume防误用；已有大batch测试的默认断言同步改为eval=batch，显式eval40兼容用例保留。云端不执行测试或模型；先在服务器跑测试与两个1轮有限batch smoke，再训练DEAP300。smoke不算正式成绩。

## 5. 服务器：测试→两组smoke→DEAP300轮

继续dslenv，新工具worktree，不改主项目分支。以下脚本参数依次为阶段、数据集、任务、数据根目录、输出根目录、唯一运行名。内部完整主函数参数固定为上表，代码见scripts/run_long_profile.sh；不是伪造main参数。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
git status --short
git worktree list
if [ -n "${GIT_SSH_COMMAND:-}" ]; then echo '已有自定义SSH命令，停止，不覆盖认证设置'; exit 1; fi
export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
timeout 600 git fetch origin feat/neurofdan-long-low-lr-profile
LONG_SHA="$(git rev-parse FETCH_HEAD)"
LONG_TOOLS=/home/fss/NeuroFDAN-long-profile-tools
if [ ! -e "$LONG_TOOLS" ]; then git worktree add --detach "$LONG_TOOLS" "$LONG_SHA"; fi
test "$(git -C "$LONG_TOOLS" rev-parse HEAD)" = "$LONG_SHA"
test -z "$(git -C "$LONG_TOOLS" status --porcelain)"
git -C "$LONG_TOOLS" log -1 --oneline
command -v python
python --version
LONG_OUTPUT=/home/fss/NeuroFDAN/outputs/phase6/long_profile
LONG_TEST_LOG="$LONG_OUTPUT/interface_tests_001.log"
test ! -e "$LONG_TEST_LOG"
(
cd "$LONG_TOOLS"
bash scripts/phase3_run_logged.sh "$LONG_TEST_LOG" \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m unittest discover -s tests -v
)
bash "$LONG_TOOLS/scripts/run_long_profile.sh" smoke DEAP arousal /home/fss "$LONG_OUTPUT" deap_arousal_fold02_long_smoke_001
bash "$LONG_TOOLS/scripts/run_long_profile.sh" smoke HCI arousal /home/fss "$LONG_OUTPUT" hci_arousal_fold01_long_smoke_001
bash "$LONG_TOOLS/scripts/run_long_profile.sh" train DEAP arousal /home/fss "$LONG_OUTPUT" deap_arousal_fold02_lr2em5_lif256_fire2p8_300ep_001
)
```

上块仅在全部测试及两组smoke退出0后自动开始DEAP300，不启动HCI长程或完整LOSO。新smoke每组仅1个优化batch及最多1个测试batch，记录smoke=true，不与正式统计混用。新默认eval400/100可能增加显存；发生OOM停止并回传，不静默缩batch。必要时可另用新run显式reference_chunk_size=1分块，积分公式不变，但不在本命令中自动切换。

当前推荐只运行上块。后续HCI300的完整命令（先看DEAP结果再执行）：

```bash
cd /home/fss/NeuroFDAN
bash /home/fss/NeuroFDAN-long-profile-tools/scripts/run_long_profile.sh \
  train HCI arousal /home/fss /home/fss/NeuroFDAN/outputs/phase6/long_profile \
  hci_arousal_fold01_lr2em5_lif256_fire2p8_300ep_001
```

同DEAP配置延续总500轮（不是新独立实验，不更换run_name；暂不执行）：

```bash
cd /home/fss/NeuroFDAN
bash /home/fss/NeuroFDAN-long-profile-tools/scripts/run_long_profile.sh \
  resume500 DEAP arousal /home/fss /home/fss/NeuroFDAN/outputs/phase6/long_profile \
  deap_arousal_fold02_lr2em5_lif256_fire2p8_300ep_001
```

原始主函数等价300轮命令如下，**已由阶段脚本调用，不要再重复执行同一运行名**；若未来选1e-5，用新运行名显式改lr，不resume换lr：

```bash
cd /home/fss/NeuroFDAN
LONG_TOOLS=/home/fss/NeuroFDAN-long-profile-tools
LONG_RESULT=/home/fss/NeuroFDAN/outputs/phase6/long_profile/DEAP/arousal/seed42/deap_arousal_fold02_lr2em5_lif256_fire2p8_300ep_001
bash "$LONG_TOOLS/scripts/phase3_run_logged.sh" "$LONG_RESULT/train.log" \
  python -u "$LONG_TOOLS/mainNeuroFDAN.py" \
  --dataset DEAP --label arousal --root_dir /home/fss \
  --output_dir /home/fss/NeuroFDAN/outputs/phase6/long_profile \
  --run_name deap_arousal_fold02_lr2em5_lif256_fire2p8_300ep_001 \
  --seed 42 --device cuda:0 --held_out_subjects 2 --epochs 300 --lr 0.00002 \
  --batch_size 400 --eval_batch_size 400 --workers 0 --sampling_mode window \
  --weight_decay 0 --fire_weight 2.8 --fd_mode full --no_balance \
  --mediator_dim 64 --hidden_dim 128 --lif_neurons 256 --dropout 0.3 --ref_dim 0 \
  --time_steps 4 --logvar_min -6 --logvar_max 2 \
  --S_train 4 --K_train 16 --S_eval 16 --K_eval 64 --N_ref 2048 --reference_chunk_size 4 \
  --lambda_obs 1 --beta 0.001 --beta_warmup_epochs 10 \
  --diagnostic_level full --log_detail_interval 10 --max_train_batches 0 --max_test_batches 0
```

## 6. 上传预览、正式上传及检查

下块覆盖本轮两组smoke/DEAP长程，亦支持后续实际生成的HCI300；不存在的目录跳过。每次唯一日志/bundle，旧6681381上传器不改。更长逐轮文件可能超过16MiB，检查manifest是否跳过；本地完整文件保留，不能把跳过描述成已完整上传。PNG/权重和原数据仍不上传。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
UPLOAD_TOOLS=/home/fss/NeuroFDAN-upload-timeout-tools
LONG_OUTPUT=/home/fss/NeuroFDAN/outputs/phase6/long_profile
test "$(git -C "$UPLOAD_TOOLS" rev-parse HEAD)" = 6681381298b74e9b69dc0df0594d5d9c7274acbc
test -z "$(git -C "$UPLOAD_TOOLS" status --porcelain)"
if [ -n "${GIT_SSH_COMMAND:-}" ]; then echo '已有自定义SSH命令，停止，不覆盖认证设置'; exit 1; fi
export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
LONG_UPLOAD_ID="$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p outputs/phase6/upload_logs outputs/phase6/upload_bundles
for LONG_REL in \
  DEAP/arousal/seed42/deap_arousal_fold02_long_smoke_001 \
  HCI/arousal/seed42/hci_arousal_fold01_long_smoke_001 \
  DEAP/arousal/seed42/deap_arousal_fold02_lr2em5_lif256_fire2p8_300ep_001 \
  HCI/arousal/seed42/hci_arousal_fold01_lr2em5_lif256_fire2p8_300ep_001
do
  LONG_RESULT="$LONG_OUTPUT/$LONG_REL"
  if [ ! -d "$LONG_RESULT" ]; then echo "尚未运行：$LONG_RESULT"; continue; fi
  LONG_NAME="$(basename "$LONG_RESULT")"
  bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" "$LONG_RESULT" \
    --log "$LONG_RESULT/train.log" --log "$LONG_OUTPUT/interface_tests_001.log" \
    --git_timeout 600 --max_file_mib 16 --max_total_mib 64 --dry-run
  LONG_UPLOAD_LOG="/home/fss/NeuroFDAN/outputs/phase6/upload_logs/${LONG_NAME}_${LONG_UPLOAD_ID}.log"
  LONG_BUNDLE="/home/fss/NeuroFDAN/outputs/phase6/upload_bundles/${LONG_NAME}_${LONG_UPLOAD_ID}"
  test ! -e "$LONG_BUNDLE"
  bash "$UPLOAD_TOOLS/scripts/phase3_run_logged.sh" "$LONG_UPLOAD_LOG" \
    bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" "$LONG_RESULT" \
    --log "$LONG_RESULT/train.log" --log "$LONG_OUTPUT/interface_tests_001.log" \
    --git_timeout 600 --max_file_mib 16 --max_total_mib 64 --bundle_dir "$LONG_BUNDLE"
  cat "$LONG_UPLOAD_LOG.exit"
  cat "$LONG_BUNDLE/upload_state.json"
done
)
```

当前正常回传三份SHA/链接；HCI长程尚未运行时应跳过。若测试先失败、尚无运行目录，回传interface_tests_001.log及.exit错误。日志在各run根目录train.log，模型和诊断在fold02（DEAP）或fold01（HCI）；smoke与train目录分离。

检查epoch50/100/200/300的完整训练判别、两路径矩阵、CE相对先验、三项加权损失、发放率、posterior与梯度。不存在“必须达到某测试ACC才算因果正确”的条件。若NaN/Inf/OOM/来源守卫失败立即停止；若长时间训练仍无判别，可主动中止保留快照，不要求为补轮数继续。中止结果明确标注实际轮数，不当成完成300或完整LOSO。

## 7. 验证状态

已执行仅历史源码/上传载荷审阅、AST、Shell、参数与Git静态检查。新增6项及既有轻量测试均未在云端运行，真实smoke与长程待服务器执行。本分支保留旧分支，不修改main，不创建PR或合并；无正式分类改善、GPU或服务器兼容性实测结论。
