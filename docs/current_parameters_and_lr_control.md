# 当前真实训练参数清单与一次学习率对照

## 1. 第2折实测及证据范围

提交40930096af11297b4a5dc3209363768b6b68b4da，runs/2026-11-12_143017_DEAP_arousal_fold02_07e3f0e6。Git fetch两次503，随后从同一固定提交的raw.githubusercontent.com读取manifest及13个载荷，逐项核验字节数/SHA256全部一致，无跳过项；不是仅根据终端上传退出码下结论。

20轮完整，原源码b7532d055bbf4dd74ff647e54a78a6dc9212907a，无已跟踪修改。Train12400/Test400，held-out2；训练及参考为1、3–32。训练和测试的事实/前门判决20轮均全为1。末轮矩阵分别[[0,5270],[0,7130]]及[[0,160],[0,240]]；ACC=0.575/0.6、BA均0.5，best为epoch1的0.6/0.75。

| epoch | 优化事实CE | 固定模型全训练eval CE | 测试前门平均P1 |
|---|---:|---:|---:|
| 1 | 0.684468 | 0.681699 | 0.570654 |
| 5 | 0.684236 | 0.682034 | 0.598672 |
| 10 | 0.683925 | 0.680462 | 0.569679 |
| 15 | 0.682021 | 0.681244 | 0.598857 |
| 20 | 0.681276 | 0.680412 | 0.591642 |

训练标签先验熵0.681854609；最终表现接近学习类别先验。概率未饱和到1、loss未明显发散、已保存梯度有限；Head裁剪前范数末轮均值0.118926/max0.227935。全局裁剪触发次数未记录，不能从单组范数断言全局从不裁剪。

学习率过大、过小或学习目标允许参考捷径等都可能造成弱输入判别。1e-4对Adam常见但不构成适配当前模型的证明；当前没有明确爆炸/饱和证据，也不能仅凭单类预测确定lr过大。值得执行一个较小lr的受控对照，不能直接宣称找到了根因。

## 2. 全部实际CLI配置与含义

这里的“全部参数”指运行超参数、数据/评价设置及实际路径中的固定控制量，不是逐元素列出所有已学习权重。权重是向量/矩阵，最终值需从服务器checkpoint读取；上传未包含权重，不能把初始化或RMS当最终逐元素值。

### 优化及损失

| 参数 | 第2折实际值 | 含义/需注意 |
|---|---|---|
| optimizer | Adam | 单参数组，不是AdamW |
| lr | 0.0001 | 无scheduler、无lr warmup，全程固定 |
| betas | (0.9,0.999) | 一阶/二阶动量，与KL beta无关 |
| eps | 1e-8 | Adam分母稳定项 |
| weight_decay | 0 | 实际关闭耦合L2；CLI默认0.01不代表本次用了0.01 |
| amsgrad | false | 不使用AMSGrad |
| 梯度裁剪max_norm | 1.0 | 全模型L2裁剪，backward后、step前 |
| epochs | 20 | 实际值；CLI默认100 |
| batch_size | 400 | 每次优化窗口数；CLI默认40 |
| eval_batch_size | 40 | 完整训练/测试推理分块，不更新参数 |
| 完整epoch步数 | 31 | 12400/400，20轮620次更新 |
| lambda_obs | 1.0 | 事实CE权重 |
| beta | 0.001 | 标准正态先验KL权重，不是lr或surrogate斜率 |
| beta_warmup_epochs | 10 | beta_t=0.001×min(epoch/10,1)；首轮0.0001，第10轮起0.001；不改变lr |
| fire_weight | 0.0 | 本次DEAP不以发放率项更新；HCI原约定1，不混作本次值 |
| 发放率目标 | 0.05 | 代码固定，出现在fire项中；本次权重0 |
| 事实CE样本轴 | Ma[:,0] | 首次层级采样；增加S_train不会自动变成S次CE平均 |
| KL归约 | D求和；CNS对B平均，ANS对S/B平均 | 维数增大可改变KL尺度 |
| class weight/BCE/focal | 均不使用 | 两logit普通CE，观察标签不监督probs_do |
| 锚定/旧rec/div | 不调用 | 不是把新候选0.1权重加进了本次真实训练 |

真实DEAP目标为CE_obs+beta_t(KLc+KLa)，fire项系数0。小KL损失标量不证明KL梯度不重要；不据此删除KL。

### 模型、概率分布与积分

| 参数 | 实际值 | 含义 |
|---|---|---|
| fd_mode | full | 主输出probs_do；none为事实路径对照，不等价于同一估计目标 |
| no_cns2ans | false | 保留采样Mc驱动ANS电流 |
| mediator_family | normal | 高斯重参数化，不使用合成tanh族 |
| mediator_dim | 64 | Mc及Ma各64维 |
| hidden_dim | 128 | 中介编码器、LIF神经元数及Head隐藏层维数；不是旧构造器未生效的32 |
| dropout | 0.3 | OutcomeHead训练期dropout，eval关闭；不在CNS posterior参数网络中 |
| num_classes | 2 | 两logit softmax；argmax判决 |
| ref_dim | 0 | 选定骨干输出完整拼接；DEAP13440维，不证明原输入充分 |
| S_train / K_train | 4 / 16 | 中介/经验参考MC次数，Head组合后softmax再均值 |
| S_eval / K_eval | 16 / 64 | 评估积分次数，影响MC误差与开销，不增加标签信息 |
| N_ref | 2048 | 训练总体均匀无放回抽库成员，积分采样有放回、排除自身 |
| reference_chunk_size | 4 | K轴计算分块；FrontDoorConfig类默认0，本次CLI配置4 |
| time_steps | 4 | 中介内部驱动步数，不是4个原始生理时间段 |
| logvar_min/max | -6 / 2 | sigma范围exp(-3)≈0.049787至exp(1)≈2.718282 |
| prior | N(0,I) | KL参照，不是已知真实中介分布 |

K_train分支不直接进入当前分类损失；增加K不能直接补足事实监督。改变采样数仍可能改变随机流和耗时，不把它当无影响的调参。sigma接近下界/上界需诊断，但本次CNS平均sigma约0.99448、ANS约0.99768，尚不能说触及方差边界。

### 实际骨干与SNN固定设置

| 控制量 | 数值 | 实际意义 |
|---|---|---|
| EEG/ANS输入 | [B,32,768]/[B,3,768] | DEAP2EMG+GSR |
| 卷积通道扩张 | 4倍 | CNS32→128，ANS3→12 |
| kernel / stride / padding | (5,3,3)/(2,2,2)/(2,1,1) | groups始终为原通道数，激活SiLU；当前对称padding非严格因果padding |
| 输出 | CNS[B,96,128]，ANS[B,96,12] | 展平12288/1152；参考相加13440 |
| BN运行统计 | 冻结 | affine可训练；PyTorch默认eps1e-5、momentum0.1，冻结时不更新running统计 |
| dag_adj初始化 | I+0.1×N(0,1)，32×32 | 学习参数，最终值不是此初始化 |
| Gumbel tau / hard | 1.0 / true | 训练随机硬选择，去自环；不保证为DAG |
| eval图 | seed42固定Gumbel后argmax | 与训练随机图不同，是已知模式差异，不擅自改图 |
| LIF tau / dt | 2.0 / 1.0 | 内部离散动力学参数，不是生理秒数 |
| alpha / 输入增益 | exp(-1/2)≈0.606531 / 1-alpha≈0.393469 | V_t=alpha V_(t-1)+(1-alpha)I_t |
| 初始电压 | 0 | 每次forward重置 |
| 阈值初始化 | 0.2 | 参数声明1.0再乘0.2，可学习向量；最终逐神经元值未上传 |
| 额外神经元bias初始化 | 0 | 可学习；Linear自身也有bias |
| spike/reset | V>theta；V←V-spike×theta | 硬发放、减法重置 |
| surrogate斜率beta | 5.0 | 反向近似5/(1+abs(5(V-theta)))²，与KL beta无关 |
| surrogate scale | 1 | 当前CurrentLIF固定ones，不是学习到的因果置信度 |
| CNS→ANS电流 | Linear(64,128)，含bias | 加到每个ANS时间步，不detach |
| posterior Head输入 | rates与平均current拼接，256维 | 零发放率不等于完全断开输入/梯度 |

hard=True时，同一Gumbel噪声下正tau不改变硬argmax选择；tau主要改变软梯度，不能简单认为降低tau便减少硬采样随机性。阈值虽声明可学习，spike自定义backward不直接给threshold导数，其他reset路径仍可能提供梯度；不能默认每个阈值都获得有效更新。

### 数据、来源、运行与评价

| 参数/设置 | 第2折实际值 |
|---|---|
| dataset / label | DEAP / arousal |
| held_out_subjects | [2]；覆盖start_subject=1/end_subject=None这两个默认范围参数 |
| Train/Test | 被试1及3–32共12400窗 / 被试2共400窗 |
| 窗口/fs/基线 | 6秒×128Hz=768点；每trial10窗；去384点基线 |
| EEG/外周索引 | 0:32 / 34、35、36 |
| 标签列/阈值 | arousal列1，严格>5为1，否则0；valence列0，本轮不用 |
| normalize_subject | 被试内部沿trial轴(axis=0)，不是跨被试混合或每窗时间轴归一化 |
| no_balance | true；全量自然总体，无类别下采样/加权采样 |
| sampling_mode / group_size | window / 1；全局窗口shuffle，非每批单被试；CLI默认grouped不代表本次实际 |
| drop_last | 训练true，但本次丢弃0窗；评估false |
| seed | 42，模型/数据初始化前；worker/generator同样管理种子 |
| cuDNN | deterministic=true、benchmark=false；不等于跨所有硬件逐位一致保证 |
| workers / cpu_threads | 0 / 0；后者保留PyTorch默认CPU线程数，不是0线程 |
| device | cuda:0（来自服务器配置，云端未验证GPU） |
| resume | None，从头初始化，不加载别的fold |
| max_train_batches/max_test_batches | 0/0，均完整执行，不是smoke |
| diagnostic_level/log_detail_interval | full/1；记录中介、梯度，每轮摘要；CLI默认basic/10 |
| root_dir | /home/fss（用户服务器路径；上传字段已脱敏） |
| output_dir/run_name | outputs/phase6/classification_coverage / deap_arousal_fold02_natural20_001 |
| 评估随机流 | 每次评估seed42、fork_rng恢复；不是新增训练种子 |
| validation | 无；保留train/test二分 |
| 主输出/正类 | probs_do / 1；事实输出诊断 |
| 选模 | test ACC>0.5且提高；相同ACC时正类F1严格提高；否则不保存best |

CLI的旧temperature=13.544和旧包装构造hidden_dim=32不控制当前main训练。旧GRU/原型rec/div路径也不能列作正在生效的参数。HCI原数据为24人、每trial5窗、ANS4通道，原fire约定1、正类0；它们不是这份DEAP第2折的实测配置。

## 3. 对参数的判断与下一步

优先做lr单因素对照：同fold2/seed42/20轮/batch400，lr从1e-4改为3e-5，约缩小3.33倍，其余不变。无需再加模型或辅助损失。使用新run从头初始化，不能resume旧优化器；加载optimizer状态可能恢复旧lr，混淆实验。

本次不同时改变dropout/beta/batch/方差边界，也不新增scheduler。batch40已有负结果；S/K主要数值积分；Gumbel、BN及LIF设置属于更深的模式/机制对照，需要单独设计。beta、Dropout可能影响信息瓶颈，但本次不据小KL标量直接删除或降低它们。

比较全20轮：事实CE是否持续脱离当前训练先验平台、训练两类Recall/BA是否改善、前门概率std/预测计数是否有效变化、梯度及posterior是否稳定；测试ACC/F1并列记录，不只挑best，不改变主输出。参考来源必须与原fold2相同。

若改善，说明当前固定预算下lr影响训练，不能说SCM被验证；若没有改善，只能说3e-5在相同620步预算未解决，不排除其他预算/参数，但不自动扩成大网格、追加fold或完整LOSO。此后需据结果重新定位目标/优化模式，不继续重复同一平台。

## 4. 服务器完整命令

继续dslenv，沿用已核验工具，无代码升级要求，无新测试。唯一研究参数变化是lr，输出位置和运行名独立。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
git status --short
LR_TOOLS=/home/fss/NeuroFDAN-natural-capacity-tools
test "$(git -C "$LR_TOOLS" rev-parse HEAD)" = b7532d055bbf4dd74ff647e54a78a6dc9212907a
test -z "$(git -C "$LR_TOOLS" status --porcelain)"
command -v python
python --version
LR_OUTPUT=/home/fss/NeuroFDAN/outputs/phase6/parameter_controls
LR_RUN=deap_arousal_fold02_lr3em5_20ep_001
LR_RESULT="$LR_OUTPUT/DEAP/arousal/seed42/$LR_RUN"
test ! -e "$LR_RESULT"
bash "$LR_TOOLS/scripts/phase3_run_logged.sh" "$LR_RESULT/train.log" \
  python -u "$LR_TOOLS/mainNeuroFDAN.py" \
  --dataset DEAP --label arousal --root_dir /home/fss \
  --output_dir "$LR_OUTPUT" --run_name "$LR_RUN" \
  --held_out_subjects 2 --seed 42 --device cuda:0 --epochs 20 \
  --batch_size 400 --eval_batch_size 40 --workers 0 --sampling_mode window \
  --lr 0.00003 --weight_decay 0 --fire_weight 0 --no_balance --fd_mode full \
  --mediator_dim 64 --hidden_dim 128 --dropout 0.3 --ref_dim 0 \
  --S_train 4 --K_train 16 --S_eval 16 --K_eval 64 --N_ref 2048 \
  --reference_chunk_size 4 --beta 0.001 --beta_warmup_epochs 10 --lambda_obs 1 \
  --time_steps 4 --logvar_min -6 --logvar_max 2 \
  --diagnostic_level full --log_detail_interval 1 \
  --max_train_batches 0 --max_test_batches 0
cat "$LR_RESULT/train.log.exit"
)
```

新fold02/fold.json应记录lr=3e-5、31步/epoch、620总更新、resume=None、无锚点。若NaN/Inf/来源异常立即停止保存日志，不改标签/阈值绕过。该fold已用于开发和参数对照，且test参与选模，不能称独立无偏泛化评价。

## 5. 上传预览及正式上传

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
UPLOAD_TOOLS=/home/fss/NeuroFDAN-upload-timeout-tools
LR_RESULT=/home/fss/NeuroFDAN/outputs/phase6/parameter_controls/DEAP/arousal/seed42/deap_arousal_fold02_lr3em5_20ep_001
test "$(git -C "$UPLOAD_TOOLS" rev-parse HEAD)" = 6681381298b74e9b69dc0df0594d5d9c7274acbc
test -z "$(git -C "$UPLOAD_TOOLS" status --porcelain)"
if [ -n "${GIT_SSH_COMMAND:-}" ]; then echo '已有自定义SSH命令，停止，不覆盖认证设置'; exit 1; fi
export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" "$LR_RESULT" \
  --log "$LR_RESULT/train.log" --git_timeout 600 --dry-run
LR_UPLOAD_ID="$(date +%Y%m%d_%H%M%S)_$$"
LR_UPLOAD_LOG="/home/fss/NeuroFDAN/outputs/phase6/upload_logs/lr_fold02_${LR_UPLOAD_ID}.log"
LR_BUNDLE="/home/fss/NeuroFDAN/outputs/phase6/upload_bundles/lr_fold02_${LR_UPLOAD_ID}"
mkdir -p outputs/phase6/upload_logs outputs/phase6/upload_bundles
test ! -e "$LR_BUNDLE"
bash "$UPLOAD_TOOLS/scripts/phase3_run_logged.sh" "$LR_UPLOAD_LOG" \
  bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" "$LR_RESULT" \
  --log "$LR_RESULT/train.log" --git_timeout 600 --bundle_dir "$LR_BUNDLE"
cat "$LR_UPLOAD_LOG.exit"
cat "$LR_BUNDLE/upload_state.json"
)
```

回传一份SHA/链接，失败也上传保留日志。旧fold2结果不重跑、不覆盖。

本轮只新增本文，核心运行文件不改；云端未运行模型、CLI、单测或训练。静态Shell/CLI及Git检查不等于服务器实验通过。不修改main、不创建PR或合并。
