# 300轮结果复核、自动回传证据与下一步只读检查

## 1. 研究主线与本轮决定

保留CNS→ANS层级潜在中介、OutcomeHead仅读Ma及参考、softmax后概率积分、主输出probs_do。目标是获得可解释且有效的生理情绪识别，不能以单类多数基线的ACC/F1作为有效分类或因果识别证据。本轮不修改模型、损失、数据、参考总体或选模规则，不启动新训练、不继续500轮、不启动完整LOSO。

下一步只复核现有DEAP fold2 epoch300的分项梯度，区分有无任务梯度、正则竞争和模式差异。不能从“有总梯度”推断有有效CE监督，也不能从损失标量大小直接推断梯度竞争。

## 2. 原始结果与完整性

三个固定快照：

| 类型 | SHA | runs目录 |
|---|---|---|
| DEAP smoke | 658abea0478f8189fa1d70a12ac2efcbfd5c4556 | 2026-11-12_182813_DEAP_arousal_fold02_07c501c2 |
| HCI smoke | 537046e6e078873921978c64ffcc3955de766797 | 2026-11-12_183308_HCI_arousal_fold01_ac7af976 |
| DEAP300 | 0cdd7392cd021022b2610397fc62a20702e91678 | 2026-11-12_184012_DEAP_arousal_fold02_9713b2df |

每份14项载荷，全部长度及SHA256一致，共42项。全部skipped为空。missing是通用白名单中的config.json、fold内run_summary/run_report/config/probe_summary等未生成文件；主要训练产物和run根摘要已上传，不能把这些可选缺少项解释成训练失败。服务器接口日志为157项测试、15.030秒、OK；这是读取服务器日志，未在云端重跑。

DEAP源版本e4c5f049e38a5d88cefd9055d8dc78acf4782d81，已跟踪工作区无修改。epoch1–300连续，每条完整且smoke=false；summary、history、epoch_metrics末轮一致。原参考成员2048个唯一三元组，训练来源1、3–32，没有held-out2。旧上传器未收集train.log.exit，因此不能只凭上传exit=0声称原训练进程exit=0；现有产物已足以确认300轮完整记录，不需要用户重传或手工解释。

最后一轮矩阵均按真实行/预测列[0,1]：

| 数据/路径 | CM | ACC | F1(类1) | Macro-F1 | BA |
|---|---|---|---|---|---|
| Train前门 | [[0,5270],[0,7130]] | 0.575000 | 0.730159 | 0.365079 | 0.500000 |
| Train事实 | [[328,4942],[270,6860]] | 0.579677 | 0.724699 | 0.418246 | 0.512185 |
| Test前门 | [[0,160],[0,240]] | 0.600000 | 0.750000 | 0.375000 | 0.500000 |
| Test事实 | [[9,151],[28,212]] | 0.552500 | 0.703151 | 0.397261 | 0.469792 |

前门训练/测试300轮全部预测1，best仍epoch1。测试P1跨样本std：前门0.006381、事实0.057547；训练为0.006263/0.048917。事实路径有变化，但判别仍弱，不能仅归因测试泛化。均值std变大不等于类别信息变强，也不能切换较高ACC路径作为主结果。

末轮优化CE=0.675098，总loss=0.676876，weighted KL=0.00007453，weighted fire=0.00170313；epoch末完整训练事实CE=0.674345，测试=0.692797。训练先验熵约0.681855：有有限拟合，远非已建立可靠判别。训练CNS/ANS平均发放率0.050559/0.049881，接近0.05，不能当成分类成功。

末轮测试CNS/ANS posterior mean跨样本std为0.022172/0.008571，posterior sigma均值0.999144/0.999010。低均值变化警告不再触发不代表信息问题解决。总梯度CNS/ANS/CNS→ANS/Outcome非零，不代表各自CE梯度足够，需下一步逐项分析。

## 3. 自动记录，不再要求人工补充停止轮数和跳过原因

新上传器保留6681381传输机制：独立临时Git库、600秒网络超时、脱敏白名单、唯一run目录、不强推。新增自动读取每个收集日志旁边的.log.exit：

- 只接收python_exit/tee_exit两个整数，拒绝链接、重复字段、超限或非白名单文本；不收集环境或认证信息。
- exit字段写入manifest.json和upload_summary.json的execution_evidence，并显示在upload_report.txt。
- 实际/计划epoch、smoke及产物完整状态由已有记录自动生成；missing/skipped及原因仍写入manifest。
- 没有退出收据时记录unknown，不能区分运行中、断电、kill或主动停止，更不能猜测人为意图。
- 不读取或上传模型权重。上传成功状态仍由upload_state.json和远端SHA核验独立记录。

旧6681381工具保留不动；本轮使用新独立worktree中的同名增强脚本。自动化只减少手工整理，不把“上传成功”“训练完成”“因果机制成立”混为一谈。

## 4. 新只读探针

旧phase6_gradient_audit.py硬编码fold1/epoch20和旧support_summary；不能直接用于本轮。新增audit_checkpoint_gradient_terms.py，复用原measure/labeled_windows及gradient_components，不更改原探针或真实训练。

输入为本地last.pt及其fold.json/reference_sources.json，严格验证epoch300/fold2/seed42/非smoke、模型配置、参考来源。GitHub快照不包含last.pt，路径为：

`/home/fss/NeuroFDAN/outputs/phase6/long_profile/DEAP/arousal/seed42/deap_arousal_fold02_lr2em5_lif256_fire2p8_300ep_001/fold02/last.pt`

固定首/中/末训练被试各随机取16个原库外窗口，原参考库无放回取64个，仅CPU；不按测试成绩筛选，不做训练样本平衡。训练模式保持原BN冻结，并与eval模式分别比较。复用原S_train4/K_train16，报告每项加权CE/KLc/KLa/fire、总目标和do-P1局部导数的模块范数、与CE梯度夹角、均值及sigma输入依赖、类别均值距离及输入统计。

使用autograd.grad，不调用optimizer.step，不创建优化器，不写参数.grad；前后比较模型state_dict(含BN)和原文件哈希。结果仅写新目录。它不是纯no_grad前向，因为计算局部梯度需要自动微分，但不进行训练。3个被试的局部小批次不能代表全训练总体；do-P1导数可能发生样本抵消，不是干预标签监督或识别证明。

判断：若正则梯度显著大于CE且方向冲突，下一轮才考虑保持其他设置的单因素权重对照；若没有该证据，不能先怪fire或KL，应继续审查监督与中介信息利用。任何情况都不在本轮静默删loss或改SCM。

## 5. 服务器：一条流程完成测试、探针和上传

继续已激活dslenv。下面不运行mainNeuroFDAN.py，不需要训练GPU，不改原项目分支或历史结果。脚本内先跑服务器测试，成功才运行探针；测试/探针失败也尝试上传已有日志和退出收据。上传自动执行dry-run再正式上传，唯一bundle及上传日志位于新输出根的upload_bundles/upload_logs。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
if [ -n "${GIT_SSH_COMMAND:-}" ]; then
    echo '保留调用方已有SSH命令'
else
    export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
fi
git status --short
git worktree list
timeout 600 git fetch origin feat/neurofdan-auto-evidence-and-gradient-review
REVIEW_SHA="$(git rev-parse FETCH_HEAD)"
REVIEW_TOOLS=/home/fss/NeuroFDAN-long-review-tools
if [ ! -e "$REVIEW_TOOLS" ]; then git worktree add --detach "$REVIEW_TOOLS" "$REVIEW_SHA"; fi
test "$(git -C "$REVIEW_TOOLS" rev-parse HEAD)" = "$REVIEW_SHA"
test -z "$(git -C "$REVIEW_TOOLS" status --porcelain)"
git -C "$REVIEW_TOOLS" log -1 --oneline
bash "$REVIEW_TOOLS/scripts/review_long_checkpoint_and_upload.sh" \
    DEAP arousal \
    /home/fss/NeuroFDAN/outputs/phase6/long_profile/DEAP/arousal/seed42/deap_arousal_fold02_lr2em5_lif256_fire2p8_300ep_001/fold02 \
    300 /home/fss/NeuroFDAN/outputs/phase6/long_profile_review \
    deap_arousal_fold02_epoch300_terms_001
)
```

结果目录：`/home/fss/NeuroFDAN/outputs/phase6/long_profile_review/DEAP/arousal/seed42/deap_arousal_fold02_epoch300_terms_001`；probe_summary.json为核心结果，train.log/.exit、interface_tests.log/.exit自动收集。正常只需回传最后SHA和GitHub链接，不再要求手抄epoch、manifest或退出码。

脚本实际执行的完整探针命令如下，已由上块调用，不重复运行：

```bash
cd /home/fss/NeuroFDAN
REVIEW_TOOLS=/home/fss/NeuroFDAN-long-review-tools
REVIEW_RESULT=/home/fss/NeuroFDAN/outputs/phase6/long_profile_review/DEAP/arousal/seed42/deap_arousal_fold02_epoch300_terms_001
bash "$REVIEW_TOOLS/scripts/phase3_run_logged.sh" "$REVIEW_RESULT/train.log" \
    python -u "$REVIEW_TOOLS/scripts/audit_checkpoint_gradient_terms.py" \
    --dataset DEAP --label arousal --root_dir /home/fss \
    --fold_dir /home/fss/NeuroFDAN/outputs/phase6/long_profile/DEAP/arousal/seed42/deap_arousal_fold02_lr2em5_lif256_fire2p8_300ep_001/fold02 \
    --checkpoint_epoch 300 \
    --output_dir /home/fss/NeuroFDAN/outputs/phase6/long_profile_review \
    --run_name deap_arousal_fold02_epoch300_terms_001 \
    --seed 42 --device cpu --held_out_subjects 2 --epochs 0 \
    --batch_size 16 --reference_samples 64 --workers 0 --diagnostic_level full
```

正常无需单独上传。如探针已完成、仅网络上传失败，使用下面的独立重传命令，不重跑探针、不覆盖旧bundle：

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
REVIEW_TOOLS=/home/fss/NeuroFDAN-long-review-tools
REVIEW_RESULT=/home/fss/NeuroFDAN/outputs/phase6/long_profile_review/DEAP/arousal/seed42/deap_arousal_fold02_epoch300_terms_001
if [ -z "${GIT_SSH_COMMAND:-}" ]; then
    export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
fi
# 若上次push结果不确定，先回传上传日志中候选SHA，由Codex核验，避免重复快照。
bash "$REVIEW_TOOLS/scripts/upload_run_diagnostics.sh" "$REVIEW_RESULT" \
    --log "$REVIEW_RESULT/interface_tests.log" \
    --git_timeout 600 --max_file_mib 16 --max_total_mib 64 --dry-run
RETRY_ID="$(date +%Y%m%d_%H%M%S)_$$"
RETRY_LOG="/home/fss/NeuroFDAN/outputs/phase6/long_profile_review/upload_logs/retry_${RETRY_ID}.log"
RETRY_BUNDLE="/home/fss/NeuroFDAN/outputs/phase6/long_profile_review/upload_bundles/retry_${RETRY_ID}"
test ! -e "$RETRY_BUNDLE"
bash "$REVIEW_TOOLS/scripts/phase3_run_logged.sh" "$RETRY_LOG" \
    bash "$REVIEW_TOOLS/scripts/upload_run_diagnostics.sh" "$REVIEW_RESULT" \
    --log "$REVIEW_RESULT/interface_tests.log" \
    --git_timeout 600 --max_file_mib 16 --max_total_mib 64 --bundle_dir "$RETRY_BUNDLE"
cat "$RETRY_BUNDLE/upload_state.json"
)
```

## 6. 验证边界

本轮云端仅读取上传产物、审阅代码，执行AST、bash -n和git diff检查。新增6项测试未在云端执行；不能将此前服务器157项测试通过等同于新增工具已经实测通过。服务器端流程先执行测试，若失败自动上传错误日志。本轮保留旧分支，不修改main，不创建PR，不合并。
