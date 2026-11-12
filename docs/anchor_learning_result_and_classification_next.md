# 锚定学习负结果与真实分类基线第2折复核

## 已核验结果

服务器提交4e091255507b3b95fbb9f23ee36ebfc7c0cf91e2，目录runs/2026-11-12_140219_unknown_unknown_run_7e16867b。7个载荷长度/SHA256全部一致，无跳过；通用run_summary/run_report缺少不影响该独立实验。4项接入测试通过。completed=true、smoke=false、training_executed=true、real_data_training=false，源码34d5d5b117573a5679d539284e5687377d3ee4fc，无已跟踪修改。

2机制×4条件×20轮严格完整，共160条历史；各机制内X/Y及初始化指纹一致。固定seed42、normal、gamma1、训练2048/测试512、16个干预点、MC重复3次。不存在按真实测试概率选epoch或alpha。

| 条件 | aligned干预概率MSE | misaligned干预概率MSE |
|---|---:|---:|
| 真实中介监督参考 | 0.00107717 | 0.00474016 |
| 无锚定 | 0.01441209 | 0.00552376 |
| 尺度稳定锚定 | 0.01465772 | 0.00588458 |
| 原锚定 | 0.01110318 | 0.00688008 |
| 训练比例常数 | 0.00017683 | 0.00179723 |

尺度稳定候选没有通过预设改善要求。misaligned的MAE由0.062553降至0.056694，但MSE、NLL变差，不能只挑MAE称成功；aligned的MAE由0.082643增至0.088824。只有单个训练种子和3次MC，差异不能称统计显著或彻底否定所有锚点方法；它足以不把该配置晋级真实训练。

aligned中CNS/ANS统一测试探针MSE从0.989514/0.768245降至0.585560/0.571982，说明统计信息确实更可解码，而非约束没有执行。CNS均值跨输入方差均值也从0.058069增至0.134117。但这没有转成更准确的干预估计，不能把更分散的中介等同于更正确的中介。

独立oracle前门与直接真值MSE分别1.504e-5/1.454e-5，工程容差通过。aligned直接概率点间std仅0.001822，接近MC分辨率，而候选估计std达0.117216，说明该组不适合被宣传为明显非零效应的恢复成功；常数在这些点有很小误差是合理参照。misaligned真值std为0.042394，候选为0.082492。参考监督组也未超过常数，有限样本条件学习/分布错设仍可能影响结果，不能把所有失败都归因于无中介识别。

## 当前决定与主线

关闭A候选晋级：不接入真实训练，不再扫alpha、加锚点、延长合成训练或制作更多成功展示图。保留所有正负结果，作为候选方法的失败证据，不宣称原始生理系统中已识别中介。

最终目标仍是原SCM下CNS→ANS前门情绪识别的LOSO ACC/F1。当前真实fold1整体训练近先验，直接扩大完整LOSO不能视为问题已解决。本次将下一步切换为**原模型的真实分类覆盖检查**：只执行DEAP/arousal held-out2、20轮，判断fold1退化是否也出现在另一个fold。不是锚点候选验收或新方法实验，不承诺ACC提高，也不是用更多fold掩盖失败。

固定使用用户认可的自然训练总体、全局窗口shuffle、训练batch400/评估40、Adam lr1e-4/wd0、DEAP fire0、原KL/warmup/S/K/N_ref。无锚点、oracle标签或新辅助项；主预测probs_do，事实仅对照，不择高。保留test-selected规则及其偏差，无validation。

当前所有真实运行核心文件与受保护b7532d055bbf4dd74ff647e54a78a6dc9212907a一致，直接使用服务器现有natural-capacity工具，不需要新模型代码或更换主工作区。本轮仅新增本文。

## 服务器运行

沿用dslenv，先检查版本和全新目录。读取原数据、仅服务器GPU真实训练；云端不运行。本次是一个新fold从头初始化，不续训旧fold1，不补原100轮。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
git status --short
git worktree list
CLASS_TOOLS=/home/fss/NeuroFDAN-natural-capacity-tools
test "$(git -C "$CLASS_TOOLS" rev-parse HEAD)" = b7532d055bbf4dd74ff647e54a78a6dc9212907a
test -z "$(git -C "$CLASS_TOOLS" status --porcelain)"
command -v python
python --version
CLASS_OUTPUT=/home/fss/NeuroFDAN/outputs/phase6/classification_coverage
CLASS_RUN=deap_arousal_fold02_natural20_001
CLASS_RESULT="$CLASS_OUTPUT/DEAP/arousal/seed42/$CLASS_RUN"
test ! -e "$CLASS_RESULT"
bash "$CLASS_TOOLS/scripts/phase3_run_logged.sh" "$CLASS_RESULT/train.log" \
  python -u "$CLASS_TOOLS/mainNeuroFDAN.py" \
  --dataset DEAP --label arousal --root_dir /home/fss \
  --output_dir "$CLASS_OUTPUT" --run_name "$CLASS_RUN" \
  --held_out_subjects 2 --seed 42 --device cuda:0 --epochs 20 \
  --batch_size 400 --eval_batch_size 40 --workers 0 --sampling_mode window \
  --lr 0.0001 --weight_decay 0 --fire_weight 0 --no_balance --fd_mode full \
  --mediator_dim 64 --hidden_dim 128 --dropout 0.3 --ref_dim 0 \
  --S_train 4 --K_train 16 --S_eval 16 --K_eval 64 --N_ref 2048 \
  --reference_chunk_size 4 --beta 0.001 --beta_warmup_epochs 10 --lambda_obs 1 \
  --time_steps 4 --logvar_min -6 --logvar_max 2 \
  --diagnostic_level full --log_detail_interval 1 \
  --max_train_batches 0 --max_test_batches 0
cat "$CLASS_RESULT/train.log.exit"
)
```

结果在classification_coverage/DEAP/arousal/seed42/deap_arousal_fold02_natural20_001/fold02，终端日志位于其run根目录train.log。来源必须held-out2，训练/参考为1及3–32，12400训练窗、400测试窗；参考库每fold重建，不能复用fold1来源。

检查完整20轮矩阵、真实/预测计数、两路径ACC/F1/BA、事实CE相对当前训练类别先验熵及posterior/梯度，不只看best。选择规则不变，无合格best仅last.pt。新fold类别比例以实际记录为准，不假设与fold1相同。

停止条件：数据来源/NaN/Inf等实现异常立即停止并保留日志。若完整训练后仍为多数类平台，则不再追加fold3或完整LOSO，也不切换到更高事实ACC；下一步必须是明确的训练方案变更。若训练具有判别而测试退化，归入泛化问题；若两者均有有效判别，可再固定少量开发fold验证稳定性，冻结配置后再主LOSO。开发中反复查看测试及test-selected选模的偏差必须披露，不称独立无偏泛化评价。

## 上传预览与正式上传

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
UPLOAD_TOOLS=/home/fss/NeuroFDAN-upload-timeout-tools
CLASS_RESULT=/home/fss/NeuroFDAN/outputs/phase6/classification_coverage/DEAP/arousal/seed42/deap_arousal_fold02_natural20_001
test "$(git -C "$UPLOAD_TOOLS" rev-parse HEAD)" = 6681381298b74e9b69dc0df0594d5d9c7274acbc
test -z "$(git -C "$UPLOAD_TOOLS" status --porcelain)"
if [ -n "${GIT_SSH_COMMAND:-}" ]; then echo '已有自定义SSH命令，停止，不覆盖认证设置'; exit 1; fi
export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" "$CLASS_RESULT" \
  --log "$CLASS_RESULT/train.log" --git_timeout 600 --dry-run
CLASS_UPLOAD_ID="$(date +%Y%m%d_%H%M%S)_$$"
CLASS_UPLOAD_LOG="/home/fss/NeuroFDAN/outputs/phase6/upload_logs/classification_fold02_${CLASS_UPLOAD_ID}.log"
CLASS_BUNDLE="/home/fss/NeuroFDAN/outputs/phase6/upload_bundles/classification_fold02_${CLASS_UPLOAD_ID}"
mkdir -p outputs/phase6/upload_logs outputs/phase6/upload_bundles
test ! -e "$CLASS_BUNDLE"
bash "$UPLOAD_TOOLS/scripts/phase3_run_logged.sh" "$CLASS_UPLOAD_LOG" \
  bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" "$CLASS_RESULT" \
  --log "$CLASS_RESULT/train.log" --git_timeout 600 --bundle_dir "$CLASS_BUNDLE"
cat "$CLASS_UPLOAD_LOG.exit"
cat "$CLASS_BUNDLE/upload_state.json"
)
```

回传一份SHA及链接，失败也保留上传，不覆盖旧目录。不是重新上传本次合成结果，不上传原始数据/权重。

本轮云端只做载荷/历史/数值与源码静态审阅，以及命令Shell/参数/Git检查；没有运行模型、CLI、测试、真实或GPU训练。仅新增本文，代码分支34d5d5b受保护；不修改main，不创建PR或合并。
