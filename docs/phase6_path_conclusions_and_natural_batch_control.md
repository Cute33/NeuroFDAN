# Phase6：末端路径检查结论与一次完整训练总体优化对照

## 1. 本次证据及边界

已读取服务器提交 `ea4c7d51f0c0e5c4fcf162cc5ee7a17c5e8f1564` 的 `runs/2026-11-11_231353_unknown_unknown_run_5e9c0739/`。5个载荷逐个核验字节数及SHA256，无跳过项。通用清单缺少的run_summary/run_report/config不属于该探针必需产物。4项新增路径测试通过，来自服务器日志，非云端执行。

`completed=True`；两个私有模型的旧端点指标均重现，记忆拟合门槛均通过；原checkpoint未改变，末端诊断前后模型状态和已有梯度未改变。私有拟合确实更新了临时模型，不应把整个操作描述成纯推理。临时权重未保存。来源为训练被试2的16窗，计数[4,12]，没有held-out样本；这16窗同时用于拟合与评分。参考成员2048个，来自被试2–32，排除查询自身。

下表为每个私有拟合端点的4次MC重复均值，S=16、K=64。自身参考与经验参考使用相同的Ma采样，均先softmax再平均概率。

| 私有拟合起点 | 自身参考ACC | 经验参考前门ACC | 自身参考P1跨窗std | 前门P1跨窗std | 自身参考观察标签NLL | 前门观察标签NLL |
|---|---:|---:|---:|---:|---:|---:|
| seed42初始化 | 1.000000 | 0.984375 | 0.425111 | 0.196407 | 0.011945 | 0.249246 |
| 自然训练epoch20权重 | 1.000000 | 1.000000 | 0.426494 | 0.239496 | 0.010061 | 0.201623 |

初始化起点的前门4次ACC分别为1、1、0.9375、1，其他上述ACC每次均为1。矩阵按[0,1]排列；前门仅一次出现[[3,1],[0,12]]，其余为[[4,0],[0,12]]。这些是已记忆16窗的观察标签评分，不是独立泛化成绩，也不是干预概率真值误差。

功能置换后的平均类别概率L1变化：

| 固定/改变的量 | 初始化起点 | epoch20起点 |
|---|---:|---:|
| 固定自身参考，轮换Ma | 0.165950 | 0.239119 |
| 固定Ma，轮换自身参考 | 0.650070 | 0.578283 |
| 固定经验参考，轮换Ma | 0.510155 | 0.600498 |
| 固定Xa、ANS标准噪声与参考，轮换Mc | 0.483435 | 0.564431 |

最后一行ANS条件均值变化RMS分别为0.958885、0.934537。该端点中，参考输入作用很强，但Ma和Mc→Ma仍有明显数值作用；经验参考积分没有必然消除类别判别。不能把这些数值换算成因果贡献比例，置换组合可能超出拟合支持域，也不是物理干预或条件独立性检验。

## 2. 目前排除什么，尚未解决什么

- 不支持普遍的硬断梯度、参数不可恢复、模型完全只能靠自身参考、前门积分必然恒定等解释。
- 不能据此排除完整训练时的posterior输入依赖过弱、采样噪声、参考捷径、优化困难和被试异质性。末端私有拟合与完整训练不是同一个优化轨迹。
- 自然训练总体基线DEAP仍是12400窗上的多数类平台：事实及前门训练ACC=0.575、Balanced Accuracy=0.5；测试400窗全部预测1，ACC=0.6、Balanced Accuracy=0.5。epoch20训练事实CE约0.681549，训练标签先验熵约0.681855。测试ACC=0.6不是有效判别证据。
- 上述基线来源提交 `f325c6b61f749184313aa198a5cf846525f8363c`；原始目录 `outputs/phase6/population_alignment/DEAP/arousal/seed42/deap_arousal_fold01_nobalance20_001/`，不得覆盖。
- 潜在真实中介在真实数据中不可观测，基本非识别缺口不会因为小批拟合或局部梯度非零而消失；不把这些结果写成真实生理机制被识别。

## 3. 下一步只做一个受控实验

暂停增加探针、合成样本量和HCI额外对照。保留用户认可的全局窗口shuffle与大batch基线；本次临时检验DEAP自然训练总体的batch40，而非自动替换batch400正式设置。其他参数沿用自然总体batch400基线，不加损失、不改SCM、骨干、中介、BN或选模规则。

只运行DEAP/arousal/fold1，从seed42重新初始化，20轮完整训练与完整评估。12400窗不下采样、不平衡、不加类别权重，每轮310步、共6200步，无尾批丢弃；原batch400每轮31步、共620步。评估batch仍为40。相同epoch不等于相同更新预算；图的训练态Gumbel重采样、中介噪声和Dropout的调用频率也随之改变。因此该配置对照检验优化安排，不单独识别batch噪声的因果效应，不声称batch400已被证明错误。

沿用事实CE+KL（DEAP原fire权重0），观察标签不监督probs_do。主要预测和选模仍使用probs_do；probs_obs仅明确标注作诊断。不增加validation；仍用held-out ACC>0.5且上升，ACC相等时F1严格上升的原规则。存在测试集选模偏差。

### 事先规定的判断

看完整1–20轮，重点并列第1/5/10/15/20轮，不只挑最佳测试ACC：

1. 全训练集事实CE是否持续低于训练标签先验熵，事实训练矩阵是否持续区分两类；同时报告Balanced Accuracy与Macro-F1，不能只看多数类ACC。
2. 全训练集前门输出是否持续产生有效输入差异，且不是单纯随机采样导致少量翻转。结合概率std、熵与矩阵，不将任何固定std阈值作为因果识别判据。
3. CNS/ANS的mu、sigma输入变化和KL如何演变，任务相关模块梯度是否有限且有效。仅靠总梯度或KL不下结论。
4. 同时报告测试事实/前门路径；不因为事实ACC更高就切换主输出。

若仍停留在先验平台，停止继续缩小batch或盲目延长100轮，返回潜在中介任务监督和生理约束的明确假设问题。若训练学会而测试仍单类，则将问题转向跨被试泛化；若事实可学、前门仍退化，则回到条件结果与参考积分的学习差异。本次结果无论正负都应回传。

正式LOSO尚未放行。进入完整实验前至少要冻结配置及评价口径，在有限的固定fold试运行中确认完整数据训练确有两类判别、概率及MC稳定性可用、日志与来源检查完整；研究中的因果结论还需单独满足合成无中介监督验证及明确的识别假设。不能承诺再跑固定几小时即可解决，也不以某个测试ACC门槛代替上述证据。

## 4. 服务器运行：只调用已有训练入口

本次没有新增或修改可运行模型代码。保护版本 `b7532d055bbf4dd74ff647e54a78a6dc9212907a` 的模型、数据、训练、损失及诊断运行文件，与本次代码起点019d837对应文件静态diff相同。继续使用已存在的natural-capacity工具worktree，避免切换项目主工作区或覆盖旧工具。环境继续使用已运行成功的dslenv；不重新安装依赖或盲目执行conda激活命令。

以下块只创建新结果目录，不读旧checkpoint续训。若目录已存在则停止，另取唯一运行名，不删除已有产物。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
git status --short
git worktree list
OPT_TOOLS=/home/fss/NeuroFDAN-natural-capacity-tools
test "$(git -C "$OPT_TOOLS" rev-parse HEAD)" = b7532d055bbf4dd74ff647e54a78a6dc9212907a
test -z "$(git -C "$OPT_TOOLS" status --porcelain)"
command -v python
python --version
OPT_RUN=deap_arousal_fold01_natural_b40_20ep_001
OPT_OUTPUT=/home/fss/NeuroFDAN/outputs/phase6/optimization_control
OPT_RESULT="$OPT_OUTPUT/DEAP/arousal/seed42/$OPT_RUN"
test ! -e "$OPT_RESULT"
mkdir -p "$OPT_RESULT"
bash "$OPT_TOOLS/scripts/phase3_run_logged.sh" "$OPT_RESULT/train.log" \
  python -u "$OPT_TOOLS/mainNeuroFDAN.py" \
  --dataset DEAP --label arousal --root_dir /home/fss \
  --output_dir "$OPT_OUTPUT" --run_name "$OPT_RUN" \
  --seed 42 --device cuda:0 --held_out_subjects 1 --epochs 20 \
  --batch_size 40 --eval_batch_size 40 --workers 0 --sampling_mode window \
  --lr 0.0001 --weight_decay 0 --fire_weight 0 --no_balance \
  --fd_mode full --mediator_dim 64 --hidden_dim 128 --dropout 0.3 --ref_dim 0 \
  --S_train 4 --K_train 16 --S_eval 16 --K_eval 64 --N_ref 2048 \
  --reference_chunk_size 4 --beta 0.001 --beta_warmup_epochs 10 --lambda_obs 1 \
  --time_steps 4 --logvar_min -6 --logvar_max 2 \
  --diagnostic_level full --log_detail_interval 1 \
  --max_train_batches 0 --max_test_batches 0
cat "$OPT_RESULT/train.log.exit"
)
```

`fold01/fold.json`应记录optimization_windows=12400、train_drop_last_windows=0、full_epoch_optimizer_steps=310、planned_fresh_full_run_optimizer_steps=6200、smoke=false、sampling_mode=window。参考来源仍应来自2–32；与基线的reference_sources.json比较来源而不是只比较库大小。CUDA不可用不是本次代码验证结果；若改为CPU执行需要如实记录设备差异和耗时，不在云端代跑。

结果准确目录为 `/home/fss/NeuroFDAN/outputs/phase6/optimization_control/DEAP/arousal/seed42/deap_arousal_fold01_natural_b40_20ep_001/`；逐轮记录在其fold01/epoch_metrics.jsonl、mediator_diagnostics.jsonl及collapse_diagnostics.jsonl，终端日志在根目录train.log。last.pt、best.pt遵循原保存规则，无合格best时只保留last，不能伪造best。若NaN/Inf或来源守卫失败，保留日志停止，不改标签或阈值绕过。

## 5. 上传预览与正式上传

继续使用确认可用的6681381工具。下块先dry-run，再正式上传，分别记录状态。SSH443只作用于该子shell，不覆盖已有认证设置、不关闭主机校验。上传器排除模型权重，保留本地日志和收据。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
UPLOAD_TOOLS=/home/fss/NeuroFDAN-upload-timeout-tools
OPT_RESULT=/home/fss/NeuroFDAN/outputs/phase6/optimization_control/DEAP/arousal/seed42/deap_arousal_fold01_natural_b40_20ep_001
test "$(git -C "$UPLOAD_TOOLS" rev-parse HEAD)" = 6681381298b74e9b69dc0df0594d5d9c7274acbc
test -z "$(git -C "$UPLOAD_TOOLS" status --porcelain)"
test -f "$OPT_RESULT/train.log"
if [ -n "${GIT_SSH_COMMAND:-}" ]; then echo '已有自定义SSH命令，停止，不覆盖认证设置'; exit 1; fi
export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" "$OPT_RESULT" \
  --log "$OPT_RESULT/train.log" --git_timeout 600 --dry-run
OPT_UPLOAD_ID="$(date +%Y%m%d_%H%M%S)_$$"
OPT_UPLOAD_LOG="/home/fss/NeuroFDAN/outputs/phase6/upload_logs/natural_b40_${OPT_UPLOAD_ID}.log"
OPT_BUNDLE="/home/fss/NeuroFDAN/outputs/phase6/upload_bundles/natural_b40_${OPT_UPLOAD_ID}"
mkdir -p outputs/phase6/upload_logs outputs/phase6/upload_bundles
test ! -e "$OPT_BUNDLE"
bash "$UPLOAD_TOOLS/scripts/phase3_run_logged.sh" "$OPT_UPLOAD_LOG" \
  bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" "$OPT_RESULT" \
  --log "$OPT_RESULT/train.log" --git_timeout 600 --bundle_dir "$OPT_BUNDLE"
cat "$OPT_UPLOAD_LOG.exit"
cat "$OPT_BUNDLE/upload_state.json"
)
```

回传本次SHA和GitHub链接。若上传失败，回传上传日志末尾与收据，不重新训练、不覆盖原目录；上传成功与训练完成分别判断。默认新runs目录不会覆盖任何历史快照。

## 6. 本轮交付与验证范围

仅新增本文；没有改模型、损失、训练器、数据或参数默认值，没有新增教师模型、监督来源或因果边。云端只核验Git、已保存JSON及manifest，静态核对参数与Shell语法，不运行模型、CLI、单测或真实训练。4项测试通过属于本次服务器上传证据；下一次20轮完整训练仍待服务器执行。本次独立文档分支不修改main，不创建PR或合并。
