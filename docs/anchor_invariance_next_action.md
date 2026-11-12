# 下一步可执行操作：候选锚点的只读尺度稳定性审查

## 1. 固定交付方式

以后每次发现问题或提出新思路，同时提供：证据及影响、一个具体下一步、完整服务器运行命令、输出位置、上传预览与正式命令、检查标准和停止条件。没有可执行的新实验时明确说明，不以“先考虑/后续讨论”代替已能完成的工作。

本次发现：首批原始幅值锚点有很强被试间差异。下一步检查更紧凑的幅值/偏置稳定候选，先不增加损失，不改训练器或原锚点定义，不启动20轮或LOSO。保护原代码a3f458316cc809e07fb4764e9c26143eb5f33456和首批结果；本轮基于审查报告b4882469542a80b63dd9756adea0146609c21a49创建独立功能分支feat/neurofdan-anchor-invariance-audit。

## 2. 候选的明确数学定义

取窗口长度N=768、fs=128、T=N/fs=6秒。对每个通道，考察x'(t)=g·x(t)+b，g>0、b在该窗口内恒定。

- EEG：只保留原五个相对频带功率，不保留log总功率，32×5=160维。沿用同一个Hann/periodogram实现和频带，不改滤波。
- EMG/ECG：每通道log(mean|x[t+1]−x[t]|/RMS(x−mean(x)))，使用每采样间隔差分，不乘fs，因此比值无量纲。DEAP2维、HCI3维。
- GSR：T·OLS_slope/std(x)，以及Σ max(x[t+1]−x[t],0)/std(x)，共2维。趋势和总正变化均按窗口标准差归一化，不包含原始水平或绝对幅值，不称SCR/HRV。

总ANS维度DEAP4、HCI5。分母太小的窗口有显式无效掩码，常量输入不会因零占位变成“有效信息”。这仅是候选筛查，没有把真实训练损失的目标换成这些值。

在精确计算且有效掩码不变时，上述量满足T(gx+b)=T(x)。这只排除一类测量增益/偏置作用，不等于排除U、去除全部身份信息或保留所有情绪信息。舍弃幅值也可能损失EMG等模态的有效信号，必须由后续有利/失败合成机制反驳，不能先认定更好。

原normalize_subject沿trial轴逐采样位置做变换，其尺度可随时间变化，不属于上述窗口内恒定仿射变换。原始/归一化后相关系数不要求接近1，不以它单独作为通过条件。

## 3. 本步具体测量

继续只读训练被试2–末尾的全部原窗口，保持原通道、窗口顺序和预处理，held-out1不读取，不使用标签。每批提取原窗口候选及原normalize_subject之后的候选，保存有效性、方差、相关和被试间方差占比。

在内存临时构造三类变换：gain（2倍）、offset（加3倍本窗口通道std）、channel_affine（各通道增益0.5–2、加2倍std）。偏置只用于探针，不写回任何信号文件。逐元素比较目标：|Δ|≤1e-5+1e-5|target|，累计最大/平均误差、所有超限元素及mask改变，不能只看平均值。

检查失败会先保存probe_summary.json，再非零退出。completed表示收集完成，必须另看affine_invariance_passed；上传成功不等于性质成立。没有模型、checkpoint、optimizer、参考库或训练步骤。

代码：新增anchor_invariance_candidates.py；在scripts/audit_physiological_anchors.py增加可选--anchor_variant与--check_affine_invariance，默认original保留旧路径；新增tests/test_anchor_invariance.py的8项CPU数学小测试。NeuroFDAN/SCM/数据集/损失/训练器完全不改，原13项测试已由此前服务器上传验证，本步仅运行新增8项无模型测试。

## 4. 服务器完整命令

沿用dslenv，不安装或覆盖依赖。以下整块执行一次；全新目录scale_audit_001，遇到已有目录停止，不删除旧结果。主项目分支不切换。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
git status --short
git worktree list
if [ -n "${GIT_SSH_COMMAND:-}" ]; then echo '已有自定义SSH命令，停止，不覆盖认证设置'; exit 1; fi
export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
timeout 600 git fetch origin feat/neurofdan-anchor-invariance-audit
SCALE_SHA="$(git rev-parse FETCH_HEAD)"
SCALE_TOOLS=/home/fss/NeuroFDAN-anchor-invariance-tools
if [ ! -e "$SCALE_TOOLS" ]; then
    git worktree add --detach "$SCALE_TOOLS" "$SCALE_SHA"
fi
test "$(git -C "$SCALE_TOOLS" rev-parse HEAD)" = "$SCALE_SHA"
test -z "$(git -C "$SCALE_TOOLS" status --porcelain)"
git -C "$SCALE_TOOLS" log -1 --oneline
command -v python
python --version
SCALE_OUTPUT=/home/fss/NeuroFDAN/outputs/phase6/physiology_anchors/scale_audit_001
test ! -e "$SCALE_OUTPUT"
mkdir -p "$SCALE_OUTPUT"
(
cd "$SCALE_TOOLS"
bash scripts/phase3_run_logged.sh "$SCALE_OUTPUT/interface_tests.log" \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m unittest discover -s tests -p test_anchor_invariance.py -v
)
for SCALE_DATASET in DEAP HCI; do
    if [ "$SCALE_DATASET" = DEAP ]; then SCALE_PREFIX=deap; else SCALE_PREFIX=hci; fi
    SCALE_RUN="${SCALE_PREFIX}_scale_stable_audit_001"
    bash "$SCALE_TOOLS/scripts/phase3_run_logged.sh" "$SCALE_OUTPUT/$SCALE_RUN/train.log" \
      env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      python -u "$SCALE_TOOLS/scripts/audit_physiological_anchors.py" \
      --dataset "$SCALE_DATASET" --label arousal --root_dir /home/fss \
      --output_dir "$SCALE_OUTPUT" --run_name "$SCALE_RUN" \
      --held_out_subjects 1 --seed 42 --device cpu --batch_size 40 --workers 0 \
      --diagnostic_level full --anchor_variant scale_stable --check_affine_invariance
    cat "$SCALE_OUTPUT/$SCALE_RUN/train.log.exit"
done
cat "$SCALE_OUTPUT/interface_tests.log.exit"
)
```

本轮无需mainNeuroFDAN.py；审查没有epoch或学习率参数，不编造训练CLI。label仅为记录信息，不访问labels字段；pickle仍会反序列化整份已有字典。

## 5. 输出与停止标准

目录：outputs/phase6/physiology_anchors/scale_audit_001/，包含interface_tests.log和deap_scale_stable_audit_001、hci_scale_stable_audit_001。各子目录包含config.json、probe_summary.json、train.log及.exit。

期望检查：新增8项测试通过；DEAP12400窗、160/4维；HCI2300窗、160/5维；held_out_subject_read=false、labels_field_accessed_or_used=false、training_executed=false；affine_invariance_passed=true，各变换两模态violations=0、mask_changes=0；all_modalities_have_active_features=true。

这是工程预期，不是实际运行结论。任何测试、来源、非有限值或不变性失败都停止，保留并上传日志，不调整容差或数据绕过。被试间方差占比和raw/normalized相关用于描述比较，不设“越低越好”的自动选特征阈值；较低被试差异不等于有情绪信息，数学不变性也不等于中介可识别。

## 6. 上传预览及正式上传

继续使用已验证6681381工具，分别上传两个新子目录；旧stage1和checkpoint不重传、不改写。正式上传每次生成唯一bundle及收据。两份附同一测试日志，只计8项，不计16项。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
UPLOAD_TOOLS=/home/fss/NeuroFDAN-upload-timeout-tools
SCALE_OUTPUT=/home/fss/NeuroFDAN/outputs/phase6/physiology_anchors/scale_audit_001
test "$(git -C "$UPLOAD_TOOLS" rev-parse HEAD)" = 6681381298b74e9b69dc0df0594d5d9c7274acbc
test -z "$(git -C "$UPLOAD_TOOLS" status --porcelain)"
if [ -n "${GIT_SSH_COMMAND:-}" ]; then echo '已有自定义SSH命令，停止，不覆盖认证设置'; exit 1; fi
export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
SCALE_UPLOAD_ID="$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p outputs/phase6/upload_logs outputs/phase6/upload_bundles
for SCALE_PREFIX in deap hci; do
    SCALE_NAME="${SCALE_PREFIX}_scale_stable_audit_001"
    SCALE_RESULT="$SCALE_OUTPUT/$SCALE_NAME"
    if [ ! -d "$SCALE_RESULT" ]; then echo "尚未执行：$SCALE_RESULT"; continue; fi
    bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" "$SCALE_RESULT" \
      --log "$SCALE_RESULT/train.log" --log "$SCALE_OUTPUT/interface_tests.log" \
      --git_timeout 600 --dry-run
    SCALE_UPLOAD_LOG="/home/fss/NeuroFDAN/outputs/phase6/upload_logs/${SCALE_NAME}_${SCALE_UPLOAD_ID}.log"
    SCALE_BUNDLE="/home/fss/NeuroFDAN/outputs/phase6/upload_bundles/${SCALE_NAME}_${SCALE_UPLOAD_ID}"
    test ! -e "$SCALE_BUNDLE"
    bash "$UPLOAD_TOOLS/scripts/phase3_run_logged.sh" "$SCALE_UPLOAD_LOG" \
      bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" "$SCALE_RESULT" \
      --log "$SCALE_RESULT/train.log" --log "$SCALE_OUTPUT/interface_tests.log" \
      --git_timeout 600 --bundle_dir "$SCALE_BUNDLE"
    cat "$SCALE_UPLOAD_LOG.exit"
    cat "$SCALE_BUNDLE/upload_state.json"
done
)
```

回传两份SHA和GitHub链接。若8项测试失败且未生成两个结果子目录，直接回传interface_tests.log及.exit中的错误末尾，不重跑旧训练、不强制继续。

## 7. 收到结果后的具体决定

若数学/来源守卫通过且候选有有效变化，结合旧锚点统计，决定是否把该候选列入下一次配对合成；届时只修改锚点工作目标，不增加损失家族，并提供新的完整学习/上传命令。若候选仍主要反映个体差异、缺乏变化或不变性失败，先拒绝或修复该候选，不进入真实训练。

任何后续合成仍需保留原锚点方案作对照及信息遗漏失败机制，不能因同一候选对gain不敏感就宣称其更有生理意义或能恢复干预概率。论文可把此次结果用于特征设计的数学与数值支持，不能当作情绪识别性能提升。

云端仅执行AST/Shell语法/参数及Git静态检查；新增8项测试和本次数据检查均未运行。没有模型、GPU或真实数据训练；无PR、合并或main修改。
