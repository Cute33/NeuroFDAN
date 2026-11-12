# 生理特征锚定A：数学目标、反驳性验证与服务器命令

## 1. 决策与实现边界

采用路线1。先验证观测生理统计是否构成有效的中介学习工作约束，不直接改真实训练器，不启动LOSO，不改SCM。保护基线5a77e1b2b75180f2c76149540c8fbeabbcfa9e65，独立分支feat/neurofdan-physiology-anchors。

本轮提供实际提取器、概率中介读出损失、训练来源审查、三组配对合成、图表导出及13项针对性测试。运行代码待服务器执行；云端不执行模型、CLI或单元测试。新代码不是伪代码，但没有实测效果结论。

真实训练接入暂不进行：先检查原文件统计的合理性，并用合成验证反驳新约束。不能把“用户同意研究A”解释成已经允许绕过合成闸门宣布A有效。

## 2. 原窗口锚点定义及实际维度

数据源是现有被试文件，不增加下载或格式转换。读取仍为Dataset/DEAP-python/1.npy…32.npy、HCI-python/1.npy…24.npy，内部pickle字典。128Hz；DEAP去384点基线后每trial10个768点窗口；HCI每trial5窗。保留原通道索引、来源顺序和训练被试集合。

锚点从文件内截取后的**归一化前窗口**计算；骨干仍使用原normalize_subject输出，不改变预处理。原沿trial轴、逐采样位置的标准化可能改变频谱，不能直接把标准化后的频带值当成原始生理功率。本轮同时比较二者相关性，只作审查，不据此自动换预处理。

| 模态 | 预先指定特征 | 维度 |
|---|---|---:|
| CNS | 每EEG通道log总带功率及delta/theta/alpha/beta/gamma相对功率 | 32×6=192 |
| DEAP ANS | 两路EMG各log去均值RMS、log平均绝对时间导数；末路GSR均值、线性斜率、log标准差、平均正时间导数 | 2×2+4=8 |
| HCI ANS | 三路ECG各上述两种波形统计；末路GSR同上 | 3×2+4=10 |

EEG频带采用左闭右开[1,4)、[4,8)、[8,13)、[13,30)、[30,45)Hz；五个相对功率之和为1，因此不宣称它们为独立测量或满秩proxy。去均值后乘周期Hann窗，采用单边periodogram：PSD=|FFT|²/(fs·Σw²)，内部非DC/Nyquist频点乘2；频带积分乘df=128/768。总功率取五带之和，log为自然对数。

近零能量EEG通道和不可定义的log波形统计有显式有效掩码；NaN/Inf直接报错；有效样本不足2或训练std≤1e-6的特征禁用。没有用零损失占位伪造可用锚点。

ECG统计不是HRV、心率或已定位的R峰指标；GSR统计不是已分解的SCR或校准的皮肤电导。文件单位、滤波历史、EMG带宽及伪迹情况仍需根据原数据说明核对，不伪造仪器校准。EEG通道先用0–31编号，未核验蒙太奇前不绘制解剖定位图。

## 3. 数学目标及梯度路径

令a_c=T_c(X_c^raw)、a_a=T_a(X_a^raw)，是同窗口的确定性观测统计，不是新的SCM节点、真实中介标签或独立测量。仅由训练被试计算每特征有效均值μ_a及标准差s_a；测试复用训练值，不重新拟合：z_j=(a_j-μ_j)/max(s_j,1e-6)。

新增两个线性训练读出：r_c:R^64→R^192，r_a:R^64→R^8或R^10。前者只读采样Mc，后者只读采样Ma；没有从CNS特征到ANS读出的旁路。读出不进入OutcomeHead、参考表示或推断输出。

线性CNS读出秩至多64，不要求对任意192维目标达到零误差，也不宣称无损或充分统计量。若既有滤波使某频带不可用，应据审查记录其无效性，不重新滤波追求结果。锚点可预测性还可能受到原标准化的信息损失影响，不能把重建失败自动归因于中介结构。

对每个模态v∈{c,a}，有效掩码V_ij同时考虑原窗口有效性和训练特征可用性：

\[
L_{anchor,v}=\frac{\sum_{i=1}^{B}\sum_{s=1}^{S}\sum_{j=1}^{d_v}
V_{ij}[r_v(m_{v,i}^{(s)})_j-z_{v,ij}]^2}
{2S\sum_{i,j}V_{ij}}.
\]

两个模态分别归约，避免192维CNS因维数更多而自动压过ANS。使用实际采样m，不只约束mu；不直接强迫sigma缩小或均值拉开。

\[
L=L_{base}+\alpha_c L_{anchor,c}+\alpha_a L_{anchor,a},\quad
L_{base}=\lambda_{obs}CE(logits_{obs},Y)+\beta_t(KL_c+KL_a)+w_{fire}L_{fire}.
\]

预设α_c=α_a=0.1，仅为首轮工作权重，不是已调优参数；β_t、KL归约和原损失定义均不变。合成fire=0；未来若接入HCI则不得默默取消其原fire项。两个alpha均为0时恢复原模型损失；各自可关闭作消融。no_cns2ans开关复用原估计器。

梯度：L_anchor,c→Mc→CNS；L_anchor,a→Ma→ANS及Mc→CNS，除非显式切断原电流路径。target提取不反向，标准化参数冻结，训练期读出可更新。记录加权/未加权锚点损失及每轮首批分项梯度、与事实CE的夹角；不把首批梯度当全epoch平均或真实因果效应。

## 4. 假设、预期及否定条件

工作假设是：选定统计在同类数据采集条件下与中介具有稳定的可预测关系，有限读出能够让概率中介保留一部分有解释意义的输入信息。单位方差平方误差只是工作评分规则，不宣称锚点误差真实高斯或独立于U。

这并未排除Head只依赖自身参考的最优解，也没有新增识别信息。不得把锚点称为真实Mc/Ma、把读出解释成M→X结构边，或声称解决仅观察X/Y的非参数识别。被试身份或伪迹也可能被保留；原窗口审查报告每特征的被试间方差占比，用于警惕这一风险，不自动去除身份信息。

预期是待检验的方向：独立测试锚点可解码性改善；mu/sigma条件变化不再完全被噪声覆盖；CNS→ANS响应保留；干预概率误差同时改善。不能预报真实ACC或承诺一定优于baseline。

否定条件：只有锚点误差或观察ACC变好，而干预概率误差更差；sigma明显趋近下界；CNS→ANS路径被削弱；错配机制仍产生过度自信；或只在专门有利的机制上有效。失败应保留，不继续按测试结果扫alpha或挑更好输出。

## 5. 三组配对合成与失败机制

同一机制内：oracle_mediator（真实中介监督参考，不是蒸馏教师）、observation_only、observation_physiological_anchors。后两组训练不持有真实M/U/do概率；候选target仅由本样本X确定。三组初始化、X/Y、数据顺序和每批初始随机流均配对，最后轮评价，不用测试真值选epoch。

X为16/8维高斯系数，沿用X_c=γA_cU+ε_c、X_a=γA_aU+ε_a。系数由固定线性基单射渲染成1路合成EEG、2路合成波形及1路合成GSR，再调用**同一个实际锚点提取器**。这里的波形是数学测试载体，不是逼真的EEG/EMG生理生成模型。学习模型读取完整16/8维系数，参考维度24，不预先把输入截成锚点；不复现真实13440/13824维骨干。锚点合成维度为6/8，真实为192及8/10。

- aligned：ψ_v(X_v)是固定tanh(a_v/3)复制到原输入维数；Mc=tanh(Bcψ_c+ε_mc)，Ma=tanh(Baψ_a+CcaMc+ε_ma)。这是有利的机制，不声称真实数据满足。
- misaligned：ψ_v(X_v)=X_v，回到原有符号系数决定中介；相同统计会遗漏符号/相位信息。对齐组CNS全局符号翻转不变，错配组不变性被破坏，测试显式覆盖。两机制Y分布不同，只在机制内比较监督组，不混算收益。

两组均Y~Bernoulli(sigmoid(w_a^TMa+γw_u^TU+b))，所有外生噪声独立，Cca非零，没有Mc→Y。固定γ∈{0,1}；直接真值重采边缘U，独立前门真值通过参考X'的U|X'求积，不调用直接函数冒充前门。真值不一致超过4倍组合近似SE+0.01时停止拟合；这只是工程守卫，不是严格等价检验。

额外保存原phase6_nonidentifiability反例。aligned只提供条件性工作模型，不声称排除了同观测不同干预的其他SCM。

normal族匹配现有真实模型，但相对tanh生成器有分布错设；tanh_normal为预定义分布匹配对照，不能按更好ACC事后挑族作为正式模型。S/K/维度/Head等继续复用原FrontDoorEstimator，不复制一套前门估计器。

## 6. 如何形成论文图表

每个机制/γ输出一张三面板PNG与PDF，数值同源写metrics.csv和probe_summary.json：

1. **干预概率MSE柱图**：三组并列、训练比例常数虚线。证明是否改善干预估计，不只是观察分类。
2. **独立测试锚点线性探针MSE**：CNS/ANS分别画。所有组统一用训练数据拟合λ=1的ridge探针，测试不拟合；不是拿候选已训练读出去对比baseline随机读出。特征为共同噪声S=64估计的E[M|X]，仅用于诊断。逐特征R²一并保存，可为负。
3. **真实vs估计do概率散点图**：展示偏置、趋常数及失败点，三个条件全部展示，不只挑成功组。

图中文字采用固定英文论文缩写以避免服务器缺中文字体；本文、help和运行说明为中文。smoke图标题明确标记SMOKE，不能用于论文效果结论。图不加虚构置信区间；MC重复不是多训练种子。

history.jsonl另提供CE、两项锚点损失、KL和分项梯度学习曲线数据；既有evaluate输出观察ACC/Macro-F1/BA、两路径概率、真实中介条件评分、后验mu/sigma输入依赖、CNS→ANS响应及独立MC误差。真实中介坐标评分对重参数化敏感，不能单独作为恢复判据。

正式论文若方案后续成立：主表仍是probs_do的完整LOSO；消融为α=0、单侧alpha=0、no_cns2ans及同checkpoint的无参考边缘化对比。真实锚点图必须来自冻结后模型的独立测试被试。当前没有生成任何真实数据模型的新成绩、归因图或生理机制结论。

## 7. 第一阶段服务器命令：先不要运行真实训练

沿用dslenv，已有requirements含numpy/scipy/torch/matplotlib，不自动安装或覆盖环境。获取独立worktree，项目原分支和旧输出不动：

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
git status --short
git worktree list
if [ -n "${GIT_SSH_COMMAND:-}" ]; then echo '已有自定义SSH命令，停止，不覆盖认证设置'; exit 1; fi
export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
timeout 600 git fetch origin feat/neurofdan-physiology-anchors
ANCHOR_SHA="$(git rev-parse FETCH_HEAD)"
ANCHOR_TOOLS=/home/fss/NeuroFDAN-physiology-anchor-tools
if [ ! -e "$ANCHOR_TOOLS" ]; then
    git worktree add --detach "$ANCHOR_TOOLS" "$ANCHOR_SHA"
fi
test "$(git -C "$ANCHOR_TOOLS" rev-parse HEAD)" = "$ANCHOR_SHA"
test -z "$(git -C "$ANCHOR_TOOLS" status --porcelain)"
git -C "$ANCHOR_TOOLS" log -1 --oneline
command -v python
python --version
bash "$ANCHOR_TOOLS/scripts/run_physiology_anchor_stage1.sh" \
    /home/fss \
    /home/fss/NeuroFDAN/outputs/phase6/physiology_anchors/stage1_001
)
```

该版本化脚本依次执行：13项CPU小测试；DEAP/HCI训练被试2–末尾的全部原窗口只读审查（arousal仅作任务标识，不访问labels值）；最后执行下面的完整合成主函数命令。测试失败立即停止，没有真实模型训练、优化器更新或checkpoint访问发生在数据审查中。合成和其中小测试有明确的CPU临时模型更新，不把整批操作说成纯只读。

下面是脚本内的合成命令，**已由上述脚本调用，不要再对相同目录执行一次**：

```bash
cd /home/fss/NeuroFDAN
ANCHOR_TOOLS=/home/fss/NeuroFDAN-physiology-anchor-tools
ANCHOR_STAGE=/home/fss/NeuroFDAN/outputs/phase6/physiology_anchors/stage1_001
bash "$ANCHOR_TOOLS/scripts/phase3_run_logged.sh" "$ANCHOR_STAGE/synthetic_normal_smoke_001/train.log" \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -u "$ANCHOR_TOOLS/physiology_anchor_experiment.py" \
  --output_dir "$ANCHOR_STAGE" --run_name synthetic_normal_smoke_001 \
  --seed 42 --device cpu --workers 0 --diagnostic_level full \
  --epochs 1 --batch_size 64 --train_size 128 --test_size 32 --points 4 \
  --gammas 1 --mechanisms aligned misaligned --lr 0.001 \
  --alpha_c 0.1 --alpha_a 0.1 --mediator_family normal \
  --mc_repeats 2 --direct_n 4096 --oracle_s 128 --oracle_k 128 --smoke --plots
```

首批输出：stage1_001/interface_tests.log及.exit；deap_training_anchor_audit_001、hci_training_anchor_audit_001、synthetic_normal_smoke_001三个子目录各有train.log及.exit、config.json、probe_summary.json。前两者应为12400/2300训练窗口、192/8及192/10维，无测试被试文件读取；检查有效维度、原始/归一化相关和被试间方差占比。smoke只检验配对学习、积分与导出接口，不评价候选有效性。

## 8. 上传预览及正式上传

下面分别上传三个子目录，不能直接上传阶段根目录以免旧白名单工具遗漏任意命名子目录。共用一份测试日志，不把它计成39项测试。PNG/PDF、CSV及权重不在6681381白名单内；图表留在服务器，可用Xshell/SFTP下载，所有评价数值已保存在可上传的probe_summary.json内。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
UPLOAD_TOOLS=/home/fss/NeuroFDAN-upload-timeout-tools
ANCHOR_STAGE=/home/fss/NeuroFDAN/outputs/phase6/physiology_anchors/stage1_001
test "$(git -C "$UPLOAD_TOOLS" rev-parse HEAD)" = 6681381298b74e9b69dc0df0594d5d9c7274acbc
test -z "$(git -C "$UPLOAD_TOOLS" status --porcelain)"
if [ -n "${GIT_SSH_COMMAND:-}" ]; then echo '已有自定义SSH命令，停止，不覆盖认证设置'; exit 1; fi
export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
ANCHOR_UPLOAD_ID="$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p outputs/phase6/upload_logs outputs/phase6/upload_bundles
for ANCHOR_NAME in deap_training_anchor_audit_001 hci_training_anchor_audit_001 synthetic_normal_smoke_001; do
    ANCHOR_RESULT="$ANCHOR_STAGE/$ANCHOR_NAME"
    if [ ! -d "$ANCHOR_RESULT" ]; then echo "尚未执行：$ANCHOR_RESULT"; continue; fi
    bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" "$ANCHOR_RESULT" \
      --log "$ANCHOR_RESULT/train.log" --log "$ANCHOR_STAGE/interface_tests.log" \
      --git_timeout 600 --dry-run
    ANCHOR_UPLOAD_LOG="/home/fss/NeuroFDAN/outputs/phase6/upload_logs/${ANCHOR_NAME}_${ANCHOR_UPLOAD_ID}.log"
    ANCHOR_BUNDLE="/home/fss/NeuroFDAN/outputs/phase6/upload_bundles/${ANCHOR_NAME}_${ANCHOR_UPLOAD_ID}"
    test ! -e "$ANCHOR_BUNDLE"
    bash "$UPLOAD_TOOLS/scripts/phase3_run_logged.sh" "$ANCHOR_UPLOAD_LOG" \
      bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" "$ANCHOR_RESULT" \
      --log "$ANCHOR_RESULT/train.log" --log "$ANCHOR_STAGE/interface_tests.log" \
      --git_timeout 600 --bundle_dir "$ANCHOR_BUNDLE"
    cat "$ANCHOR_UPLOAD_LOG.exit"
    cat "$ANCHOR_BUNDLE/upload_state.json"
done
)
```

若仅小测试失败、三个目录均不存在，只回传interface_tests.log及.exit的错误末尾即可，不重新训练或强制跳过测试。其他阶段失败也上传已生成目录。正常时回传三份SHA和链接，不要求密码、Token或私钥。

## 9. 下一阶段预定20轮命令——首批核验前不运行

这是预先记录的验证计划，不要求当前就启动。首批完成且审查后，按normal/tanh_normal各自配对，机制aligned/misaligned、γ=0/1全部报告，不按最好结果选择族、机制或epoch。不是正式真实数据LOSO。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
ANCHOR_TOOLS=/home/fss/NeuroFDAN-physiology-anchor-tools
test -z "$(git -C "$ANCHOR_TOOLS" status --porcelain)"
for ANCHOR_FAMILY in normal tanh_normal; do
    ANCHOR_RUN="paired_${ANCHOR_FAMILY}_20ep_001"
    ANCHOR_OUTPUT=/home/fss/NeuroFDAN/outputs/phase6/physiology_anchors/learning
    test ! -e "$ANCHOR_OUTPUT/$ANCHOR_RUN"
    bash "$ANCHOR_TOOLS/scripts/phase3_run_logged.sh" "$ANCHOR_OUTPUT/$ANCHOR_RUN/train.log" \
      env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      python -u "$ANCHOR_TOOLS/physiology_anchor_experiment.py" \
      --output_dir "$ANCHOR_OUTPUT" --run_name "$ANCHOR_RUN" \
      --seed 42 --device cpu --workers 0 --diagnostic_level full \
      --epochs 20 --batch_size 128 --train_size 2048 --test_size 512 --points 16 \
      --gammas 0 1 --mechanisms aligned misaligned --lr 0.001 \
      --alpha_c 0.1 --alpha_a 0.1 --mediator_family "$ANCHOR_FAMILY" \
      --mc_repeats 3 --direct_n 32768 --oracle_s 1024 --oracle_k 512 --plots
done
)
```

每族12个拟合条件，共24个、每个20轮，串行CPU合成，不要求GPU。计算规模公开，不把它说成单个20轮；smoke后再评估是否运行。本轮先上传首批三个目录。以下是后续20轮批次对应的完整上传命令，同样先预览再上传；仅在实际运行该批后使用。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
UPLOAD_TOOLS=/home/fss/NeuroFDAN-upload-timeout-tools
test "$(git -C "$UPLOAD_TOOLS" rev-parse HEAD)" = 6681381298b74e9b69dc0df0594d5d9c7274acbc
test -z "$(git -C "$UPLOAD_TOOLS" status --porcelain)"
if [ -n "${GIT_SSH_COMMAND:-}" ]; then echo '已有自定义SSH命令，停止，不覆盖认证设置'; exit 1; fi
export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
ANCHOR_UPLOAD_ID="$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p outputs/phase6/upload_logs outputs/phase6/upload_bundles
for ANCHOR_FAMILY in normal tanh_normal; do
    ANCHOR_RESULT="/home/fss/NeuroFDAN/outputs/phase6/physiology_anchors/learning/paired_${ANCHOR_FAMILY}_20ep_001"
    if [ ! -d "$ANCHOR_RESULT" ]; then echo "尚未运行：$ANCHOR_RESULT"; continue; fi
    bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" "$ANCHOR_RESULT" \
      --log "$ANCHOR_RESULT/train.log" --git_timeout 600 --max_file_mib 16 --max_total_mib 64 --dry-run
    ANCHOR_UPLOAD_LOG="/home/fss/NeuroFDAN/outputs/phase6/upload_logs/anchor_${ANCHOR_FAMILY}_${ANCHOR_UPLOAD_ID}.log"
    ANCHOR_BUNDLE="/home/fss/NeuroFDAN/outputs/phase6/upload_bundles/anchor_${ANCHOR_FAMILY}_${ANCHOR_UPLOAD_ID}"
    test ! -e "$ANCHOR_BUNDLE"
    bash "$UPLOAD_TOOLS/scripts/phase3_run_logged.sh" "$ANCHOR_UPLOAD_LOG" \
      bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" "$ANCHOR_RESULT" \
      --log "$ANCHOR_RESULT/train.log" --git_timeout 600 --max_file_mib 16 --max_total_mib 64 --bundle_dir "$ANCHOR_BUNDLE"
    cat "$ANCHOR_UPLOAD_LOG.exit"
    cat "$ANCHOR_BUNDLE/upload_state.json"
done
)
```

回传两份SHA和链接；任何阶段失败同样保留记录，不把completed=false的部分结果当正式方法验证。

## 10. 实际代码、测试及下一步

- physiological_anchors.py：唯一锚点提取与标准化/读出损失，可供未来真实数据接入；没有改Dataset或训练器。
- anchor_synthetic_scm.py：两种固定SCM机制、单射信号渲染、独立直接/前门真值。
- physiology_anchor_experiment.py：监督白名单、三组配对训练、原前门评价、统一ridge探针、CSV/PDF/PNG。
- scripts/audit_physiological_anchors.py：训练来源读取、原/归一化锚点对比、被试方差，不读测试文件或labels值。
- scripts/run_physiology_anchor_stage1.sh：先小测试，再两个只读审查，最后1轮smoke，失败停止。
- tests/test_physiological_anchors.py：13项覆盖频带、常量/NaN、通道/来源配对、训练统计冻结、S轴归约、权限、alpha0、梯度路径、渲染秩、两种真值和CPU优化。

已有全部轻量测试未在云端重跑；新增13项也未运行，不能写“通过”。仅做AST、Shell语法、参数静态核对与Git差异检查。真实训练接入仍需通过当前闸门后在独立配置中实现，不假装当前mainNeuroFDAN.py已有锚点开关。原SCM、OutcomeHead、真实骨干、损失、数据、自然训练总体、BN、seed42及test-selected协议均未改变；不修改main，不创建PR或合并。
