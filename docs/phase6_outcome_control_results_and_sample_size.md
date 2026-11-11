# Phase6：条件结果对照未通过迁移判据，限定检查样本量

## 1. 已核验结果与结论

服务器提交83c8255e80a79e502e2607c530e6cda246aa7a5b，目录runs/2026-11-11_205906_unknown_unknown_run_d99778cb。全部载荷长度和SHA256一致，无跳过项；缺少的通用run_summary.json不影响完整性。4项测试通过，completed=True、source_files_unchanged=True，原中介/参考状态与公共MC流核验通过。training_executed=True只表示独立合成Head拟合，real_data_training=False。源码为a3bc7b893d2060ee767fa5016f65aeafcc1432c5。

| 指标 | gamma0原MLP | gamma0线性 | gamma1原MLP | gamma1线性 |
|---|---:|---:|---:|---:|
| 真实Ma事实测试条件概率MSE | 0.018077 | 0.012028 | 0.018068 | 0.011672 |
| 真实Ma事实测试观察NLL | 0.713203 | 0.685730 | 0.681805 | 0.662466 |
| TL干预概率MSE | 0.004780 | 0.004979 | 0.003409 | 0.003231 |
| LL干预概率MSE | 0.004597 | 0.004684 | 0.004045 | 0.003863 |

TL为真实中介+拟合Head，LL为旧已学中介+拟合Head。干预MSE基于相同16点、直接干预真值和四次MC平均概率。线性对照改善事实条件风险，却没有一致改善干预概率，不能迁移替换正式OutcomeHead。

两个MLE分别34/49次迭代收敛，最大梯度约2.78e-7/2.37e-7，均远未达到500次上限。gamma0原MLP训练观察NLL为0.551438、测试0.713203；线性为0.622135、0.685730。符合容量和有限样本泛化需要区分的现象，不是唯一根因证明。gamma0真实参考系数为零、拟合参考系数范数约0.39369，未经协方差检验不能声称显著非零。

不增加新模型模块，不调KL，不改正式训练。术语统一为“合成中介真值监督对照”，没有教师—学生或蒸馏结构。

## 2. 一个限定的统计问题

gamma0下线性logit包含真实条件模型，求解器已收敛；2048个带噪二分类样本是否不足以精确估计89个参数，是剩余的一个可直接检验的问题。89/(4×2048)约0.0109，与条件MSE约0.012同量级，只是量级提示，不是理论误差公式或已证明的根因。

预先固定学习曲线：保留已有n=2048结果，只增加8192和32768两点。唯一变化是条件结果拟合样本量。中介仍为原2048样本训练的模型；经验参考仍为原2048自然样本；观察测试512、干预点16、SCM参数和真值、S256/K128、MC重复4次都不变。固定经验参考用于隔离条件拟合误差，不改真实数据的参考总体规则。

原训练块沿用seed1042；追加固定2048样本一块，种子10042、10043……，因此8192是32768的严格前缀。不能直接调用scm.sample(n,同一种子)冒充前缀，因为按变量整块消耗随机数会改变前2048样本。

fit_head仍只接收训练Ma/Xc/Xa/Y；追加样本的U/Mc不进入拟合。真实Ma仅用于该合成监督参考，不提供给无中介标签训练。函数族、无惩罚MLE、求解器均不改。gamma0是正确设定控制，gamma1仍为近似logit族，不能强求相同极限误差。

脚本仅新增--head_train_size，默认None保留原2048行为；head_training_view记录训练指纹及追加流。新增3项测试验证原数据保留、嵌套前缀、非法视图/规模拒绝，连同已有4项共7项，待服务器运行。

判读与停止规则：TT/LT逐次概率必须与原组件002一致；MLE收敛及状态保护必须通过。若gamma0的独立条件风险和TL/LL误差随规模下降，支持有限样本是重要因素，不证明潜在识别。若仅训练目标降低、独立风险不降，停止此方向，不继续扩大规模或加权重。本次后结束这条真值监督诊断支线，回到真实X/Y权限下的可辩护假设和约束。

own_pairs.train仍评估共同的原2048样本，便于与旧报告比较；head_fit.final_mean_nll才是全部扩展样本的训练目标，head_fit.n及head_training_data.n给出规模。不得混淆这些统计总体。

## 3. 服务器运行

继续使用dslenv，不升级环境，不运行mainNeuroFDAN.py、不重训中介、不使用真实数据/GPU。两个新规模分别保存，原2048结果不重跑。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
git status --short
git worktree list
git fetch origin feat/neurofdan-phase6-outcome-sample-size
SIZE_SHA="$(git rev-parse FETCH_HEAD)"
SIZE_TOOLS=/home/fss/NeuroFDAN-phase6-outcome-size-tools
if [ ! -e "$SIZE_TOOLS" ]; then
    git worktree add --detach "$SIZE_TOOLS" "$SIZE_SHA"
fi
test "$(git -C "$SIZE_TOOLS" rev-parse HEAD)" = "$SIZE_SHA"
test -z "$(git -C "$SIZE_TOOLS" status --porcelain)"
git -C "$SIZE_TOOLS" log -1 --oneline
SIZE_TEST_LOG=/home/fss/NeuroFDAN/outputs/phase6/evaluation/outcome_sample_size_tests_001.log
test ! -e "$SIZE_TEST_LOG"
(
cd "$SIZE_TOOLS"
bash scripts/phase3_run_logged.sh "$SIZE_TEST_LOG" \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m unittest discover -s tests -p test_phase6_outcome_control.py -v
)
for HEAD_N in 8192 32768
do
    SIZE_RESULT="/home/fss/NeuroFDAN/outputs/phase6/evaluation/tanh_oracle_outcome_n${HEAD_N}_001"
    test ! -e "$SIZE_RESULT"
    mkdir -p "$SIZE_RESULT"
    bash "$SIZE_TOOLS/scripts/phase3_run_logged.sh" "$SIZE_RESULT/train.log" \
      env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      python -u "$SIZE_TOOLS/scripts/phase6_outcome_control.py" \
      --run_dir /home/fss/NeuroFDAN/outputs/phase6/synthetic/paired_tanh_normal_20ep_001 \
      --component_dir /home/fss/NeuroFDAN/outputs/phase6/evaluation/tanh_oracle_components_002 \
      --output_dir "$SIZE_RESULT" --head_train_size "$HEAD_N" \
      --seed 42 --device cpu --workers 0 --diagnostic_level full --S 256 --K 128 --repeats 4
    cat "$SIZE_RESULT/train.log.exit"
done
)
```

没有epoch/训练batch参数：使用指定规模完整batch求解同一89参数模型。原2048拟合各约0.004/0.005秒，扩展仍是轻量CPU任务，实际以日志为准，Git网络耗时可能更长。失败立即停止，保留部分摘要和日志，不重跑旧训练。

## 4. 上传预览：沿用已成功的SSH 443

SSH配置仅在各自子shell生效，不修改origin或全局设置，不替换密钥。已有自定义GIT_SSH_COMMAND则停止，保留原认证。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
UPLOAD_TOOLS=/home/fss/NeuroFDAN-upload-timeout-tools
test "$(git -C "$UPLOAD_TOOLS" rev-parse HEAD)" = 6681381298b74e9b69dc0df0594d5d9c7274acbc
test -z "$(git -C "$UPLOAD_TOOLS" status --porcelain)"
if [ -n "${GIT_SSH_COMMAND:-}" ]; then echo '已有自定义SSH命令，停止，不覆盖认证设置'; exit 1; fi
export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
for HEAD_N in 8192 32768
do
    SIZE_RESULT="/home/fss/NeuroFDAN/outputs/phase6/evaluation/tanh_oracle_outcome_n${HEAD_N}_001"
    if [ ! -d "$SIZE_RESULT" ]; then echo "尚未生成：$SIZE_RESULT"; continue; fi
    bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" "$SIZE_RESULT" \
      --log "$SIZE_RESULT/train.log" \
      --log /home/fss/NeuroFDAN/outputs/phase6/evaluation/outcome_sample_size_tests_001.log \
      --git_timeout 600 --dry-run
done
)
```

## 5. 正式上传及收据

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
UPLOAD_TOOLS=/home/fss/NeuroFDAN-upload-timeout-tools
test "$(git -C "$UPLOAD_TOOLS" rev-parse HEAD)" = 6681381298b74e9b69dc0df0594d5d9c7274acbc
test -z "$(git -C "$UPLOAD_TOOLS" status --porcelain)"
if [ -n "${GIT_SSH_COMMAND:-}" ]; then echo '已有自定义SSH命令，停止，不覆盖认证设置'; exit 1; fi
export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
SIZE_UPLOAD_ID="$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p outputs/phase6/upload_logs outputs/phase6/upload_bundles
for HEAD_N in 8192 32768
do
    SIZE_RESULT="/home/fss/NeuroFDAN/outputs/phase6/evaluation/tanh_oracle_outcome_n${HEAD_N}_001"
    if [ ! -d "$SIZE_RESULT" ]; then echo "尚未生成：$SIZE_RESULT"; continue; fi
    SIZE_UPLOAD_LOG="/home/fss/NeuroFDAN/outputs/phase6/upload_logs/outcome_n${HEAD_N}_${SIZE_UPLOAD_ID}.log"
    SIZE_BUNDLE="/home/fss/NeuroFDAN/outputs/phase6/upload_bundles/outcome_n${HEAD_N}_${SIZE_UPLOAD_ID}"
    bash "$UPLOAD_TOOLS/scripts/phase3_run_logged.sh" "$SIZE_UPLOAD_LOG" \
      bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" "$SIZE_RESULT" \
      --log "$SIZE_RESULT/train.log" \
      --log /home/fss/NeuroFDAN/outputs/phase6/evaluation/outcome_sample_size_tests_001.log \
      --git_timeout 600 --bundle_dir "$SIZE_BUNDLE"
    cat "$SIZE_UPLOAD_LOG.exit"
    cat "$SIZE_BUNDLE/upload_state.json"
done
)
```

回传两份SHA和链接。第一组若失败，上传已有失败日志；未创建的第二组明确跳过。上传成功不代表拟合成功，需核对completed、head_fit.n、收敛、冻结及源文件未变。模型局部系数仍仅保存在本地models/*.npz，不上传。

## 6. 执行边界与正式实验

本轮云端只读服务器产物、静态追踪、修改独立合成脚本的可选数量参数及测试、检查AST/Shell语法和Git差异；不执行CLI、单元测试、模型、真实训练或GPU实验。新增7项测试待服务器核验，原4项通过来自83c8255实测日志。

核心NeuroFDAN、frontdoor_scm、真实训练器、数据处理和损失均不修改。没有创建PR、合并或修改main。正式实验仍需真实X/Y权限下的学习方案和明确假设、固定单折受控验证及冻结协议；此样本量对照不提供不存在的真实中介标签，也不等同于方法已经可迁移。
