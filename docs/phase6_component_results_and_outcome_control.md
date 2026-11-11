# Phase6：四格评价已定位条件结果学习瓶颈

## 1. 已核验的服务器证据

提交ad6724fd721674024dca71a77f7737ff3bce5932，目录runs/2026-11-11_180843_unknown_unknown_run_fbda7ce3。逐文件载荷长度和SHA256一致，无跳过项；缺少的通用run_summary.json不是该程序必需文件。

8项测试全部通过，四格摘要completed=True、training_executed=False、real_data_training=False、source_files_unchanged=True。两个gamma均成功严格加载原20轮教师checkpoint，state_unchanged=True、parameter_gradients_absent=True。源码f964547be5300d4c0778e38d8c2f0fef796f1cf7、无已跟踪修改；参考库2048/2048，无不足。上一轮目录冲突已在实际服务器流程中消除。

以下为4次MC平均概率相对原直接干预真值的误差，不是4次误差的平均，也不是分类ACC。

| 组合 | 含义 | gamma0 MSE | gamma1 MSE | gamma0 MAE | gamma1 MAE |
|---|---|---:|---:|---:|---:|
| TT | 真实中介+真实条件结果 | 0.00001973 | 0.00003446 | 0.003483 | 0.004674 |
| TL | 真实中介+已学Head | 0.00478023 | 0.00340943 | 0.051972 | 0.049998 |
| LT | 已学中介+真实条件结果 | 0.00040023 | 0.00028079 | 0.016048 | 0.013244 |
| LL | 已学中介+已学Head | 0.00459745 | 0.00404470 | 0.050562 | 0.052768 |

相对同一经验库TT的MSE：TL约0.004863/0.003267，LT约0.000366/0.000223，LL约0.004704/0.003898。各次重复均保持TL/LL明显高于LT的方向。MC标准误只反映固定模型/经验库下的积分波动，不覆盖训练样本、随机初始化或总体参考抽样误差；不宣称统计显著或百分比因果归因。

真实Ma的事实配对Head MAE为0.108244/0.108983，交叉配对平均约0.123842/0.114375。两种风险对应不同输入分布，不能据此单独证明分布外失效。关键是事实配对本身也拟合不足，问题并非只在交叉配对时出现。

## 2. 已定位与尚未定位的部分

已知条件Head配合经验参考和真实中介能接近直接真值；保留学习中介但替换已知Head后，误差也大幅下降。这将当前合成教师模型的优先瓶颈指向条件结果估计器；中介条件学习仍有次要误差，不能宣称已经恢复全部真实中介分布。

本结果不能区分MLP有限样本方差、训练期Dropout、优化不足、参考分支偏用或局部函数错设各自的贡献。未观察到本次运行异常，不能把统计拟合不足直接叫作实现bug。也不能将教师模型的结论直接套用到真实无中介标签训练。

不再重复：环境启动检查、原20轮、100轮、旧输入保真候选、增加S/K积分次数。原输入保真候选依然不进入真实迁移。

## 3. 推荐的一个小型条件结果对照

只拟合下式，不训练原CNS/ANS中介、ReferenceEncoder或真实数据骨干：

\[
\eta=a^\top m_a+b^\top[x_c,x_a]+c,\qquad q(Y=1\mid m_a,X)=\sigma(\eta).
\]

以两个logits [0,eta]输出，接口仍只有Ma和参考表示，不读取Mc。Head中的X是条件变量，不是SCM中新增的X→Y边。

训练目标是原2048个自然观察样本上、教师权限下的平均二项负对数似然：

\[
\min_{a,b,c}\;\frac1n\sum_i\{\log(1+e^{\eta_i})-y_i\eta_i\}.
\]

没有惩罚项、干预标签监督或额外KL权重调整。采用L-BFGS-B及解析梯度、500次迭代上限、gtol=1e-8/ftol=1e-12；记录收敛和最终梯度，不收敛即停止。完整batch为2048，89个自由参数。训练特征标准化仅是求解预条件化，系数精确换回原坐标，不改原数据预处理。

理由：gamma0下真实条件概率恰为sigma(w_a^T Ma+b)，参考系数为零；线性logit包含正确条件模型，是一个可解释的统计拟合对照。我们不强制拟合参考系数为零，检查其是否学出不必要依赖。gamma1下需要对U|X作logistic-normal混合，通常不再是精确线性logit，因此该设置必须标为近似模型，不能借此声称找到了精确真实Head。

本对照改变的是条件结果估计器整体，包含函数族与求解方式，不能将改善进一步归因为“仅换架构”或“仅换优化器”。它比直接加MLP轮数、削弱KL或加新中介约束更适合当前已定位的瓶颈。也不主张它作为最终真实模型。

权限严格隔离：fit_head只接收训练m_a/x_c/x_a/y，不接收SCM、U、Mc、测试集或干预真值。真实中介仅在这个合成教师对照中合法使用，绝不接入DEAP/HCI。已知概率仅在拟合结束后评分。原三个监督条件的实验不被覆盖，无监督模型并没有因此获得中介标签。

## 4. 预先定义的判读与失败条件

- 对原16点、两个gamma、同训练集、同自然参考库进行配对评价；S256/K128、4次重复保持不变。TT/LT逐次概率须与成功002记录一致，否则停止，不能比较。
- 报告原MLP与对照在真实Ma事实训练/独立测试配对上的观察NLL及真实条件概率误差；再比较TL/LL的干预MSE/MAE、NLL、校准和MC波动。
- gamma0若条件结果误差与TL/LL同时改善，支持“更受限且可充分求解的条件结果估计器有帮助”，不证明潜在中介识别。如果仅训练NLL下降、独立条件/干预误差不改善，拒绝迁移；gamma1近似失配也必须如实报告，不只挑gamma0。
- 这是已知16点上的诊断开发，不是独立的最终确认实验，也不反复调参数挑最好结果。真实迁移前还需处理无标签中介的额外识别约束并另做受控确认。

原始probs_do仍为正式主输出，不事后换成probs_obs。真实LOSO、类别平衡、参考总体、seed和test-selected选模规则均不改。

## 5. 代码与验证边界

新增scripts/phase6_outcome_control.py、tests/test_phase6_outcome_control.py及本报告。新脚本复用evaluate_gamma及唯一FrontDoorEstimator.marginalize；没有另造学习前门积分器。只在独立对照实例里安装线性Head，原checkpoint文件不写回，中介/参考状态前后核对，源文件SHA256前后核对。结果明确training_executed=True且training_scope仅为合成条件结果拟合，不能把它叫成纯前向操作。

4项新测试检查解析梯度、可收敛的两类拟合与常量特征、训练视图权限、Head输入接口。云端只进行静态AST、Shell/内嵌Python语法和Git差异检查，不执行这些测试、CLI或模型；服务器已有8项成功来自原始上传。新增运行行为待下述服务器执行。

## 6. 服务器完整命令

继续使用当前dslenv，不安装或升级依赖。新独立worktree保留旧工具；不切换主项目分支。这里只做小型合成教师Head拟合，不运行mainNeuroFDAN.py，没有真实dataset/label/fold或epoch参数。它使用固定完整batch和求解器收敛迭代，原20轮中介不重训。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
git status --short
git worktree list
git fetch origin feat/neurofdan-phase6-outcome-control
OUTCOME_SHA="$(git rev-parse FETCH_HEAD)"
OUTCOME_TOOLS=/home/fss/NeuroFDAN-phase6-outcome-tools
if [ ! -e "$OUTCOME_TOOLS" ]; then
    git worktree add --detach "$OUTCOME_TOOLS" "$OUTCOME_SHA"
fi
test "$(git -C "$OUTCOME_TOOLS" rev-parse HEAD)" = "$OUTCOME_SHA"
test -z "$(git -C "$OUTCOME_TOOLS" status --porcelain)"
git -C "$OUTCOME_TOOLS" log -1 --oneline
OUTCOME_SOURCE=/home/fss/NeuroFDAN/outputs/phase6/synthetic/paired_tanh_normal_20ep_001
OUTCOME_COMPONENT=/home/fss/NeuroFDAN/outputs/phase6/evaluation/tanh_oracle_components_002
test -f "$OUTCOME_SOURCE/models/gamma0.0_oracle_mediator_last.pt"
test -f "$OUTCOME_SOURCE/models/gamma1.0_oracle_mediator_last.pt"
test -f "$OUTCOME_COMPONENT/probe_summary.json"
OUTCOME_RESULT=/home/fss/NeuroFDAN/outputs/phase6/evaluation/tanh_oracle_outcome_control_001
test ! -e "$OUTCOME_RESULT"
mkdir -p "$OUTCOME_RESULT"
(
cd "$OUTCOME_TOOLS"
bash scripts/phase3_run_logged.sh "$OUTCOME_RESULT/component_tests.log" \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m unittest discover -s tests -p test_phase6_outcome_control.py -v
)
bash "$OUTCOME_TOOLS/scripts/phase3_run_logged.sh" "$OUTCOME_RESULT/train.log" \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -u "$OUTCOME_TOOLS/scripts/phase6_outcome_control.py" \
  --run_dir "$OUTCOME_SOURCE" --component_dir "$OUTCOME_COMPONENT" --output_dir "$OUTCOME_RESULT" \
  --seed 42 --device cpu --workers 0 --diagnostic_level full --S 256 --K 128 --repeats 4
cat "$OUTCOME_RESULT/component_tests.log.exit"
cat "$OUTCOME_RESULT/train.log.exit"
python - <<'PY'
import json
from pathlib import Path
p=Path('/home/fss/NeuroFDAN/outputs/phase6/evaluation/tanh_oracle_outcome_control_001/probe_summary.json')
r=json.loads(p.read_text())
assert r['completed'] and r['source_files_unchanged'] and not r['real_data_training']
assert r['training_executed']
print('仅合成条件结果对照已拟合：',r['training_scope'])
for g in r['gammas']:
    assert g['frozen_core_unchanged'] and g['shared_random_streams_verified']
    print('gamma=',g['gamma'],'拟合诊断=',g['head_fit'])
    for name in ('true_m_learned_h','learned_m_learned_h'):
        print(name,'原=',g['baseline_cells'][name]['against_direct'],
              '新=',g['control_evaluation']['cells'][name]['against_direct'])
PY
)
```

成功后新目录含config.json、probe_summary.json、run_report.txt、两份日志和退出码；局部拟合系数仅保存在models/*.npz，不上传。若失败，保留部分摘要和日志并回传，不修改旧模型或盲目增加迭代。

已有四格核心计算总约5秒；新任务仅两个89参数拟合及相同评价，属于轻量CPU任务，实际耗时记录在head_fit及评价seconds，不承诺固定时间内解决研究问题。

## 7. 结果上传：预览

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
UPLOAD_TOOLS=/home/fss/NeuroFDAN-upload-timeout-tools
OUTCOME_RESULT=/home/fss/NeuroFDAN/outputs/phase6/evaluation/tanh_oracle_outcome_control_001
test "$(git -C "$UPLOAD_TOOLS" rev-parse HEAD)" = 6681381298b74e9b69dc0df0594d5d9c7274acbc
test -z "$(git -C "$UPLOAD_TOOLS" status --porcelain)"
bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" "$OUTCOME_RESULT" \
  --log "$OUTCOME_RESULT/train.log" --log "$OUTCOME_RESULT/component_tests.log" \
  --git_timeout 600 --dry-run
)
```

## 8. 结果上传：正式执行

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
UPLOAD_TOOLS=/home/fss/NeuroFDAN-upload-timeout-tools
OUTCOME_RESULT=/home/fss/NeuroFDAN/outputs/phase6/evaluation/tanh_oracle_outcome_control_001
test "$(git -C "$UPLOAD_TOOLS" rev-parse HEAD)" = 6681381298b74e9b69dc0df0594d5d9c7274acbc
test -z "$(git -C "$UPLOAD_TOOLS" status --porcelain)"
OUTCOME_UPLOAD_ID="$(date +%Y%m%d_%H%M%S)_$$"
OUTCOME_UPLOAD_LOG="/home/fss/NeuroFDAN/outputs/phase6/upload_logs/outcome_control_${OUTCOME_UPLOAD_ID}.log"
OUTCOME_BUNDLE="/home/fss/NeuroFDAN/outputs/phase6/upload_bundles/outcome_control_${OUTCOME_UPLOAD_ID}"
mkdir -p outputs/phase6/upload_logs outputs/phase6/upload_bundles
bash "$UPLOAD_TOOLS/scripts/phase3_run_logged.sh" "$OUTCOME_UPLOAD_LOG" \
  bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" "$OUTCOME_RESULT" \
  --log "$OUTCOME_RESULT/train.log" --log "$OUTCOME_RESULT/component_tests.log" \
  --git_timeout 600 --bundle_dir "$OUTCOME_BUNDLE"
cat "$OUTCOME_UPLOAD_LOG.exit"
cat "$OUTCOME_BUNDLE/upload_state.json"
)
```

回传新的SHA和GitHub链接；失败也上传本次已有日志。新run_id独立，不覆盖组件002或原20轮模型。通用run_summary.json缺少不等于失败，需结合实际completed、求解收敛、源文件及冻结状态判断。

## 9. 正式实验的距离

现在已从一般塌缩排查推进到具体的条件结果学习瓶颈，不再循环环境或积分检查。下一步就是上述一个受控合成拟合对照。它若成功，只提供条件模型改进的依据；真实无中介标签的输入保真候选仍已失败，尚未形成可直接迁移的新方法。

正式LOSO仍需：明确可辩护的潜在中介学习假设；对应观察权限下的合成证据；DEAP/HCI固定单折短程受控验证；冻结方法/主输出/参数与统计协议。已有baseline可作为明确标注的失败基线保留，不把其分类分数包装成已验证的因果机制。无需为了等待放行重跑相同100轮，也不以某个ACC阈值替代这些条件。
