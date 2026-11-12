# 方案A首次拟合结果与上传恢复

## 已核对的证据

依据服务器回传终端日志，运行源码为 `30152089b943ccf904d205810c72fb20161a483a`：

- 服务器181项测试通过（15.077秒）。负参数测试中的argparse error随后对应ok，不是训练入口异常。
- 固定自然训练子集400窗执行300次优化step；优化NLL从0.683808变至0.665313，各检查点3次MC的BA均为0.5。
- 程序按既定验收规则退出3，`fold20=not_started`。没有完成20轮fold训练，不能称为正式fold或LOSO成绩。
- 结果上传在fetch阶段等待600秒超时；本次没有进入commit或push，没有成功的结果SHA。旧失败bundle无需删除。
- 源码第1步会检查主NLL到CNS、ANS、CNS→ANS和OutcomeHead的梯度是否为正且有限；能继续至300步说明该首步守卫没有触发。这只排除了首步整条路径完全断开，不证明梯度足够、后续不衰减或中介有有效信息。
- BA=0.5不单独证明全部预测某一类，也不证明SCM结构在理论上不能拟合。必须读取已有完整混淆矩阵、类别数、常数NLL基线和分项梯度，才能决定下一项受控改动。

当前主线仍是CNS→ANS层级中介、以`probs_do`为主要输出的前门形式预测。现阶段首先要让该输出在训练数据上具备可重复判别；事实输出是诊断对照。此轮恢复不改变Scheme A目标，不重新训练，不加入教师、锚点或辅助分类头。

## 上传器的最小修复

旧流程已有`fetch --depth=1`，但最新提交的树仍引用所有历史run文件，随后checkout会取得整个树的内容。仅延长超时或再次使用depth=1不能消除这部分传输。

新流程在临时仓库请求`--depth=1 --filter=blob:none --no-tags`，用`read-tree`复用旧树，只将经白名单及SHA256校验的新载荷写入索引；`write-tree --missing-ok`复用远端已存在的旧对象引用，`commit-tree -p`保留父提交。没有checkout旧结果文件，也不将旧文件暂存为删除。只普通push；并发non-fast-forward时从新tip重新追加，不强推。GitHub支持该过滤，但网络、认证及元数据获取仍可能失败，不能保证固定完成时间。

每项新载荷先安全读取并校验，再hash已经校验的同一份字节；不在hash前再次打开原bundle。原run、旧bundle、实验worktree保持不变。已有同名`runs/<id>`拒绝覆盖。新收据保留`parent_sha`、`candidate_sha`、transport和远端核验状态。

`summarize_scheme_a_fit.py`只读取JSON，打印类别计数、每次MC混淆矩阵、预测计数、概率均值/标准差、常数基线差、首/末分项梯度和末次CNS/ANS posterior统计。不读取checkpoint、信号或环境凭证，不导入torch。梯度来自每步更新前train模式，而MC矩阵/posterior来自更新后eval模式；摘要保留时间及模式区别，不将二者当成严格对齐比较。

## 服务器操作：执行这一块即可

继续使用已有`dslenv`，不安装依赖；新工具worktree不会切换原项目分支。命令只获取修复代码分支，不预先fetch整个`server-diagnostics`结果分支。

```bash
cd /home/fss/NeuroFDAN
conda activate dslenv
(
set -euo pipefail
if [ -z "${GIT_SSH_COMMAND:-}" ]; then
    export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
fi
git status --short
git worktree list
timeout 600 git fetch --no-tags origin fix/neurofdan-partial-diagnostics-upload
RECOVERY_SHA="$(git rev-parse FETCH_HEAD)"
RECOVERY_TOOLS=/home/fss/NeuroFDAN-upload-recovery-tools
if [ ! -e "$RECOVERY_TOOLS" ]; then
    git worktree add --detach "$RECOVERY_TOOLS" "$RECOVERY_SHA"
fi
test "$(git -C "$RECOVERY_TOOLS" rev-parse HEAD)" = "$RECOVERY_SHA"
test -z "$(git -C "$RECOVERY_TOOLS" status --porcelain)"
git -C "$RECOVERY_TOOLS" log -1 --oneline
bash "$RECOVERY_TOOLS/scripts/recover_scheme_a_upload.sh" \
    /home/fss/NeuroFDAN/outputs/phase6/scheme_a/DEAP/arousal/seed42/deap_arousal_fold02_scheme_a_001 \
    /home/fss/NeuroFDAN/outputs/phase6/scheme_a/upload_recovery
)
```

脚本执行顺序：

1. 先在终端和独立`summary.log`输出已有结果关键内容；即使网络中断也可分析。
2. 执行19项上传/摘要相关服务器测试（11项传输、4项退出证据、4项摘要）；Git集成测试只用临时`file://`裸仓库，不联网、不运行模型。测试不通过则停止发布，保留摘要及测试日志。
3. 对原运行目录做`--dry-run`预览，随后正式上传；明确`--git_timeout 600 --max_file_mib 16 --max_total_mib 64`。
4. 自动附带新摘要、测试日志和已有`interface_tests.log`、`fit_check.log`及`.exit`证据。`train.log`仅在实际存在时附带；不手工创建虚假的训练产物。
5. 为新恢复记录创建独立日志、bundle和远端run目录，打印退出状态与收据。

**本轮无需运行`mainNeuroFDAN.py`，也无需重跑300步拟合。** 上方一块同时完成服务器读取和GitHub上传，无需再运行旧的全流程训练脚本。

## 准确的源目录、日志与回传

上传源始终是：

`/home/fss/NeuroFDAN/outputs/phase6/scheme_a/DEAP/arousal/seed42/deap_arousal_fold02_scheme_a_001`

新增恢复日志：

`/home/fss/NeuroFDAN/outputs/phase6/scheme_a/upload_recovery/logs/<自动日期_PID>/`

其中包含`summary.log`、`recovery_tests.log`、`preview.log`、`upload.log`及各自`.exit`。快照及`upload_state.json`位于同级`upload_recovery/bundles/<自动日期_PID>/`。脚本自动打印实际路径，无需用户拼接。

成功后回传工具最后打印的**结果提交SHA和GitHub链接**。结果快照中的原`probe_summary.json`、`config.json`和manifest由Codex逐项读取，不需要手抄统计。若再次失败，直接回传本次终端输出；摘要在网络步骤之前已打印。若失败发生在push之后，收据中的candidate需先核对远端，不能把结果未知当成未上传后立刻重复。

此前`/home/fss/NeuroFDAN-upload-timeout-tools`及失败bundle保持原样；本轮使用新修复worktree的上传器。

## 收到完整结果后如何决策

先判断三次MC是否都只预测多数类，以及末NLL是否只是逼近该400窗子集的最优常数NLL；再看任务梯度是否随step衰减、与已加权KL/fire是否冲突、CNS/ANS的mu和sigma是否保留输入差异、MC概率变化是否大于输入相关变化。

- 若训练和评价均接近常数基线，优先定位表示/梯度瓶颈；不以更换评价输出或降低门槛晋级。
- 若训练S/K下有进展而评价S/K积分后消失，才设计针对MC噪声和训练/推断一致性的单因素验证。
- 若任务梯度正常而只输出先验，仍需结合各模块实际信号判断；非零梯度不能证明信息充分。

这些是待完整数据支持的决策条件，不是已经证实的根因。下一项涉及核心模型的改动需先明确修改哪部分、数学目标及否定标准，再与用户确认。当前不启动正式LOSO，不重复长训。

## 验证边界

本轮云端仅AST语法检查、`bash -n`和Git差异检查。新19项测试未在云端运行，也不把此前181项通过当作本补丁已通过。运行与Git集成行为由上述服务器命令验证后自动记录。
