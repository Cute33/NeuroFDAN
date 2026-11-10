# Phase 5：日志分层与服务器结果回传

服务器目录/home/fss/NeuroFDAN。沿用当前环境和真实数据，不重装PyTorch/CUDA。本轮Codex只审查代码和用户日志，不执行测试、上传演练或模型实验。新工具尚需下面的服务器dry-run及首次实际上传验证。

## 1. 获取工程版本，保护Phase 4

如果Phase 4任务已结束、工作区干净，可以切到新代码分支：

```bash
cd /home/fss/NeuroFDAN
git status --short
git fetch origin
if git show-ref --verify --quiet refs/heads/feat/neurofdan-phase5-diagnosis; then
    git switch feat/neurofdan-phase5-diagnosis
else
    git switch --track -c feat/neurofdan-phase5-diagnosis origin/feat/neurofdan-phase5-diagnosis
fi
git pull --ff-only origin feat/neurofdan-phase5-diagnosis
git rev-parse HEAD
```

不要删除outputs/phase4，也不要把已有诊断结果复制为正式新实验。Phase 4保护分支和bc4d642b44b408419b489e084cb12d8c7a48c8e9保持不变。

**如果Phase 4仍在运行，优先不切换其代码。** 可以只创建独立工具worktree来上传现有结果，不停止实验、不安装环境：

```bash
cd /home/fss/NeuroFDAN
git fetch origin
git worktree add --detach /home/fss/NeuroFDAN-phase5-tools origin/feat/neurofdan-phase5-diagnosis
bash /home/fss/NeuroFDAN-phase5-tools/scripts/upload_run_diagnostics.sh \
  /home/fss/NeuroFDAN/outputs/phase4/DEAP/arousal/seed42/deap_arousal_fold01_full \
  --log /home/fss/NeuroFDAN/logs/phase4/deap_arousal_fold01_full.log --dry-run
```

路径已存在时不要重复worktree add或强制覆盖。工具worktree是detached HEAD，manifest会分别记录工具checkout和实验目录所在checkout；它们都不自动证明旧训练启动时的SHA。首次预览后按第3节去掉dry-run实际上传。运行中的文件可能继续写入，快照不是多文件事务；变化的文件会跳过或标记尾行省略，需要看manifest并可再次上传新快照。

## 2. 首次认证与作者配置

工具仅接受Cute33/NeuroFDAN的标准无明文凭证origin地址。先在服务器检查，但不要把包含凭证的remote输出粘贴到聊天：

```bash
cd /home/fss/NeuroFDAN
git config --get user.name
git config --get user.email
git ls-remote --heads origin
```

缺少作者信息时，用自己的实际信息设置git config user.name/user.email；可用GitHub账户设置中提供的noreply邮箱。不要使用编造身份，不需要把邮箱发给Codex。

仅在缺少配置时执行，输入自己的信息：

```bash
cd /home/fss/NeuroFDAN
read -r -p 'Git作者名称：' phase5_git_name
read -r -p 'Git作者邮箱（可用账户noreply邮箱）：' phase5_git_email
git config user.name "$phase5_git_name"
git config user.email "$phase5_git_email"
```

认证优先复用已有SSH agent或全局Git凭证助手。SSH方式可在服务器自行执行ssh -T git@github.com确认身份（GitHub成功认证提示仍可能退出1，需看文字）；不要上传私钥。HTTPS如果已有GitHub CLI且选择用它，可在服务器通过gh auth login和gh auth setup-git完成，Token不写入脚本、远程URL或训练日志。没有gh时用服务器原有凭证助手，不要求安装新工具。

工具启用非交互Git认证；它不会在训练终端索取密码。仅仓库本地配置的特殊credential helper未必可用于临时仓库，建议使用已有全局助手或SSH agent。ls-remote成功只证明读取访问，push写权限仍需真实上传验证。认证失败返回非零并保留本地快照，不声称成功。

## 3. 上传已有Phase 4结果

先激活原Python环境。上传工具不导入torch；校验/重编码矩阵NPZ时需要已有NumPy，无新增依赖。下面用arousal路径举例，改成服务器实际存在的任务/run，不根据所贴日志猜label。

第一次可先预览，结果写到系统临时目录，不联网、不提交、不修改实验文件：

```bash
cd /home/fss/NeuroFDAN
bash scripts/upload_run_diagnostics.sh \
  outputs/phase4/DEAP/arousal/seed42/deap_arousal_fold01_full \
  --log logs/phase4/deap_arousal_fold01_full.log \
  --dry-run
```

终端给出本地快照目录，查看manifest.json及脱敏后的文本。不要把快照目录设为实验run内部；--bundle_dir如需指定，必须是空目录，不能覆盖已有文件。预览和正式上传会各自重新收集，运行期间文件可能变化。

正式上传只需一条命令：

```bash
cd /home/fss/NeuroFDAN
bash scripts/upload_run_diagnostics.sh \
  outputs/phase4/DEAP/arousal/seed42/deap_arousal_fold01_full \
  --log logs/phase4/deap_arousal_fold01_full.log
```

也支持传单个fold01目录。默认尝试run/train.log及logs/*/<run名>.log，不遍历全部服务器日志；无法定位时使用--log，可重复传多个.log。

成功时打印本次提交SHA、核验时远端SHA以及固定commit的目录链接。把该链接和manifest中的缺失/跳过项发给Codex，Codex即可从GitHub读取server-diagnostics对应目录；不需要服务器登录信息。上传后的结果目录例如：

```text
server-diagnostics 分支
  runs/2026-11-10_203000_DEAP_arousal_fold01_唯一ID/
    manifest.json
    run_summary.json             源run已有时收集
    upload_summary.json          根据现有产物重新生成的只读摘要
    upload_report.txt
    fold01/
      fold.json
      reference_sources.json
      epoch_metrics.jsonl
      mediator_diagnostics.jsonl
      collapse_diagnostics.jsonl
      epoch_confusion_matrices.npz
      ...其他白名单文件
    logs/train_编号.log           脱敏终端日志
```

首次成功上传时才会创建server-diagnostics分支；Codex本轮没有创建虚假服务器快照。可用下列命令核对远端（不会切换实验分支）：

```bash
cd /home/fss/NeuroFDAN
git branch --show-current
git ls-remote origin refs/heads/server-diagnostics
```

并发上传后远端可能已继续提交；工具会验证本次提交仍是远端祖先，显示核验时的远端SHA。本次commit链接保持固定，不会被下一次上传覆盖。

## 4. 白名单、限制与隐私边界

允许的固定文件：fold.json、summary.json、history.jsonl、reference_sources.json、epoch_metrics.jsonl、mediator_diagnostics.jsonl、collapse_diagnostics.jsonl、epoch_confusion_matrices.npz、run_summary.json、run_report.txt、config.json、probe_summary.json，以及明确匹配的.log。只扫描run本身与直接fold编号子目录；不递归搜集其他目录。

默认单文件8MiB、总包32MiB（含manifest）。参数--max_file_mib最多16、--max_total_mib最多64；超限文件跳过并写原因，不静默截断整个历史。NPZ检查压缩前大小、头部与规定计数形状，只接受epochs/train/test/complete整数/布尔数组并重新编码，不原样搬运任意ZIP内容。JSONL正在写入的末行可省略并标记；中间解析错误则跳过该文件。缺失文件明确列出，部分缺失不会阻止其他诊断上传。

config.json、probe_summary.json及fold内run_summary等可选文件缺失并不代表训练失败。单次快照的大小限制也不是整个结果分支的总大小限制；长期积累的历史目录会增加Git存储和获取成本。

不会扫描或上传.pt/.pth、last/best权重、Dataset、虚拟环境、缓存、原始信号、特征大数组、私钥或.env。用户显式指定--log也只接受.log文本，并执行同样大小和脱敏规则。

JSON按敏感键脱敏，文本屏蔽常见Token、认证信息、私钥块、密码赋值及本机绝对路径。自动规则不能保证发现任意自由文本中的自定义秘密；首次预览很重要，不要在训练日志写入凭证。上传结果的公开程度由GitHub仓库权限决定。

manifest分别保留：

- upload_checkout.tool_checkout：上传工具代码的SHA/分支/已跟踪改动标记。
- upload_checkout.experiment_checkout_at_upload：run所在Git工作区上传时的代码身份，可为null。
- recorded_run_sources：fold.json最近一次启动记录的source_version；旧Phase 4可能为null，不补造。
- 每个新epoch的source_version：可区分后续跨版本续训；旧记录不回填。
- uploaded/missing/skipped及脱敏后文件哈希、大小、末行省略标记。

只有已跟踪源码的改动标记，没有声称记录所有未跟踪代码。运行中快照不是事务；run_summary也不能代替CalculateResult --require_complete或权重推理。

## 5. 不干扰训练工作区的Git流程

上传工具：检查目录及无凭证origin → 只读获取源码身份 → 白名单快照 → 生成清单 → 新建独立临时Git仓库 → 获取server-diagnostics最新tip（首次为空分支）→ 追加唯一runs目录 → 显式文件列表提交 → 普通push → 远端SHA/祖先核验。

不切换实验分支、不暂存实验源码、不使用git add .、不改main或Phase 4。对清单内文件的git add -f只覆盖缓存忽略规则，不是强推历史；push从不使用force。远端并发更新时从最新tip重建仅含本快照的提交，最多重试3次；冲突/认证失败明确退出，不覆盖历史。

失败时本地脱敏快照保留，显示路径；原实验结果不移动。成功后自动临时快照默认清理；--keep_bundle可保留，自选--bundle_dir始终保留。没有自动上传cron、后台凭证采集或外部消息通知。

常见问题：

| 现象 | 处理 |
|---|---|
| Permission denied / Authentication failed | 检查服务器SSH agent/凭证助手与仓库写权限；不要把Token发到聊天。 |
| origin含用户名/Token或指向其他仓库 | 自行改为无明文凭证的正确地址，再用已有认证；工具不自动改origin。 |
| 文件超限 | 查看manifest.skipped；优先只传问题fold，保留原文件，不改权重或科学配置。 |
| 缺run_summary | 旧Phase 4正常；上传快照会生成upload_summary，缺项留在manifest。 |
| 文件仍在写入/尾行省略 | 稍后重复上传，产生新的唯一目录；不覆盖第一次证据。 |
| git作者未设置 | 配置自己的name/email后重试；可使用GitHub noreply邮箱。 |
| 推送后SHA不一致 | 看是否有并发上传及祖先核验；若工具报未确认成功，保留快照和错误，不声称已上传。 |

## 6. 新日志版本怎么运行

只有需要新的训练日志时才启动独立run；优先先上传已有结果，不为上传重跑100轮。若决定运行，保留原模型和参数，显式指定新的输出：

```bash
cd /home/fss/NeuroFDAN
mkdir -p logs/phase5
bash scripts/phase3_run_logged.sh logs/phase5/deap_arousal_fold01_full.log \
  python mainNeuroFDAN.py \
  --dataset DEAP --label arousal --root_dir /home/fss \
  --output_dir outputs/phase5 --run_name deap_arousal_fold01_full \
  --seed 42 --device cuda:0 --held_out_subjects 1 \
  --epochs 100 --batch_size 40 --workers 0 \
  --diagnostic_level full --log_detail_interval 10
```

每轮紧凑摘要含训练/测试两组CM，每10轮关键量摘要；完整机器诊断仍逐轮保存。训练结束后run目录有run_summary.json、run_report.txt。正式统计仍需原完整性审计，单fold不是完整LOSO，test-selected偏差仍存在。服务器最小探针及解释见[行动计划](phase5_action_plan.md)。
