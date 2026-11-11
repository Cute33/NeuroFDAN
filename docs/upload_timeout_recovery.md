# Git上传超时：定位、重试与推送状态核验

## 已有证据及边界

用户贴出的记录显示：dry-run成功；多次正式上传均在生成本地快照后报通用超时；curl访问GitHub返回200；SSH认证成功；原仓库git ls-remote约5.94秒完成。origin为既有SSH地址，SSH配置指定专用密钥且严格检查主机密钥。

这些支持“某些Git连接可以完成”，不能证明临时上传工作区的fetch、写权限或push成功。旧脚本在生成快照后会创建全新的临时Git仓库，查询远端、fetch server-diagnostics最新整棵树、checkout、commit、push及核验；每条均固定60秒，错误未标出子命令。浅fetch虽只取一层提交，仍需取该提交引用的文件内容，慢速链路可超过60秒。当前诊断树有182个文件，逻辑文件大小合计11,131,660字节；这不是压缩后的实际传输量，不能据此断言耗时。fetch是候选卡点，不是已实测确定的根因；本地签名/其他操作也需根据新日志判定。

云端核验时远端server-diagnostics仍为cfcdbdff5696ed0061a5ef410dca5c1c6820da70，与上传前一致；未见此次汇总的新远端提交。不能把旧工具的每次失败都一概解释成“绝未推送”，因为push接收后也可能在确认阶段超时。

原实验、两份支持审计及已有总结均有效。不重跑模型，不更改Git origin、密钥或全局Git/SSH配置，不禁用主机密钥检查。结果整理只是复核记录，不是继续研究所必需的新实验。

## 最小修复

- 每条Git命令打印子命令、超时上限、退出码和耗时，不打印完整参数或环境。
- 新增--git_timeout，网络命令ls-remote/fetch/push默认300秒，可设30–3600秒；本地命令仍60秒。一次600秒上限不是要求总上传恰好600秒，多条命令各有独立上限。
- 默认SSH使用BatchMode、15秒连接上限和连接保活，继续读取既有SSH配置；显式GIT_SSH_COMMAND原样保留。
- 超时错误标明子命令，部分stderr脱敏。push超时明确标为结果不确定，不自动重推。
- 本地快照新增upload_state.json收据，push前保存目标目录与候选SHA，核验成功后更新状态。收据不加入上传白名单；成功仍只在远端核验后报告。
- 原白名单、脱敏、server-diagnostics目标、唯一目录、非强推及并发重试规则保留。

## 服务器获取新工具

在已有dslenv中执行。原support-review-tools不改，新建独立工具worktree。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
git status --short
git worktree list
timeout 600 git fetch origin \
  refs/heads/fix/neurofdan-upload-timeout:refs/remotes/origin/fix/neurofdan-upload-timeout
UPLOAD_TOOLS=/home/fss/NeuroFDAN-upload-timeout-tools
if [ ! -e "$UPLOAD_TOOLS" ]; then
    git worktree add --detach "$UPLOAD_TOOLS" origin/fix/neurofdan-upload-timeout
fi
test "$(git -C "$UPLOAD_TOOLS" rev-parse HEAD)" = "$(git rev-parse origin/fix/neurofdan-upload-timeout)"
test -z "$(git -C "$UPLOAD_TOOLS" status --porcelain)"
cd "$UPLOAD_TOOLS"
python -m unittest discover -s tests -p test_upload_transport.py -v
)
```

测试全部模拟子进程/网络，只用标准库，不加载模型。云端按现有用户约束不运行这些单元测试或CLI；本轮仅完成静态AST、Shell语法及diff检查。服务器需先执行此命令核验新工具。

## 预览与正式上传

预览沿用既有结果和原train.log，不运行任何主训练或探针脚本。

```bash
cd /home/fss/NeuroFDAN
bash /home/fss/NeuroFDAN-upload-timeout-tools/scripts/upload_run_diagnostics.sh \
  /home/fss/NeuroFDAN/outputs/phase5_probes/support_results_review_001 \
  --log /home/fss/NeuroFDAN/outputs/phase5_probes/support_results_review_001/train.log \
  --git_timeout 600 --dry-run
```

正式上传保存独立日志与持久快照；不会覆盖任何旧run。若新快照目录已存在，停止，避免不加核验地重复提交。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
UPLOAD_TOOLS=/home/fss/NeuroFDAN-upload-timeout-tools
UPLOAD_BUNDLE=/home/fss/NeuroFDAN/outputs/diagnostic_upload_bundles/support_review_retry_001
UPLOAD_LOG=/home/fss/NeuroFDAN/outputs/phase5_probes/support_review_upload_retry_001.log
test ! -e "$UPLOAD_BUNDLE"
mkdir -p /home/fss/NeuroFDAN/outputs/diagnostic_upload_bundles
bash "$UPLOAD_TOOLS/scripts/phase3_run_logged.sh" "$UPLOAD_LOG" \
  bash "$UPLOAD_TOOLS/scripts/upload_run_diagnostics.sh" \
  /home/fss/NeuroFDAN/outputs/phase5_probes/support_results_review_001 \
  --log /home/fss/NeuroFDAN/outputs/phase5_probes/support_results_review_001/train.log \
  --git_timeout 600 --bundle_dir "$UPLOAD_BUNDLE"
cat "$UPLOAD_LOG.exit"
)
```

成功会打印核验SHA/结果链接。失败时保留新终端输出，尤其最后一条“[Git]开始”及超时子命令；不要重新跑模型。可查看：

```bash
cd /home/fss/NeuroFDAN
tail -n 100 outputs/phase5_probes/support_review_upload_retry_001.log
cat outputs/phase5_probes/support_review_upload_retry_001.log.exit
if [ -f outputs/diagnostic_upload_bundles/support_review_retry_001/upload_state.json ]; then
    cat outputs/diagnostic_upload_bundles/support_review_retry_001/upload_state.json
fi
```

传回日志和本地收据即可，不要发送私钥、密码、Token、完整环境或SSH调试原始转储。若Git网络仍受阻，可直接在聊天附上小型probe_summary.json和脱敏日志；该汇总不含模型权重，上传故障不需要通过重训解决。

## 结果不确定时先查询远端目录

仅在生成了本地收据后使用下面的只读核验。它只fetch诊断分支并检查目标manifest，不修改原工作区，不重新push。目录存在只表示远端有该记录，还需核对manifest与本地快照，不能只凭目录名确认本次载荷。

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
timeout 600 git fetch origin server-diagnostics
python - <<'PY'
import hashlib
import json
import pathlib
import subprocess
bundle = pathlib.Path('/home/fss/NeuroFDAN/outputs/diagnostic_upload_bundles/support_review_retry_001')
state = json.loads((bundle/'upload_state.json').read_text())
target = 'runs/' + state['run_id']
def read(name):
    return subprocess.check_output(['git', 'show', 'FETCH_HEAD:'+target+'/'+name])
remote_manifest = read('manifest.json')
if remote_manifest != (bundle/'manifest.json').read_bytes():
    raise RuntimeError('远端manifest与本地不同，不能认定核验成功')
for item in json.loads(remote_manifest)['uploaded']:
    raw = read(item['file'])
    if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
        raise RuntimeError('远端载荷完整性不匹配')
remote_sha = subprocess.check_output(['git','rev-parse','FETCH_HEAD'], text=True).strip()
print('远端目录及全部载荷已核验；本次快照已存在，不再重复上传。')
print('远端SHA：', remote_sha)
print('https://github.com/Cute33/NeuroFDAN/tree/'+remote_sha+'/'+target)
PY
)
```

若查询不到目标路径，当前fetch到的远端版本尚无该快照；结合收据阶段和Git错误再决定重试。候选SHA为null且停留在fetch阶段，说明尚未创建该候选提交；若已有候选SHA或处于push/核验阶段，则先保留并核对。不要使用force push、reset或清空原实验目录。
