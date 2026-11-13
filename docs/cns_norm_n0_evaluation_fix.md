# N0独立测试视图缺少中介：最小修复与重跑

失败结果提交6de184c0f74dedd11ff9510391bf9dc8ca5f4023，目录runs/2026-11-13_124023_unknown_unknown_run_b920ab8c。已逐项校验载荷长度/SHA256，上传成功；服务器209项测试通过（skipped=1），但N0在第一组独立评价读取t['m_c']时退出1。probe_summary.completed=false、rows为空，不能判断归一化性能。

根因是cns_norm_synthetic.main把训练的观察字段视图view(test)复用于独立评价；该视图按设计只含Xc/Xa/Y，而中介恢复评分需要测试集真实Mc/Ma。生成数据本身含真实中介，没有DEAP/HCI数据接口或模型结构错误的证据。

最小修复：新增evaluation_view(raw)，只给独立测试阶段提供Xc/Xa/Y/Mc/Ma，main评价使用此视图。两个观察训练组仍用view(raw,False)，fit仍拒绝含中介的观察训练字典；oracle组仍是唯一允许真实中介训练的分支。没有将测试真值、U或do概率接入观察训练，没有改变归一化、损失、超参数、初始化和参考积分。

新增2项服务器测试：评价/训练视图权限检查；实际调用完整N0入口，每组三组各1步，执行预测、条件中介恢复、干预误差、初始化一致性和反例输出。测试还记录fit实际接收的字段，确认两个观察组无中介。旧209项测试没有覆盖完整评价入口，是此前漏检的原因。新测试未在云端执行；云端只AST和Git静态检查。

原001目录保留。失败发生在第一组训练后、结果行保存前，且N0未保存该模型权重；无法仅凭GitHub快照补完评价。因此重跑同配置小规模CPU N0，使用002新目录；不重跑任何DEAP/HCI训练，不读取或覆盖旧checkpoint。本次测试通过后仍是每组200步，ACC/F1作为主要分类输出，其他统计和do误差保留；不自动进入真实数据或LOSO。

## 一次执行测试、N0与自动上传

```bash
cd /home/fss/NeuroFDAN
conda activate dslenv
(
set -euo pipefail
if [ -z "${GIT_SSH_COMMAND:-}" ]; then
    export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
fi
git status --short
timeout 600 git fetch --no-tags origin fix/neurofdan-cns-n0-evaluation-view
N0_FIX_SHA="$(git rev-parse FETCH_HEAD)"
N0_FIX_TOOLS=/home/fss/NeuroFDAN-cns-n0-fix-tools
if [ ! -e "$N0_FIX_TOOLS" ]; then git worktree add --detach "$N0_FIX_TOOLS" "$N0_FIX_SHA"; fi
test "$(git -C "$N0_FIX_TOOLS" rev-parse HEAD)" = "$N0_FIX_SHA"
test -z "$(git -C "$N0_FIX_TOOLS" status --porcelain)"
git -C "$N0_FIX_TOOLS" log -1 --oneline
bash "$N0_FIX_TOOLS/scripts/run_cns_norm_n0_and_upload.sh" \
    /home/fss/NeuroFDAN/outputs/phase6/cns_input_norm/n0_gaussian_002
)
```

无需另行运行mainNeuroFDAN.py。runner先CPU全量轻量测试，再调用完整命令`python -u <新工具目录>/scripts/cns_norm_synthetic.py --output_dir /home/fss/NeuroFDAN/outputs/phase6/cns_input_norm/n0_gaussian_002 --seed 42 --steps 200`；它的device固定CPU、batch64、三组配对，合成配置与001相同，不编造真实数据参数。

上传源是n0_gaussian_002；日志位于n0_gaussian_002_logs，自动附带tests.log、train.log及.exit。先dry-run，再用600秒Git超时、16MiB单文件/64MiB总量正式上传，追加独立server-diagnostics/runs目录。唯一bundle在n0_gaussian_002_bundle_日期_PID目录，保留upload_state.json。无需单独指定日志、重复上传001或手工摘抄统计；成功回传最后SHA和链接，失败回传终端输出。

独立修复分支只修改N0脚本及测试、新增本文。核心模型、CLI、数据和训练损失不变；没有main修改、PR或合并，没有在云端运行模型、测试或训练。
