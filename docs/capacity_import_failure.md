# 固定小批检查未启动：PyTorch导入错误的只读诊断

## 本次真实结果

固定提交a6e5bb2e5f77dfbe400af632e8a55ea0e5c0e339，runs/2026-11-11_124345_unknown_unknown_run_d31a7334。三个实际载荷大小及SHA256核验通过；uploaded仅包含logs/train_01.log、upload_summary.json、upload_report.txt，没有probe_summary.json。上传器missing是可选白名单启发式，未列缺失probe并不表示probe已存在；上传成功不是实验成功。

日志在check_fixed_batch_capacity.py第83行import torch失败，torchgen.model创建DispatchKey(Enum)时出现RuntimeError: dictionary changed size during iteration。尚未读取checkpoint、数据或建立临时模型/优化器，因此没有任何记忆能力检查结论，不能说新初始化或epoch20模型拟合失败。

这定位到Python/PyTorch/torchgen导入阶段，现有脱敏日志不能确定唯一原因。不能直接判定模型错误、某个Python版本必不兼容或PyTorch安装损坏，不盲目pip升级/重装，也不访问用户服务器。

## 本轮只收集环境导入证据

新增标准库脚本diagnose_torch_import.py，在当前解释器下分别运行普通和python -I子进程，各3次，捕获版本、enum/torch/torchgen来源、退出码及完整错误尾部。来源保留路径指纹及末级目录，便于绝对路径脱敏后仍能比较是否相同。无环境变量值、密码或Token输出。

-I仅影响该子进程，忽略PYTHONPATH、用户site和当前目录搜索，不禁用所有site定制。它不是环境修复，也不会安装或更改包。脚本不导入NeuroFDAN，不创建张量/模型或执行前向/反向，不查GPU健康；torch.version.cuda只是构建元数据。子进程每次最多60秒，超时明确记录，不当作成功。

## 完整服务器命令

保留当前dslenv和失败目录，不重建环境，不重新跑容量检查或主训练函数：

```bash
cd /home/fss/NeuroFDAN
export IMPORT_TOOL_DIR=/home/fss/NeuroFDAN-import-audit-tools
export IMPORT_RESULT_DIR=/home/fss/NeuroFDAN/outputs/phase5_probes/torch_import_audit_001
(
set -euo pipefail
git status --short
git worktree list
git fetch origin refs/heads/feat/neurofdan-metrics-and-import-audit:refs/remotes/origin/feat/neurofdan-metrics-and-import-audit
if [ ! -e "$IMPORT_TOOL_DIR" ]; then
    git worktree add --detach "$IMPORT_TOOL_DIR" origin/feat/neurofdan-metrics-and-import-audit
fi
test "$(git -C "$IMPORT_TOOL_DIR" rev-parse HEAD)" = "$(git rev-parse origin/feat/neurofdan-metrics-and-import-audit)" || {
    echo '版本不符，停止，不覆盖。'; exit 1;
}
test -z "$(git -C "$IMPORT_TOOL_DIR" status --porcelain)" || {
    echo '工具有修改，停止，不覆盖。'; exit 1;
}
mkdir -p outputs/phase5_probes
mkdir "$IMPORT_RESULT_DIR"
bash "$IMPORT_TOOL_DIR/scripts/phase3_run_logged.sh" "$IMPORT_RESULT_DIR/train.log" \
  python -u "$IMPORT_TOOL_DIR/scripts/diagnose_torch_import.py" --output_dir "$IMPORT_RESULT_DIR"
cat "$IMPORT_RESULT_DIR/train.log.exit"
)
```

probe_summary.json将分别记录all_normal_imports_ok和all_isolated_imports_ok。包装器退出0只代表收集完成，即使6次导入都失败也会生成真实诊断，不表示torch已修复。原训练/权重和容量检查失败记录不覆盖。

## 上传

```bash
cd /home/fss/NeuroFDAN
bash /home/fss/NeuroFDAN-import-audit-tools/scripts/upload_run_diagnostics.sh \
  /home/fss/NeuroFDAN/outputs/phase5_probes/torch_import_audit_001 \
  --log /home/fss/NeuroFDAN/outputs/phase5_probes/torch_import_audit_001/train.log --dry-run
```

查看manifest及脱敏内容后：

```bash
cd /home/fss/NeuroFDAN
bash /home/fss/NeuroFDAN-import-audit-tools/scripts/upload_run_diagnostics.sh \
  /home/fss/NeuroFDAN/outputs/phase5_probes/torch_import_audit_001 \
  --log /home/fss/NeuroFDAN/outputs/phase5_probes/torch_import_audit_001/train.log
```

回传新SHA/链接，原失败快照仍保留。不要反复执行同一个容量结果目录；待导入问题厘清后再用capacity_002等新目录。

## 下一步决策

| 普通导入 | 隔离导入 | 下一步 |
|---|---|---|
| 稳定成功 | 稳定成功 | 本次错误可能是瞬态，记录实际环境后重新运行有界容量检查，不宣称根因已确定 |
| 失败 | 成功 | 检查模块搜索路径、用户site/定制来源；先比较来源指纹与版本，不直接称为某个文件污染 |
| 失败 | 失败 | 依据真实解释器、包来源和完整异常定位安装/版本/定制问题，再给最小修复 |
| 交替成功失败 | 任意 | 先检查可重复性和导入过程干扰，不启动正式训练 |

如果隔离模式选择了不同版本/来源，不能隐去这一环境差异或直接当作与旧实验环境等价。没有证据前不升级torch、删环境或改模型/损失。导入恢复后，容量检查结果仍按已有预声明分支解读，不能从本次环境错误推断潜在中介学习失败。

## 实际执行状态

本轮读取的是GitHub已有失败日志并做静态源码审查。云端没有执行torch导入、CLI、模型、单元测试或训练。新增诊断尚待服务器执行；没有修改正式代码、SCM、损失或汇报主输出，没有PR或合并main。

实际通过：父脚本与WORKER子进程源码的AST解析、三个bash块的bash -n、git diff --check；正式模型/训练器/数据/损失/容量检查源码均无差异。静态通过不等于环境问题已复现或解决。
