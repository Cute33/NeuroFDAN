# Phase 3 实际轻量检查记录

执行日期：2026-11-10。检查对象为从 `c66b4d3c1aa17453a0d024033b1c8ab85b67cdde` 创建的 `feat/neurofdan-phase3` 工作区。云端没有真实 DEAP/HCI 文件，没有 GPU；以下不能解释为服务器实测。

## 单元测试

在云端已有 Python 环境运行：

```text
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
Ran 57 tests in 30.215s
OK (skipped=1)
```

56 项通过，1 项 CUDA 测试因无 GPU 跳过。包含 Phase 1 和 Phase 2 已有测试及 Phase 3 新增的 10 项测试。已有套件包含小型 oracle 数值一致性检查；本轮未运行完整合成实验脚本或任何真实数据训练。

新增测试覆盖：

- 全部预期被试的状态审计与 CSV/JSON 导出，规格与真实加载器常量一致。
- 未运行、未完成、完成但无合格 best、smoke、来源泄漏及配置不一致的区别。
- 续训中旧 summary 不能冒充新计划已完成。
- 正式完整性验收返回非零，导出带不完整标识的新报告，防止旧完整统计残留。
- 非有限/越界/不归一化概率报错，合法概率保持原张量和梯度。
- 日志包装脚本的追加行为、带空格路径、Python 失败及 tee 失败退出状态。
- 指南训练和统计命令由真实 argparse 接口解析；不执行这些训练命令。

结果审计测试使用临时 JSON/NumPy 产物夹具，`.pt` 仅为文件存在性夹具，不是真实 checkpoint。审计器不会加载这些文件；真实 checkpoint 的保存/加载由既有 CPU 接口测试覆盖。不得把产物夹具解释为真实数据实验。

## 静态与接口检查

- 仓库 Python 文件语法解析、编译。
- 八个核心模块导入：NeuroFDAN、frontdoor_scm、Datasets、ModelTrain、mainNeuroFDAN、CalculateResult、loss、result_audit。
- 两个入口的 `--help` 返回 0。
- 服务器指南全部 Bash 代码块通过 `bash -n`；仅检查语法，不在云端执行服务器操作。
- 日志脚本 `bash -n`、`git diff --check`。
- 对照 Phase 2 基线，NeuroFDAN.py、frontdoor_scm.py、Datasets.py、mainNeuroFDAN.py、loss.py、requirements.txt 内容保持不变。

原始终端检查输出保留在云端临时位置，不作为训练日志提交仓库。最终版本和远端 SHA 以交付时实际 Git 核验为准。

## 未执行

服务器 Python/PyTorch/CUDA/GPU 兼容检查、真实文件读取、真实数据 smoke、正式显存与耗时测试、单折完整训练、完整 LOSO、论文效果及服务器续训均未执行。按 [服务器指南](phase3_server_guide.md) 在用户服务器逐步验证。
