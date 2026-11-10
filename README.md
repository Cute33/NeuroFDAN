# NeuroFDAN

当前开发版本把 Phase 1 前门 SCM 核心接入真实生理骨干和 LOSO train/test 流程。第一阶段备份为 `feat/neurofdan-phase1`，第二阶段稳定版本为 `feat/neurofdan-phase2`。第三阶段 `feat/neurofdan-phase3` 从第二阶段核验提交独立创建，提供服务器操作指南与结果完整性审计。核验时第二阶段已由 PR #3 合并 main；本轮不修改 main。

第四阶段 `feat/neurofdan-phase4` 从 Phase 3 的 `cd844c85d28f2335acf4d41083184d8ad98e5bbd` 创建，仅增强训练诊断。默认每轮完整评估优化训练子集与测试集，打印混淆矩阵；`--diagnostic_level full` 增加中介和裁剪前梯度统计。模型、损失、数据及测试选模协议保持不变。

- [第四阶段：诊断依据、输出路径与科学边界](docs/phase4_diagnostic_analysis.md)
- [第四阶段：保留旧结果、服务器第1折诊断与产物读取](docs/phase4_server_guide.md)
- [第四阶段：完整轻量测试结果](docs/phase4_test_results.md)
- [第三阶段：完整 Xshell/Linux 服务器操作指南](docs/phase3_server_guide.md)
- [第三阶段：修改、版本关系与交付报告](docs/phase3_delivery_report.md)
- [第三阶段：实际轻量检查记录](docs/phase3_checks.md)
- [第二阶段实现与限制](docs/phase2_integration_report.md)
- [服务器环境、数据格式、单fold检查和完整LOSO命令](docs/phase2_server_guide.md)
- [第一阶段历史合成验证报告](docs/phase1_implementation_report.md)

## 安装与运行

建议 Python 3.10+，PyTorch 2.0+。先按第三阶段指南检查并优先复用服务器已有环境；安装其余依赖时保留已验证的 PyTorch/NumPy 版本。无可用环境时才新建环境，GPU 版本依据实际驱动和官方安装说明选择。

```bash
cd /home/fss/NeuroFDAN
python mainNeuroFDAN.py --help
python CalculateResult.py --help
```

数据仍为 `Dataset/DEAP-python/1.npy…32.npy` 和 `Dataset/HCI-python/1.npy…24.npy` 的既有 pickle 字典。标签阈值5，窗口6秒；数据、checkpoint和临时产物不提交。

默认完整LOSO。用 `--held_out_subjects 1` 运行单折；用 `--max_train_batches 1 --max_test_batches 1` 执行有限batch接口检查，不能当作正式结果。输出默认在 `outputs/phase2`，用独立 `--run_name` 避免覆盖已有fold。

没有独立validation，保留 held-out test performance 选择 best checkpoint 的旧协议并明确其选择偏差。参考库只来自当前fold未平衡的训练窗口。真实训练使用事实CE、KL及明确的发放率约束，不使用合成oracle监督，也不把公式实现解释为真实潜在中介已获无条件识别。
