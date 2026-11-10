# NeuroFDAN

当前开发版本把 Phase 1 前门 SCM 核心接入真实生理骨干和 LOSO train/test 流程。代码版本以 Git 分支为准：第一阶段备份为 `feat/neurofdan-phase1`，第二阶段为 `feat/neurofdan-phase2`；未合并 main。

- [第二阶段实现与限制](docs/phase2_integration_report.md)
- [服务器环境、数据格式、单fold检查和完整LOSO命令](docs/phase2_server_guide.md)
- [第一阶段历史合成验证报告](docs/phase1_implementation_report.md)

## 安装与运行

建议 Python 3.10+，PyTorch 2.0+。GPU 用户先按官方说明安装匹配 CUDA 驱动的 PyTorch。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export NEURO_ROOT=/path/to/data_root
python mainNeuroFDAN.py --help

# 在自己的实验服务器运行；云端没有执行正式数据训练
python mainNeuroFDAN.py --dataset DEAP --label arousal --seed 42 \
  --root_dir "$NEURO_ROOT" --run_name full42 --device cuda:0
python mainNeuroFDAN.py --dataset HCI --label valence --seed 42 \
  --root_dir "$NEURO_ROOT" --run_name full42 --device cuda:0
```

数据仍为 `Dataset/DEAP-python/1.npy…32.npy` 和 `Dataset/HCI-python/1.npy…24.npy` 的既有 pickle 字典。标签阈值5，窗口6秒；数据、checkpoint和临时产物不提交。

默认完整LOSO。用 `--held_out_subjects 1` 运行单折；用 `--max_train_batches 1 --max_test_batches 1` 执行有限batch接口检查，不能当作正式结果。输出默认在 `outputs/phase2`，用独立 `--run_name` 避免覆盖已有fold。

没有独立validation，保留 held-out test performance 选择 best checkpoint 的旧协议并明确其选择偏差。参考库只来自当前fold未平衡的训练窗口。真实训练使用事实CE、KL及明确的发放率约束，不使用合成oracle监督，也不把公式实现解释为真实潜在中介已获无条件识别。
