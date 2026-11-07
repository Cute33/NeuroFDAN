# NeuroFDAN

用于 DEAP / HCI 情绪识别的训练与结果统计。

## 安装

建议使用 Python 3.10+ 和虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

GPU 训练请按照 [PyTorch 官方说明](https://pytorch.org/get-started/locally/)安装与本机 CUDA 驱动兼容的 PyTorch。依赖根据源码 import 列出，未锁定版本。

## 数据与路径

根目录优先级为：显式 `--root_dir` > 环境变量 `NEURO_ROOT` > `/home/fss`。
直接使用 `HCIDataset_Python` 或 `DEAPDataset_Python` 时，显式 `root_dir` 参数同样优先于环境变量。
代码可以放在任意目录，模块导入路径根据 `__file__` 定位。

请自行准备数据；仓库不包含数据集或预处理脚本。目录结构：

```text
$NEURO_ROOT/
├── Dataset/
│   ├── DEAP-python/    # 1.npy … 32.npy
│   └── HCI-python/     # 1.npy … 24.npy
└── NeuroFDAN/          # 混淆矩阵输出目录，运行前创建
```

尽管后缀为 `.npy`，数据加载器使用 `pickle.load` 读取包含 `data` 和 `labels` 的字典。
DEAP 的 `data` 应包含 40 个 trial、至少 37 个通道和 8064 个时间点（含 384 点基线）；
HCI 的 `data` 应包含 20 个 trial、至少 36 个通道和 3840 个时间点。
HCI 当前预处理实现需要支持 `.numpy()` 的数据（如 CPU Tensor）。
标签包含 valence（第 0 列）与 arousal（第 1 列）。

## 运行

在仓库目录执行：

```bash
export NEURO_ROOT=/path/to/root
mkdir -p "$NEURO_ROOT/NeuroFDAN"
python mainNeuroFDAN.py --dataset HCI --label arousal
# 或显式指定根目录（覆盖 NEURO_ROOT）
python mainNeuroFDAN.py --dataset DEAP --label valence --root_dir /path/to/root
python CalculateResult.py --dataset HCI --label arousal
tensorboard --logdir ./logs
```

`--device` 为 CUDA 设备编号，`--start_subject` 为起始测试被试。
保留原有默认训练配置：学习率 0.0001、100 epochs、batch size 40。
混淆矩阵保存在 `<root_dir>/NeuroFDAN/NeuroFDAN-<dataset>_<label>_cm.npy`；
checkpoint、`epoch.png` 和 `logs/` 保存在当前工作目录下。

## 当前代码限制

本次路径修改未改动训练逻辑：训练器初始化仍无条件调用 CUDA 设备名称查询，CPU 训练可能在此失败；
当前模型返回 7 项结果，而 DEAP 训练器只解包 3 项，DEAP 训练存在接口不匹配。
完整训练需要数据集、可用的 CUDA 环境，并另行处理上述原有问题。
