# Phase 3：Xshell/Linux 服务器操作指南

本指南所有服务器命令均在 `/home/fss/NeuroFDAN` 执行。Codex没有访问该服务器、真实数据或GPU，也没有执行正式训练。按“环境/文件检查 → 单fold smoke → 单fold完整训练 → 四项完整LOSO → 完整性审计与统计”的顺序操作；前一步无误再继续。

Phase 3基于Phase 2提交 `c66b4d3c1aa17453a0d024033b1c8ab85b67cdde`，不从旧main起步。核验时远端main其实已通过PR #3合并Phase 2；本阶段不修改main、不自动合并PR。模型、损失、数据预处理及选模规则保持Phase 2设置。

## 1. 登录后检查并拉取分支

Xshell登录服务器后：

```bash
cd /home/fss/NeuroFDAN
pwd
git status --short
git branch --show-current
git log -1 --oneline
git fetch origin
```

若有本地源码改动，先保存或提交到自己的分支，不执行reset/强制覆盖。工作区干净后，以下分支选择命令可兼容首次获取和已有本地分支：

```bash
cd /home/fss/NeuroFDAN
if git show-ref --verify --quiet refs/heads/feat/neurofdan-phase3; then
    git switch feat/neurofdan-phase3
else
    git switch --track -c feat/neurofdan-phase3 origin/feat/neurofdan-phase3
fi
git pull --ff-only origin feat/neurofdan-phase3
git branch --show-current
git log -1 --oneline
git rev-parse HEAD
git ls-remote origin refs/heads/feat/neurofdan-phase3
```

两处SHA应一致；如分支有本地提交导致不一致，先保留现场反馈，不自动重置。Phase 2稳定回退分支保留不变。

## 2. 先检查已有环境，再决定复用或新建

```bash
cd /home/fss/NeuroFDAN
command -v python || command -v python3
python --version
python -m pip --version
printf '已激活环境：%s\n' "${VIRTUAL_ENV:-未激活venv}"
ls -ld .venv venv .venv-phase3 2>/dev/null
command -v conda >/dev/null && conda env list
```

如果只有python3命令，先用 `python3 --version` 检查。项目源码使用Python 3.10+语法和PyTorch 2.0+接口；这不意味着服务器必须使用Codex云端的具体版本。

**已有可用环境：优先激活它**，例如仅选择实际存在的一项：

```bash
cd /home/fss/NeuroFDAN
source .venv/bin/activate
# 若实际使用其他venv/conda环境，请改成其真实激活方式，不重复新建或覆盖。
python --version
python -c "import sys; print('当前解释器：', sys.executable)"
```

**没有可用环境：**确认python3版本不低于3.10、`.venv-phase3`不存在后，才执行：

```bash
cd /home/fss/NeuroFDAN
python3 -m venv .venv-phase3
source .venv-phase3/bin/activate
python --version
python -m pip --version
```

如果系统python3过旧，请使用服务器已安装的合适解释器创建环境，例如实际存在的python3.10；不要盲目替换系统Python或CUDA。新venv目录被Git忽略。

### 检查PyTorch与GPU

```bash
cd /home/fss/NeuroFDAN
python - <<'PY'
import sys
print('Python：', sys.version)
try:
    import torch
except ImportError:
    print('当前解释器尚未安装PyTorch；先按下文选择安装方式。')
else:
    print('PyTorch：', torch.__version__)
    print('PyTorch编译CUDA版本：', torch.version.cuda)
    print('CUDA可用：', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU数量：', torch.cuda.device_count())
        print('GPU0：', torch.cuda.get_device_name(0))
PY
command -v nvidia-smi
nvidia-smi
```

`nvidia-smi`找不到或失败，先判断驱动工具是否在PATH、驱动是否工作、容器/账号是否可见GPU；不直接归因于模型。它显示的CUDA版本是驱动支持能力，不等于PyTorch wheel运行时或本地toolkit版本。若PyTorch的CUDA版本为None，一般是CPU构建；CUDA有版本但不可用，还需检查驱动兼容、设备权限或GPU可见性。不要盲目修改系统CUDA或LD_LIBRARY_PATH。

### 安装依赖，保留已验证的PyTorch

先在所选环境测试PyTorch/NumPy桥接；GPU可用时仅做一个小tensor运算：

```bash
cd /home/fss/NeuroFDAN
python - <<'PY'
import torch, numpy as np
print('NumPy：', np.__version__)
print('NumPy/Torch桥接：', torch.from_numpy(np.zeros(2, dtype=np.float32)))
if torch.cuda.is_available():
    print('GPU小tensor检查：', torch.ones(2, device='cuda:0').sum().item())
PY
```

如果现有PyTorch已工作，不要先卸载或升级它。先保存依赖快照并建立当前torch/NumPy版本约束，再安装项目requirements：

```bash
cd /home/fss/NeuroFDAN
mkdir -p logs/phase3
python -m pip freeze > logs/phase3/environment-before.txt
python - <<'PY'
from importlib.metadata import version
from pathlib import Path
entries = [f'{name}=={version(name)}' for name in ('torch', 'numpy')]
Path('logs/phase3/preserve-runtime.txt').write_text('\n'.join(entries) + '\n')
print('已记录当前torch和NumPy约束；安装冲突时停止排查，不自动覆盖。')
PY
python -m pip install -r requirements.txt -c logs/phase3/preserve-runtime.txt
python -m pip check
python -c "import torch, numpy, scipy; from torch.utils.tensorboard import SummaryWriter; print('核心依赖导入通过')"
```

上面的约束步骤要求torch和NumPy已安装。若现有torch版本低于项目要求或桥接本身失败，应另建兼容环境，不能强行用约束掩盖不兼容。

**全新环境尚无torch时：**GPU安装先依据服务器驱动、Python版本，在 [PyTorch官方安装选择器](https://pytorch.org/get-started/locally/)确认版本和wheel索引，不预设某个CUDA版本。可先在shell设置官方确认的索引，再执行（只适用于新环境）：

```bash
cd /home/fss/NeuroFDAN
# 先自行设置TORCH_INDEX_URL为官方确认的https://download.pytorch.org/whl/...地址。
python -m pip install 'torch>=2.0' --index-url "${TORCH_INDEX_URL:?请先按官方说明设置匹配驱动的索引地址}"
```

如果明确只需要CPU，新环境可改用：

```bash
cd /home/fss/NeuroFDAN
python -m pip install 'torch>=2.0' --index-url https://download.pytorch.org/whl/cpu
```

CPU/GPU安装二选一，不同时执行。全新环境的PyTorch安装不保证带有NumPy；先补充NumPy，再执行上面的桥接、快照、约束安装和pip check：

```bash
cd /home/fss/NeuroFDAN
# 仅用于新建环境；较旧PyTorch需先按其兼容说明选择NumPy版本。
python -m pip install numpy
```

较旧PyTorch与NumPy新主版本可能不兼容，应采用官方兼容组合，不盲目升级已有环境的所有软件。

## 3. 核验数据目录、文件名和内容

实际路径逻辑为 `root_dir/Dataset/<数据集>-python/<被试编号>.npy`，默认root_dir为NEURO_ROOT环境变量或/home/fss。本指南显式传/home/fss，避免激活环境后的变量改变路径。

```text
/home/fss/
├── NeuroFDAN/
└── Dataset/
    ├── DEAP-python/1.npy ... 32.npy
    └── HCI-python/1.npy ... 24.npy
```

```bash
cd /home/fss/NeuroFDAN
ls -lah /home/fss
ls -lah /home/fss/Dataset
ls -lah /home/fss/Dataset/DEAP-python
ls -lah /home/fss/Dataset/HCI-python
python - <<'PY'
from pathlib import Path
for dataset, count in [('DEAP', 32), ('HCI', 24)]:
    directory = Path('/home/fss/Dataset') / f'{dataset}-python'
    expected = {f'{i}.npy' for i in range(1, count + 1)}
    actual = {p.name for p in directory.glob('*.npy')}
    print(dataset, '预期文件数', count, '实际npy文件数', len(actual))
    print('缺失：', sorted(expected - actual))
    print('额外：', sorted(actual - expected))
PY
```

代码固定支持DEAP 1–32、HCI 1–24，不会根据较少文件数自动缩小LOSO。`s01.npy`等名称不符合当前接口，不要未经核实批量改名。别名MAHNOB_HCI仍读取HCI-python目录。

文件虽然叫.npy，内部实际是既有pickle字典，必须有data和labels；不是普通np.save数组。可在服务器上检查自己的既有实验文件各一个，不要求重新上传或转换：

```bash
cd /home/fss/NeuroFDAN
python - <<'PY'
import pickle
from pathlib import Path
import numpy as np
import torch
for dataset, trials, channels, points in [('DEAP', 40, 37, 8064), ('HCI', 20, 36, 3840)]:
    path = Path('/home/fss/Dataset') / f'{dataset}-python' / '1.npy'
    with path.open('rb') as stream:
        record = pickle.load(stream)
    raw, labels = record['data'], record['labels']
    raw = raw.detach().cpu().numpy() if torch.is_tensor(raw) else np.asarray(raw)
    labels = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
    print(dataset, 'data', raw.shape, raw.dtype, 'labels', labels.shape, labels.dtype)
    assert raw.ndim == 3 and raw.shape[0] == trials and raw.shape[1] >= channels and raw.shape[2] == points
    assert labels.ndim == 2 and labels.shape[0] == trials and labels.shape[1] >= 2
    assert np.isfinite(labels[:, :2]).all()
    print('单文件结构检查通过；其余文件仍需由加载器核验。')
PY
```

DEAP的8064点含384点基线，去掉后每trial得到10个6秒窗口；HCI文件应已截取30秒、3840点，得到5个窗口，读取器不自行决定原始长记录的截取位置。采样率128Hz，窗口768点。EEG为32通道；DEAP外周为2EMG+GSR，HCI为3ECG+GSR。labels第0列valence、第1列arousal，>5为1，否则0。HCI支持NumPy/CPU Tensor。错误格式先保留原文件并反馈形状/日志，不能直接假定需要重新转换。

## 4. 日志与退出状态

训练前确认CLI：

```bash
cd /home/fss/NeuroFDAN
python mainNeuroFDAN.py --help
python CalculateResult.py --help
mkdir -p logs/phase3
```

本阶段提供日志包装脚本，继承已激活的Python环境，使用pipefail和立即保存PIPESTATUS，同时记录Python与tee退出码：

```bash
cd /home/fss/NeuroFDAN
bash scripts/phase3_run_logged.sh logs/phase3/help.log python mainNeuroFDAN.py --help
cat logs/phase3/help.log.exit
```

日志追加而非清空；`.exit`记录最近一次调用。Python非零时脚本返回该退出码；Python成功但tee失败时返回tee退出码。不要只看tee的成功状态。后文每个任务均单独执行并检查退出码后再继续。

若手动使用tee，必须立即捕获管道状态：

```bash
cd /home/fss/NeuroFDAN
set -o pipefail
python mainNeuroFDAN.py --help 2>&1 | tee logs/phase3/manual-help.log
phase3_codes=("${PIPESTATUS[@]}")
printf 'Python退出码=%s，tee退出码=%s\n' "${phase3_codes[0]}" "${phase3_codes[1]}"
```

正式长任务可在服务器已有tmux会话中执行，避免关闭Xshell终止前台程序。若有tmux，用 `tmux new -s neurofdan-phase3`，在新会话重新进入项目并激活所选环境；Ctrl-b后按d脱离，重连后用 `tmux attach -t neurofdan-phase3`。未安装则先确认服务器允许的方式，不盲目安装系统工具。

## 5. 第一轮：DEAP arousal，第1折smoke

GPU版保留当前默认S/K/N_ref，仅限制batch数：

```bash
cd /home/fss/NeuroFDAN
bash scripts/phase3_run_logged.sh logs/phase3/deap_arousal_smoke.log \
  python mainNeuroFDAN.py \
  --dataset DEAP --label arousal --root_dir /home/fss \
  --output_dir outputs/phase3 --run_name deap_arousal_smoke \
  --seed 42 --device cuda:0 --held_out_subjects 1 \
  --epochs 1 --batch_size 40 --workers 0 \
  --max_train_batches 1 --max_test_batches 1
cat logs/phase3/deap_arousal_smoke.log.exit
```

无GPU时用独立run的轻量CPU版；这里只为接口检查显式减小采样，不修改正式实验默认值：

```bash
cd /home/fss/NeuroFDAN
bash scripts/phase3_run_logged.sh logs/phase3/deap_arousal_smoke_cpu.log \
  python mainNeuroFDAN.py \
  --dataset DEAP --label arousal --root_dir /home/fss \
  --output_dir outputs/phase3 --run_name deap_arousal_smoke_cpu \
  --seed 42 --device cpu --cpu_threads 1 --held_out_subjects 1 \
  --epochs 1 --batch_size 2 --workers 0 \
  --N_ref 8 --S_train 1 --K_train 2 --S_eval 1 --K_eval 2 \
  --max_train_batches 1 --max_test_batches 1
```

smoke仍读取该fold全部31名训练被试和1名测试被试，不是只读取一个数据文件。现有实现还需构建参考库和评估特征缓存；限制一个batch不等于零准备开销。

查看：

```bash
cd /home/fss/NeuroFDAN
tail -f logs/phase3/deap_arousal_smoke.log
```

正常应有来源检查通过、有限batch提示、有限loss、测试指标、last.pt和history记录；正常结束日志无Traceback/OOM，退出码0。best.pt可有可无，不能把未达门槛当作保存故障。`fold.json`和`summary.json`标记smoke=true，测试通常complete_test=false；统计审计将其排除，绝不能复制这些文件到正式run。

## 6. 第二轮：单fold完整训练

smoke检查正常后，使用新run_name，去掉两个max参数：

```bash
cd /home/fss/NeuroFDAN
bash scripts/phase3_run_logged.sh logs/phase3/deap_arousal_singlefold.log \
  python mainNeuroFDAN.py \
  --dataset DEAP --label arousal --root_dir /home/fss \
  --output_dir outputs/phase3 --run_name deap_arousal_singlefold \
  --seed 42 --device cuda:0 --held_out_subjects 1 \
  --epochs 100 --batch_size 40 --workers 0
```

观察事实CE、KL warmup、中介均值/方差/跨样本离散度、发放率、主输出概率、参考库样本数和测试窗口数。DEAP完整测试必须400窗口，HCI必须100窗口。若出现非有限loss/概率、源被试混入、错误窗口数、形状错误或OOM，先停止新实验并保存日志；不悄悄改S/K、数据划分或损失。

不要因smoke分数好坏决定正式模型。原规则下整折无best时，只保留last.pt，并标记“完成但无合格best”。

## 7. 第三轮：四项完整LOSO

以下四个独立任务逐项运行，不同时占用同一GPU启动四份任务；每项独立run_name。默认所有科学参数与Phase 2一致，seed固定42，无validation。

### DEAP arousal，32折

```bash
cd /home/fss/NeuroFDAN
bash scripts/phase3_run_logged.sh logs/phase3/deap_arousal_seed42.log \
  python mainNeuroFDAN.py \
  --dataset DEAP --label arousal --root_dir /home/fss \
  --output_dir outputs/phase3 --run_name deap_arousal_seed42 \
  --seed 42 --device cuda:0 --start_subject 1 --end_subject 32 \
  --epochs 100 --batch_size 40 --workers 0
```

### DEAP valence，32折

```bash
cd /home/fss/NeuroFDAN
bash scripts/phase3_run_logged.sh logs/phase3/deap_valence_seed42.log \
  python mainNeuroFDAN.py \
  --dataset DEAP --label valence --root_dir /home/fss \
  --output_dir outputs/phase3 --run_name deap_valence_seed42 \
  --seed 42 --device cuda:0 --start_subject 1 --end_subject 32 \
  --epochs 100 --batch_size 40 --workers 0
```

### HCI arousal，24折

```bash
cd /home/fss/NeuroFDAN
bash scripts/phase3_run_logged.sh logs/phase3/hci_arousal_seed42.log \
  python mainNeuroFDAN.py \
  --dataset HCI --label arousal --root_dir /home/fss \
  --output_dir outputs/phase3 --run_name hci_arousal_seed42 \
  --seed 42 --device cuda:0 --start_subject 1 --end_subject 24 \
  --epochs 100 --batch_size 40 --workers 0
```

### HCI valence，24折

```bash
cd /home/fss/NeuroFDAN
bash scripts/phase3_run_logged.sh logs/phase3/hci_valence_seed42.log \
  python mainNeuroFDAN.py \
  --dataset HCI --label valence --root_dir /home/fss \
  --output_dir outputs/phase3 --run_name hci_valence_seed42 \
  --seed 42 --device cuda:0 --start_subject 1 --end_subject 24 \
  --epochs 100 --batch_size 40 --workers 0
```

若资源不足，先记录nvidia-smi、配置和错误；K分块参数可用于控制峰值内存，但任何调整都需记入实验配置，不用更改科学参数掩盖问题。当前未实测服务器正式显存或运行时长。

## 8. 中断与续训

已存在fold.json时，新任务拒绝覆盖。不能从第1折重新执行整个旧run来“自动跳过”。例如正式DEAP arousal中断于第7折：

```bash
cd /home/fss/NeuroFDAN
bash scripts/phase3_run_logged.sh logs/phase3/deap_arousal_resume_fold07.log \
  python mainNeuroFDAN.py \
  --dataset DEAP --label arousal --root_dir /home/fss \
  --output_dir outputs/phase3 --run_name deap_arousal_seed42 \
  --seed 42 --device cuda:0 --held_out_subjects 7 \
  --epochs 100 --batch_size 40 --workers 0 \
  --resume outputs/phase3/DEAP/arousal/seed42/deap_arousal_seed42/fold07/last.pt
```

第7折完成后，不带resume继续第8–32折：

```bash
cd /home/fss/NeuroFDAN
bash scripts/phase3_run_logged.sh logs/phase3/deap_arousal_fold08_32.log \
  python mainNeuroFDAN.py \
  --dataset DEAP --label arousal --root_dir /home/fss \
  --output_dir outputs/phase3 --run_name deap_arousal_seed42 \
  --seed 42 --device cuda:0 --start_subject 8 --end_subject 32 \
  --epochs 100 --batch_size 40 --workers 0
```

必须同一run/fold、数据集/任务、模型、参考来源、平衡和smoke设置；原用过非默认参数需继续传相同配置。resume只支持单fold，计划epochs必须大于checkpoint已完成epoch。保持原100轮目标，不因续训把某折任意加到200轮；各折选模时长应一致。续训不是逐位等同未中断训练的承诺。

如果没有last.pt，不能假定可以resume；如果last.pt已到计划末轮但summary缺失，也不要用增加epochs绕过限制。保留现场反馈收尾/保存日志，不删除best或伪造summary。多文件产物不是单一事务，断电时需核验一致性。

## 9. 输出和完整性检查

以正式DEAP arousal为例：

```text
outputs/phase3/DEAP/arousal/seed42/deap_arousal_seed42/
  fold01/
    fold.json                 配置、来源、窗口数、smoke标识
    reference_sources.json    每个参考窗口的subject/trial/window/dataset
    history.jsonl             每epoch训练/测试指标、选模决定
    last.pt                   最近完整保存的训练状态
    best.pt                   仅满足原规则时存在
    best_predictions.json     最佳模型的每窗口概率/标签/来源
    best_confusion.npy        最佳2×2混淆矩阵
    summary.json              正常结束后的总结，有无合格best
    tensorboard/              训练记录
  ... fold32/
  NeuroFDAN-DEAP_arousal_cm.npy
```

最佳模型还按旧命名规则保存NeuroFDAN-数据集-任务-被试-正确数.pth。checkpoint是结构字典，不是裸state_dict。模型权重、数据与训练日志不提交GitHub。

快速检查已保存checkpoint（仅加载自己训练得到的文件，不运行模型）：

```bash
cd /home/fss/NeuroFDAN
python - <<'PY'
import torch
path = 'outputs/phase3/DEAP/arousal/seed42/deap_arousal_seed42/fold01/last.pt'
state = torch.load(path, map_location='cpu', weights_only=False)
print('epoch：', state['epoch'], '最佳：', state['best'])
print('模型配置：', state['frontdoor_config'])
print('参考来源报告：', state['reference_report'])
print('测试样本数：', state['metrics']['count'])
PY
```

smoke的last.pt路径请改成对应smoke目录。源码支持map_location，不代表已验证服务器GPU可用。

### 审计参考来源、概率和数值

下面的审计命令可在运行期间执行；此时“不完整”是预期，不会生成正式完整LOSO结论：

```bash
cd /home/fss/NeuroFDAN
python CalculateResult.py --dataset DEAP --label arousal --positive_class 1 \
  --run_dir outputs/phase3/DEAP/arousal/seed42/deap_arousal_seed42 \
  --export_dir outputs/phase3/tables/deap_arousal_progress
```

导出逐fold状态、来源检查、概率/混淆矩阵一致性和未完成项。程序不根据文件缺失伪造“失败原因”：

| 情况 | 判断方法 |
|---|---|
| 未运行 | 对应fold目录不存在 |
| 运行中或非正常结束 | 有配置/部分历史，但没有与计划轮数相符的结束summary；结合进程和日志判定 |
| 运行失败 | Python退出码非零或日志有Traceback；目录状态本身不足以证明失败 |
| 被中断 | Ctrl-C/连接中断等证据，或未正常收尾；若未写.exit，需查进程及终端日志 |
| 完成但无合格best | summary正常结束、has_eligible_best=false；只有last，不伪造best |
| 正常完成 | 计划轮数、完整测试、来源、最佳概率、矩阵和文件均一致 |
| smoke | 配置/summary带smoke，或有限batch参数非零；不进入正式统计 |

检查进程与日志：

```bash
cd /home/fss/NeuroFDAN
pgrep -af mainNeuroFDAN.py
cat logs/phase3/deap_arousal_seed42.log.exit
tail -n 100 logs/phase3/deap_arousal_seed42.log
```

没有进程且没有summary，不等于“未运行”；非零退出码可以确认失败，缺少.exit则保留“未完成/中断待查”。审计不会执行checkpoint推理，只核验文件存在及JSON/NumPy产物，不能据此保证权重或真实数据正确。

## 10. 四项正式统计与论文表导出

四个任务各自全部fold完成后执行。`--require_complete`要求所有预期被试都有正常结束且合格best、完整测试和一致产物；否则保存审计、返回非零，不把合格子集冒充完整LOSO。

```bash
cd /home/fss/NeuroFDAN
python CalculateResult.py --dataset DEAP --label arousal --positive_class 1 \
  --run_dir outputs/phase3/DEAP/arousal/seed42/deap_arousal_seed42 \
  --require_complete --export_dir outputs/phase3/tables/deap_arousal
```

```bash
cd /home/fss/NeuroFDAN
python CalculateResult.py --dataset DEAP --label valence --positive_class 1 \
  --run_dir outputs/phase3/DEAP/valence/seed42/deap_valence_seed42 \
  --require_complete --export_dir outputs/phase3/tables/deap_valence
```

```bash
cd /home/fss/NeuroFDAN
python CalculateResult.py --dataset HCI --label arousal --positive_class 0 \
  --run_dir outputs/phase3/HCI/arousal/seed42/hci_arousal_seed42 \
  --require_complete --export_dir outputs/phase3/tables/hci_arousal
```

```bash
cd /home/fss/NeuroFDAN
python CalculateResult.py --dataset HCI --label valence --positive_class 0 \
  --run_dir outputs/phase3/HCI/valence/seed42/hci_valence_seed42 \
  --require_complete --export_dir outputs/phase3/tables/hci_valence
```

导出audit.json、per_subject.csv、summary.csv。CSV带完整性/样本数/正类设置，汇总包含比例和百分数，不能忽略complete=false。若全部无合格best，不会制造均值。没有`--require_complete`时可供进度诊断，但只代表明确标识的合格子集。

矩阵行是真值、列是预测。按指定正类计算：ACC=(TP+TN)/N，Sensitivity=TP/(TP+FN)，Specificity=TN/(TN+FP)，F1=2TP/(2TP+FP+FN)，分母为零时按既有规则记0。先计算每个held-out被试的指标，再取被试均值和标准差；保留既有np.std的ddof=0，不是随机种子标准差。DEAP正类1、HCI正类0须在论文表注明，不能把不同正类F1不加说明直接比较。

正常完成但无合格best的fold也是重要实验状态；应如实报告，不能只挑有best的被试声称32/24折完整成功。原始--result_file参数保留用于旧结果查看，但它单独不能证明完整LOSO。

## 11. TensorBoard与Xshell端口转发

推荐只监听服务器环回地址，通过SSH隧道访问：

```bash
cd /home/fss/NeuroFDAN
tensorboard --logdir outputs/phase3 --host 127.0.0.1 --port 6006
```

Xshell会话属性 → SSH → 隧道/端口转发：新建“本地”转发，本地地址127.0.0.1、本地端口16006，目标主机127.0.0.1、目标端口6006。目标127.0.0.1指SSH服务器自身。重新连接该会话后，本地浏览器输入 `http://127.0.0.1:16006`。这是用户服务器隧道，不是Codex云端预览。

若明确使用服务器受控网络直接访问，可改为--host 0.0.0.0，但仍需服务器防火墙/访问配置，不默认要求开放端口。端口占用时换服务器端口并同步修改隧道目标端口。

## 12. 出错时反馈什么

```bash
cd /home/fss/NeuroFDAN
git branch --show-current
git rev-parse HEAD
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvidia-smi
tail -n 150 logs/phase3/deap_arousal_seed42.log
cat logs/phase3/deap_arousal_seed42.log.exit
```

反馈实际命令、分支/SHA、完整Traceback前后文、数据形状（不要发真实数据）、相关fold.json/summary或审计原因、错误退出码。OOM补充GPU占用及S/K/N_ref/batch/chunk。Codex据此做最小修复、推送Phase 3；服务器先保存现场，再git fetch/pull。不要在训练进程运行期间切换代码版本。

## 13. 科学边界与检查范围

保持无validation的train/test二分。ACC提升且>0.5保存，平ACC时F1严格上升保存；DEAP正类1/HCI正类0。固定seed42，不增加多种子实验。held-out test参与选模存在选择偏差，不能称独立无偏泛化；其他方法还需核验划分、预处理、选模和统计口径，不能仅凭相似策略声称公平。

真实数据没有真实M_c/M_a标签，也没有oracle干预概率监督。潜在中介、SCM和参考总体假设仍需证据；实现前门积分不自动证明真实中介被识别。no_cns2ans是估计器消融，不是生理物理干预。被试内归一化使用held-out自身无标签统计，不是严格在线未知被试协议。合成oracle数值精度不能等同真实因果识别精度。

云端仅完成编译/import/CLI、已有单元测试及小型接口/结果文件夹具测试。未核验服务器GPU、全部真实文件、正式显存、训练耗时、完整LOSO或论文效果。新增统计审计也不是这些验证的替代物。
