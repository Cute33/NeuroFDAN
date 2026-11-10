# 实验服务器运行指南

这些命令针对第二阶段源码；需要用户自己的真实数据。云端未执行正式训练。建议Linux、Python 3.10及以上、PyTorch 2.0及以上；云端接口检查实际使用Python 3.12和PyTorch CPU版本，服务器CUDA兼容性待核验。

## 获取源码和依赖

先保存服务器本地改动，再进入既有检出目录：

```bash
git status --short
git fetch origin
git switch feat/neurofdan-phase2
git pull --ff-only origin feat/neurofdan-phase2
git log -1 --oneline
python -m venv .venv
source .venv/bin/activate
# GPU服务器先按PyTorch官方安装说明选择匹配驱动的torch，再安装其余依赖。
python -m pip install -r requirements.txt
python mainNeuroFDAN.py --help
```

没有检出目录时可先git clone https://github.com/Cute33/NeuroFDAN.git。Phase 1备份在feat/neurofdan-phase1，第二阶段不合并main。

## 数据路径

```bash
export NEURO_ROOT=/path/to/data_root
```

目录结构必须为：

```text
data_root/
  Dataset/
    DEAP-python/1.npy ... 32.npy
    HCI-python/1.npy ... 24.npy
```

可用--root_dir覆盖NEURO_ROOT，未配置时回退/home/fss。文件仍是pickle字典，不是普通np.save数组。DEAP data形状为[40,至少37,8064]，含3秒基线；HCI为[20,至少36,3840]，已截取30秒；labels第0/1列为valence/arousal。HCI支持NumPy或CPU Tensor。异常形状、缺失文件或非有限值会报错，不能用接口检查代替真实文件核验。

## 单fold服务器接口检查

仍读取该fold所有非held-out训练被试，未改变LOSO；只运行一个训练batch、一个测试batch：

```bash
python mainNeuroFDAN.py --dataset HCI --task Valence --seed 42 \
  --root_dir "$NEURO_ROOT" --output_dir outputs/phase2 --run_name smoke \
  --held_out_subjects 1 --epochs 1 --batch_size 2 --device cpu --cpu_threads 1 \
  --N_ref 8 --S_train 1 --K_train 2 --S_eval 1 --K_eval 2 \
  --max_train_batches 1 --max_test_batches 1
```

此输出标记smoke=true，禁止作为正式结果。正式训练另设run_name，不复用smoke的best指标。数据不足以按原整trial平衡规则下采样时会报错；如决定关闭平衡，必须显式--no_balance并在实验记录中说明。

## 完整LOSO与单fold正式训练

```bash
# DEAP arousal，完整32折
python mainNeuroFDAN.py --dataset DEAP --label arousal --seed 42 \
  --root_dir "$NEURO_ROOT" --output_dir outputs/phase2 --run_name full42 --device cuda:0

# HCI valence，完整24折；MAHNOB_HCI是同义数据集名
python mainNeuroFDAN.py --dataset MAHNOB_HCI --task Valence --seed 42 \
  --root_dir "$NEURO_ROOT" --output_dir outputs/phase2 --run_name full42 --device cuda:0

# DEAP valence，仅第1个held-out，训练仍用另外31人
python mainNeuroFDAN.py --dataset DEAP --label valence --seed 42 \
  --root_dir "$NEURO_ROOT" --output_dir outputs/phase2 --run_name full42 \
  --held_out_subjects 1 --device cuda:0
```

默认epochs=100、batch_size=40、lr=0.0001。可显式覆盖。--start_subject和--end_subject设置折范围，--held_out_subjects覆盖范围。无独立validation，保留测试集参与best选择的协议；需要如实报告选择偏差。

内存资源不足时先调整--reference_chunk_size（默认4），再明确记录batch或S/K/N_ref调整。--ref_dim 64是额外压缩消融，不是默认严格充分表示；不能只因能运行就宣称精确识别。未在云端测量正式GPU显存。

新run若已有fold.json会拒绝覆盖。继续同一fold的未中断配置可：

```bash
python mainNeuroFDAN.py --dataset DEAP --label arousal --seed 42 \
  --root_dir "$NEURO_ROOT" --output_dir outputs/phase2 --run_name full42 \
  --held_out_subjects 1 --epochs 200 --device cuda:0 \
  --resume outputs/phase2/DEAP/arousal/seed42/full42/fold01/last.pt
```

其他模型/来源/预测设置需与原checkpoint一致；如之前使用非默认参数，继续传入。CPU/CUDA可通过map_location切换，不承诺跨设备或中断续训逐位相同。新checkpoint不是旧裸state_dict。

## 消融入口

```bash
# 没有CNS→ANS条件电流；是真实估计器消融，不改变生成机制
python mainNeuroFDAN.py --dataset HCI --label arousal --seed 42 \
  --root_dir "$NEURO_ROOT" --run_name no_cns42 --no_cns2ans --device cuda:0

# Phase 1 none：同一中介/结果模型，一次事实路径，不执行K参考边缘化
python mainNeuroFDAN.py --dataset HCI --label arousal --seed 42 \
  --root_dir "$NEURO_ROOT" --run_name none42 --fd_mode none --device cuda:0
```

所有真实训练只用事实标签监督观察路径。不存在真实oracle中介标签或干预概率输入。

## 检查参考来源和结果

每fold查看fold.json和reference_sources.json：held_out_subject不得出现在任何参考subject_id；参考subject必须属于train_subjects；population_windows是未平衡总体，optimization_windows可能较少；shortfall明确容量不足，dataset/trial/window均保存。

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path('outputs/phase2/DEAP/arousal/seed42/full42')
for folder in sorted(root.glob('fold*')):
    config = json.loads((folder / 'fold.json').read_text())
    sources = json.loads((folder / 'reference_sources.json').read_text())
    subjects = {row['subject_id'] for row in sources}
    assert config['held_out_subject'] not in subjects
    assert subjects <= set(config['train_subjects'])
    assert all(row['dataset'] == config['dataset'] for row in sources)
    print(folder.name, '来源检查通过', len(sources))
PY

python CalculateResult.py --dataset DEAP --label arousal --positive_class 1 \
  --result_file outputs/phase2/DEAP/arousal/seed42/full42/NeuroFDAN-DEAP_arousal_cm.npy

tensorboard --logdir outputs/phase2
```

HCI使用相应HCI/valence路径和--positive_class 0。统计脚本默认正类0保留旧含义；DEAP选模的F1正类为1，务必明确区分。被试间标准差不是五次随机种子标准差。

best.pt只在原ACC>0.5/平ACC更高F1规则满足时产生。没有合格best的fold只有last.pt，汇总不能把未完成/无合格best的全零行当有效结果；要检查全部32/24个summary.json，不能仅挑有结果的折声称完整LOSO。

## 轻量级测试

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
```

包含Phase 1既有数值积分单元测试及第二阶段接口夹具，不运行完整合成训练或真实LOSO。完整实验与GPU兼容性仍须在服务器确认。
