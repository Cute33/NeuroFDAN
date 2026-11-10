# E2服务器结果分析与W1耦合L2单变量短程对照

本报告基于服务器实测产物，不是云端重跑。E2代码提交2c289669cd41c5b87469876a41b83a516b8168a5；上传提交aa97f71f75e95384be7ed964337ca48f1d313116，目录runs/2026-11-10_223448_unknown_unknown_run_6d18da47。后续代码独立分支feat/neurofdan-weight-decay-control，不修改历史分支或main。

## 1. 完整性、兼容性与执行边界

manifest所列四个载荷的字节数和SHA256均核验一致，skipped为空，日志无Traceback。缺少的是纯探针不需要的训练摘要/config文件；unknown命名源于没有训练fold.json，不是失败。

E2使用训练被试2的16个窗口，两类各8个。best实际epoch1、Adam step260；last实际epoch37、step9620。两份checkpoint哈希与E1b一致，参数/BN buffer和文件均记录不变，未创建优化器或执行step。原Phase4是用户主动在epoch37终止的快照，不应补跑100轮。

模式为train_frozen_bn，S=4，保留训练态Gumbel/dropout及Gaussian采样；fd_mode=none只跳过本来不参与真实监督目标的参考积分。它是局部梯度探针，不是重训消融或完整数据性能评估。单个均衡批次不代表历史所有批次，probe取自训练总体但不保证每个窗口都在当时平衡后的优化子集中。

## 2. 主要结果

下表分子是原Adam数学耦合L2方向lambda*theta范数；分母是本批加权事实CE梯度范数。两次总梯度范数分别0.609529、0.041313，裁剪系数均为1，因此本批没有因max_norm=1缩小梯度。

| 模块 | epoch1 L2/CE | epoch37 L2/CE |
|---|---:|---:|
| 图参数 | 12.05 | 449628.10 |
| CNS骨干 | 1.386 | 591.97 |
| ANS骨干 | 2.521 | 760.94 |
| CNS中介（不含ANS） | 206.28 | 475.16 |
| ANS中介（不含CNS电流投影） | 4.357 | 0.216 |
| CNS→ANS电流投影 | 5.228 | 0.500 |
| OutcomeHead | 0.0596 | 0.0222 |

模块总范数仍会掩盖早期核失效。epoch1的六个卷积核L2/CE约为CNS 8.73/3.50/2.07、ANS 2.26/2.66/2.84。CNS encoder.2.weight的CE为1.7299e-8，L2为0.08221，约475万倍；但encoder.0.weight的比值仅0.692，不能说每个参数都由衰减主导。

epoch37第一层卷积核CE范数CNS为8.7321e-37、ANS为5.9241e-34；第三层卷积bias和末层BN bias仍有梯度，解释了为什么“整个骨干梯度非零”不等于前层还能有效学习。两支第三层卷积在这次新训练被试窗口上也逐元素跨样本相同，支持E1b观察。

epoch1的CNS→ANS CE梯度范数0.009070，epoch37为3.6412e-5，说明本批该参数路径并未被完全detach。它不能证明CNS生理输入仍被保留，也不能把随机Mc经此路径产生的变化当作输入信息。

epoch1：CE=0.716212，加权CNS KL=2.3409e-6，加权ANS KL=7.8360e-5；CNS中介对应梯度CE=4.8522e-4、加权KLc=2.2898e-5、加权KLa=1.3565e-5。该早期批次不支持“KL是首要压倒性梯度”的说法。epoch37 CNS KL梯度可大于已很弱的CE，不能据此倒推早期原因。

DEAP原fire_weight=0，记录的加权fire梯度为0是预期；错误分支的梯度为None也是结构预期。epoch1 CNS阈值CE为0不能单独证明整个SNN断路，当前surrogate不直接返回阈值梯度，阈值还可经reset路径影响后续时刻；本轮不改代理梯度。

## 3. 数值注意事项与不能得出的结论

epoch37的加权CNS KL为-2.7415e-10，数学KL应非负。原float32表达式mu^2+exp(logvar)-1-logvar在接近先验时存在相消；该极小负值与舍入量级相符，应标记为数值误差候选，不能解释成真实负KL或收益。本轮不加clamp、不删除KL、不改变原损失公式；若后续出现明显负值/非有限值再单独处理。

epoch37事实P1均值0.505435、跨窗口std约0.0003075；训练态dropout和中介噪声仍能产生概率变化，不足以证明对生理输入有判别依赖。

范数比不是实际Adam更新量之比，也不是历史因果贡献占比：方向、moment、eps和不同批次均影响实际更新。early/late两个局部点不能确定唯一根因。当前最有依据的工作假设是“任务梯度弱、原单组耦合L2引起参数收缩，两者形成退化反馈”，值得优先做单变量验证；没有证据支持修改SCM、标签、BN或删除KL。

## 4. W1最小实现与实验设计

W1不是原行动计划的E3；E3仍指只读ANS输入/中介依赖探针。这里不机械地继续运行所有探针，而以E2证据优先验证衰减假设。

新增--weight_decay，默认仍为原0.01；原Adam单参数组、lr=1e-4、betas/eps、clip顺序及全部参数范围不变，不换AdamW。显式取0的B组改变的是Adam内部耦合L2正则，不能声称其有效优化目标与A组完全相同。CE/KL/fire定义、权重、warmup、SCM、数据和选模规则均不变。

full诊断新增fold.json的start_parameter_rms、optimizer实际配置，以及history/epoch_metrics中optimization.parameter_rms（history位于train.parameter_rms），都是只读逐参数标量。初始RMS不是完整参数哈希，不能单凭RMS相同证明逐位一致。读统计不消耗随机数、不参与优化和选模。续训若权重文件中的实际weight_decay与CLI不同就停止，避免load_state_dict静默覆盖新设置；旧checkpoint没有CLI字段但优化器保存0.01时仍兼容默认值。

建议只做DEAP/arousal、held-out1，A/B各5轮、完整训练和测试窗口、seed42、相同硬件/环境，两组都从头初始化，不使用--resume，不继承epoch37的参数/moment。5轮用于观察原轨迹中epoch1–5的早期退化，不是正式LOSO，不是完整warmup训练，也不是超参数搜索。历史epoch1–37仍保持原样。

优先比较：六个卷积核RMS、CNS编码权重RMS、中介均值跨样本变化和posterior std、事实训练集CE/矩阵、事实/前门差异。测试ACC只作既有协议记录，不单凭最高ACC判断假设成立。若B保留尺度但事实学习仍无效，说明衰减可能参与退化但不足以解决任务/中介依赖问题，再做原E3/E4。若两组都类似退化，不立即把beta也设零，应查任务路径、尺度和优化批次。若A短程未复现旧趋势，应先核对来源、配置和设备差异，不宣布B有效。

## 5. 获取工具与可选服务器测试

继续使用原dslenv，不安装/替换环境。原项目checkout和结果不切换、不覆盖；只有新工具worktree使用此分支。以下先执行一次：

```bash
cd /home/fss/NeuroFDAN
git status --short
git branch --show-current
git worktree list
export W1_TOOL_DIR=/home/fss/NeuroFDAN-wd-control-tools
(
set -euo pipefail
git fetch origin refs/heads/feat/neurofdan-weight-decay-control:refs/remotes/origin/feat/neurofdan-weight-decay-control
if [ ! -e "$W1_TOOL_DIR" ]; then
    git worktree add --detach "$W1_TOOL_DIR" origin/feat/neurofdan-weight-decay-control
fi
test "$(git -C "$W1_TOOL_DIR" rev-parse HEAD)" = "$(git rev-parse origin/feat/neurofdan-weight-decay-control)" || {
    echo '工具版本不符，停止，不覆盖。'; exit 1;
}
test -z "$(git -C "$W1_TOOL_DIR" status --porcelain)" || {
    echo '工具有本地修改，停止，不覆盖。'; exit 1;
}
git -C "$W1_TOOL_DIR" log -1 --oneline
)
```

后续命令中的工具路径固定为/home/fss/NeuroFDAN-wd-control-tools。真正运行前先按交付回复中的版本检查命令核验HEAD、工作区干净，以及新CLI能读取参数。服务器可执行轻量测试，不会读取真实数据或训练NeuroFDAN：

```bash
cd /home/fss/NeuroFDAN
(cd /home/fss/NeuroFDAN-wd-control-tools && python -m unittest discover -s tests -p test_weight_decay_control.py -v)
```

仅该测试子进程切换工具目录以保证导入新版本；主训练和上传命令仍从原项目目录启动。测试失败先回传输出，不继续两组训练。

## 6. A组：原衰减系数，5轮

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
W1_RUN_DIR=/home/fss/NeuroFDAN/outputs/phase5_controls/DEAP/arousal/seed42/deap_arousal_fold01_wd001_5ep_001
mkdir -p "$(dirname "$W1_RUN_DIR")"
mkdir "$W1_RUN_DIR"
bash /home/fss/NeuroFDAN-wd-control-tools/scripts/phase3_run_logged.sh "$W1_RUN_DIR/train.log" \
  python -u /home/fss/NeuroFDAN-wd-control-tools/mainNeuroFDAN.py \
  --dataset DEAP --label arousal --root_dir /home/fss \
  --output_dir /home/fss/NeuroFDAN/outputs/phase5_controls \
  --run_name deap_arousal_fold01_wd001_5ep_001 \
  --held_out_subjects 1 --seed 42 --device cuda:0 \
  --epochs 5 --batch_size 40 --workers 0 --lr 0.0001 --weight_decay 0.01 \
  --diagnostic_level full --log_detail_interval 1 --fd_mode full \
  --mediator_dim 64 --hidden_dim 128 --dropout 0.3 --ref_dim 0 \
  --S_train 4 --K_train 16 --S_eval 16 --K_eval 64 --N_ref 2048 \
  --reference_chunk_size 4 --beta 0.001 --beta_warmup_epochs 10 \
  --lambda_obs 1 --fire_weight 0 --time_steps 4 --logvar_min=-6 --logvar_max 2 \
  --max_train_batches 0 --max_test_batches 0
cat "$W1_RUN_DIR/train.log.exit"
)
```

## 7. B组：只关闭耦合L2，5轮

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
W1_RUN_DIR=/home/fss/NeuroFDAN/outputs/phase5_controls/DEAP/arousal/seed42/deap_arousal_fold01_wd0_5ep_001
mkdir -p "$(dirname "$W1_RUN_DIR")"
mkdir "$W1_RUN_DIR"
bash /home/fss/NeuroFDAN-wd-control-tools/scripts/phase3_run_logged.sh "$W1_RUN_DIR/train.log" \
  python -u /home/fss/NeuroFDAN-wd-control-tools/mainNeuroFDAN.py \
  --dataset DEAP --label arousal --root_dir /home/fss \
  --output_dir /home/fss/NeuroFDAN/outputs/phase5_controls \
  --run_name deap_arousal_fold01_wd0_5ep_001 \
  --held_out_subjects 1 --seed 42 --device cuda:0 \
  --epochs 5 --batch_size 40 --workers 0 --lr 0.0001 --weight_decay 0 \
  --diagnostic_level full --log_detail_interval 1 --fd_mode full \
  --mediator_dim 64 --hidden_dim 128 --dropout 0.3 --ref_dim 0 \
  --S_train 4 --K_train 16 --S_eval 16 --K_eval 64 --N_ref 2048 \
  --reference_chunk_size 4 --beta 0.001 --beta_warmup_epochs 10 \
  --lambda_obs 1 --fire_weight 0 --time_steps 4 --logvar_min=-6 --logvar_max 2 \
  --max_train_batches 0 --max_test_batches 0
cat "$W1_RUN_DIR/train.log.exit"
)
```

两组分开串行运行，不共享run/fold。不创建validation；仍为probs_do主评价、事实CE监督、测试ACC/F1选模。无best只保留last仍是合法情况。缺GPU时两组都改为--device cpu，不跨设备混合后声称严格配对。若发生非有限值/OOM/读取错误先停止并上传日志，不增加别的超参数改动。

两组完成后可只读检查RMS、来源和逐轮事实训练表现（不导入模型，不改结果）：

```bash
cd /home/fss/NeuroFDAN
python - <<'PY'
import json
from pathlib import Path
base = Path('/home/fss/NeuroFDAN/outputs/phase5_controls/DEAP/arousal/seed42')
names = ['deap_arousal_fold01_wd001_5ep_001', 'deap_arousal_fold01_wd0_5ep_001']
configs, sources = [], []
for name in names:
    folder = base / name / 'fold01'
    cfg = json.loads((folder / 'fold.json').read_text())
    configs.append(cfg)
    sources.append(json.loads((folder / 'reference_sources.json').read_text()))
    print(name, '实际优化器：', cfg['optimizer'])
    for line in (folder / 'epoch_metrics.jsonl').read_text().splitlines():
        row = json.loads(line)
        rms = row['optimization']['parameter_rms']
        mean_spread = row['mediator_summary']['train']['cns']['posterior_mean_between_sample_std']['mean']
        factual = row['train']['prediction_diagnostics']['factual']['metrics']['acc']
        print('轮次', row['epoch'], '事实CE', row['optimization']['factual_ce'],
              '事实训练ACC', factual, 'CNS均值跨样本变化', mean_spread,
              'CNS首卷积RMS', rms['backbone.cns_enc.0.weight'],
              'ANS首卷积RMS', rms['backbone.ans_enc.0.weight'])
assert sources[0] == sources[1], '两组参考来源不一致，请先核查'
assert configs[0]['start_parameter_rms'] == configs[1]['start_parameter_rms'], '两组起始RMS不一致，请先核查'
print('参考来源和起始RMS一致；RMS相同不等于逐位参数哈希一致。')
PY
```

## 8. 两组各自dry-run和上传

分别运行以下四条命令。日志路径已显式指定，上传器只上传白名单轻量产物，不上传权重。新run名称独立，不覆盖原Phase4或E1/E1b/E2。

```bash
cd /home/fss/NeuroFDAN
bash /home/fss/NeuroFDAN-wd-control-tools/scripts/upload_run_diagnostics.sh \
  /home/fss/NeuroFDAN/outputs/phase5_controls/DEAP/arousal/seed42/deap_arousal_fold01_wd001_5ep_001 \
  --log /home/fss/NeuroFDAN/outputs/phase5_controls/DEAP/arousal/seed42/deap_arousal_fold01_wd001_5ep_001/train.log --dry-run
```

```bash
cd /home/fss/NeuroFDAN
bash /home/fss/NeuroFDAN-wd-control-tools/scripts/upload_run_diagnostics.sh \
  /home/fss/NeuroFDAN/outputs/phase5_controls/DEAP/arousal/seed42/deap_arousal_fold01_wd001_5ep_001 \
  --log /home/fss/NeuroFDAN/outputs/phase5_controls/DEAP/arousal/seed42/deap_arousal_fold01_wd001_5ep_001/train.log
```

```bash
cd /home/fss/NeuroFDAN
bash /home/fss/NeuroFDAN-wd-control-tools/scripts/upload_run_diagnostics.sh \
  /home/fss/NeuroFDAN/outputs/phase5_controls/DEAP/arousal/seed42/deap_arousal_fold01_wd0_5ep_001 \
  --log /home/fss/NeuroFDAN/outputs/phase5_controls/DEAP/arousal/seed42/deap_arousal_fold01_wd0_5ep_001/train.log --dry-run
```

```bash
cd /home/fss/NeuroFDAN
bash /home/fss/NeuroFDAN-wd-control-tools/scripts/upload_run_diagnostics.sh \
  /home/fss/NeuroFDAN/outputs/phase5_controls/DEAP/arousal/seed42/deap_arousal_fold01_wd0_5ep_001 \
  --log /home/fss/NeuroFDAN/outputs/phase5_controls/DEAP/arousal/seed42/deap_arousal_fold01_wd0_5ep_001/train.log
```

先看dry-run输出的manifest，再执行对应正式上传。分别回传两组的固定commit链接；不以当前server-diagnostics可变HEAD替代历史链接。要重跑时改用新编号，不能清空已有目录。

## 9. 实际检查状态

本轮仅静态审查和解析既有服务器JSON/log及Git对象。没有云端执行模型、CLI、单元测试、训练或GPU操作。新增测试文件尚未运行，不能写成测试通过。W1是待服务器执行的实验方案，没有A/B实测结果。真实潜在中介与因果识别边界、被试内无标签归一化及test-selected checkpoint偏差均保持原说明。

已实际通过：三个修改/新增Python文件AST解析；文档bash块的bash -n；训练参数名与CLI源码的静态匹配；git diff --check。AST对比确认评价、选模、参考库、种子、frontdoor_config及checkpoint保存/加载函数不变，train_one_epoch除full末尾读取RMS外与E2基线相同；模型/SCM/数据/loss.py/上传器文件无改动。这些都不能代替服务器运行测试。
