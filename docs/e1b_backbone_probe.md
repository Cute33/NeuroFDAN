# E1核验与E1b：骨干逐层只读诊断

代码起点：feat/neurofdan-phase5-diagnosis / 967f529c6d916f9ccf3acf50de71f2d7fbe338bc。新分支feat/neurofdan-e1b-backbone-probe只增加探针与说明，不修改模型、SCM、loss、预处理、BN策略、参考库、采样或选模规则。本轮不运行云端模型、CLI或测试。

## 1. 已读取的服务器E1证据

固定快照：server-diagnostics / eb2bfa42039b2741a155452777cec9600174b233，目录runs/2026-11-10_212416_unknown_unknown_run_af68ca84。probe_summary.json与logs/train_01.log一致，没有Traceback，epoch37正确；model_state_unchanged、checkpoint_file_unchanged为true，optimizer_created为false。

| 指标 | 服务器E1结果 |
|---|---:|
| CNS输入跨样本std | 0.9782742246549536 |
| ANS输入跨样本std | 0.9700111405864411 |
| CNS骨干输出跨样本std | 0 |
| ANS骨干输出跨样本std | 0 |
| 联合参考表示跨样本std | 0 |
| CNS posterior mu/logvar跨样本std | 均为0 |
| 非自环有效图行数 | 30/32 |
| 事实概率跨样本std | 约3.07e-4 |

结论限于同一checkpoint及所选16窗口：输入确有变化，但到骨干输出已失去跨样本差异。**标准差0不等于激活值全为0**，也可能是非零常量。ANS没有CNS图混合仍发生该现象，所以“所有图边被去掉”不能作为统一解释；但仍需查看CNS图前特征，不能声称图完全无问题。

CNS参数常量、ANS骨干常量时，ANS posterior及事实概率的少量变化仍可由随机中介引起，不能据此证明保留输入信息。尚不能确定哪一层先失效，更不能只凭epoch37断言骨干是训练早期的唯一根因。需要用原best的实际epoch和未训练参考进行时间对照；best名称不代表其分类有效。

上传没有发现跳过文件。目录unknown_unknown是旧上传器按训练fold.json提取命名、而纯探针没有该文件的归档限制；manifest.metadata为空及upload_summary无fold不表示E1失败。诊断内容应看probe_summary，不应把空训练摘要当探针结果。本轮不要求为目录名称重做E1，也不伪造训练fold.json。

## 2. 本轮新增探针的边界

scripts/probe_backbone_checkpoint.py先验证last.pt的格式、任务、被试、epoch、配置、SHA256及16窗口来源与E1相同。验证失败立即停止，不用随机模型替代last。

--include_initial添加seed42的未训练参考初始化，仅用于检查同一输入在构造网络时是否也失去变化，不保证与历史启动逐位相同。--include_best存在时附加读取实际best.pt，缺失或元数据不兼容会记录未加载；实际前向/状态校验异常仍停止。三个条件用相同输入和eval模式，不计算损失/梯度，不创建优化器。

沿真实backbone.forward安装只读hook，统计每个Conv/BN/SiLU的输入差异传播；记录参数RMS/L2、零值比例、BN运行统计和有效gamma/sqrt(var+eps)，以及CNS图前后差异。统计先转float64以避免对极小float32数平方造成测量下溢；**网络本身仍以原float32运行**，没有用更高精度改变前向。

原checkpoint只读并在结束时核验文件哈希；每次前向前后检查模型state_dict和grad。只保存形状与标量，不上传原信号、特征数组、权重或优化器动量。

## 3. 服务器完整执行命令

使用与E1相同的Python环境。在同一Xshell会话中依次执行运行、预览、上传。不会切换原项目分支；工具worktree固定在独立分支，原Phase4与E1结果保留。

```bash
cd /home/fss/NeuroFDAN

# 复用已激活的Phase4/E1环境；没有激活时仅尝试已有项目venv。
if [ -z "${VIRTUAL_ENV:-}" ] && [ -z "${CONDA_PREFIX:-}" ]; then
    for e1b_activate in .venv/bin/activate .venv-phase3/bin/activate venv/bin/activate; do
        if [ -f "$e1b_activate" ]; then
            source "$e1b_activate"
            break
        fi
    done
fi

export E1B_TOOL_DIR=/home/fss/NeuroFDAN-e1b-tools
export E1B_RESULT_DIR=/home/fss/NeuroFDAN/outputs/phase5_probes/deap_arousal_fold01_epoch37_e1b_001

(
set -euo pipefail
cd /home/fss/NeuroFDAN
python -c "import sys, torch, numpy; print('解释器：', sys.executable); print('PyTorch：', torch.__version__)"
git status --short
git branch --show-current
git worktree list
git fetch origin refs/heads/feat/neurofdan-e1b-backbone-probe:refs/remotes/origin/feat/neurofdan-e1b-backbone-probe
if [ ! -e "$E1B_TOOL_DIR" ]; then
    git worktree add --detach "$E1B_TOOL_DIR" origin/feat/neurofdan-e1b-backbone-probe
fi
test "$(git -C "$E1B_TOOL_DIR" rev-parse HEAD)" = "$(git rev-parse origin/feat/neurofdan-e1b-backbone-probe)" || {
    echo '工具worktree版本与远端不同；停止，不自动覆盖。'; exit 1;
}
test -z "$(git -C "$E1B_TOOL_DIR" status --porcelain)" || {
    echo '工具worktree不干净；停止，不覆盖。'; exit 1;
}
mkdir -p outputs/phase5_probes
mkdir "$E1B_RESULT_DIR"
bash "$E1B_TOOL_DIR/scripts/phase3_run_logged.sh" "$E1B_RESULT_DIR/train.log" \
  python -u "$E1B_TOOL_DIR/scripts/probe_backbone_checkpoint.py" \
  --root_dir /home/fss --dataset DEAP --task arousal \
  --held_out_subject 1 --seed 42 --device cpu --samples 16 \
  --expected_epoch 37 \
  --fold_dir /home/fss/NeuroFDAN/outputs/phase4/DEAP/arousal/seed42/deap_arousal_fold01_full/fold01 \
  --e1_summary /home/fss/NeuroFDAN/outputs/phase5_probes/deap_arousal_fold01_epoch37_e1_001/probe_summary.json \
  --output_dir "$E1B_RESULT_DIR" \
  --include_initial --include_best
cat "$E1B_RESULT_DIR/train.log.exit"
)
```

如使用其他conda/venv，请先激活E1原环境。输出目录已存在时mkdir直接停止，不追加原日志；再次操作请统一把e1b_001改为e1b_002。不需要运行mainNeuroFDAN.py，不需要任何训练epoch、S/K调整或validation。

结果路径：

```text
/home/fss/NeuroFDAN/outputs/phase5_probes/deap_arousal_fold01_epoch37_e1b_001/
  probe_summary.json
  train.log
  train.log.exit
```

正常退出码为python_exit=0、tee_exit=0。权重缺失/哈希不同/配置不符时先保留日志，不重训、不自动改用best或随机初始化；可上传失败日志供进一步核查。

## 4. 完整预览与正式上传命令

train.log位于结果目录，上传器会自动识别；下面仍显式给出。保持相同环境，先预览：

```bash
cd /home/fss/NeuroFDAN
bash /home/fss/NeuroFDAN-e1b-tools/scripts/upload_run_diagnostics.sh \
  /home/fss/NeuroFDAN/outputs/phase5_probes/deap_arousal_fold01_epoch37_e1b_001 \
  --log /home/fss/NeuroFDAN/outputs/phase5_probes/deap_arousal_fold01_epoch37_e1b_001/train.log \
  --dry-run
```

查看临时快照manifest及脱敏文件后正式上传：

```bash
cd /home/fss/NeuroFDAN
bash /home/fss/NeuroFDAN-e1b-tools/scripts/upload_run_diagnostics.sh \
  /home/fss/NeuroFDAN/outputs/phase5_probes/deap_arousal_fold01_epoch37_e1b_001 \
  --log /home/fss/NeuroFDAN/outputs/phase5_probes/deap_arousal_fold01_epoch37_e1b_001/train.log
```

server-diagnostics会新增唯一目录，不覆盖E1或epoch1–37快照。纯探针仍可能以unknown命名，内容中的dataset/task/fold有效；把工具输出的固定commit链接发给Codex即可。不发送.pt/.pth文件。

## 5. 结果如何决定下一步

- 初始网络有变化、best/last逐渐失去变化：重点查训练期间参数退化，不证明某个正则项是唯一原因。
- 某卷积输出首次变常量，且卷积权重全零：已定位到该卷积的输入依赖失效；再查它为什么在优化中变零。
- 卷积仍有变化、BN后变常量，且gamma为零或极小：重点查BN仿射尺度和参数更新；本轮不解冻BN统计或改权重。
- 图前CNS仍有变化、图后没有：进一步核验选中的通道及图操作；ANS问题仍需独立解释。
- 参数非零但差异随深度消失：检查尺度衰减、偏置主导和float32分辨率，不能仅凭非零范数断言模块有效。
- E1b不能复现同权重的E1：先核查样本、环境、数据文件或数值差异，不继续推导原因。

拿到逐层结果后，才选择一个最小训练修复或分项梯度验证；不先批量调参或重跑100轮/LOSO。

## 6. 云端验证范围

仅静态语法与Git差异检查，未导入或执行模型、CLI、单元测试或探针，也未生成任何E1b服务器结果。原模型和训练相关文件保持基线内容。需要服务器验证实际加载、逐层统计、输出及上传。

已对E1快照中manifest列出的4个载荷逐一核验字节数和SHA256，均一致；日志内的结果JSON与独立probe_summary.json完全一致。这是已有结果的数据完整性分析，不是云端模型测试。新探针及本文运行/上传命令仅通过静态语法解析，没有声称运行通过。
