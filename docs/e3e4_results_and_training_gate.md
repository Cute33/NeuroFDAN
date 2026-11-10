# E3/E4结果、诊断收敛与20轮有限学习验证

## 已核验结果

固定提交e8d2893838160af4224af413393bb7fa6fd52f61，目录runs/2026-11-10_234301_unknown_unknown_run_5aed407e。四个载荷大小及SHA256匹配，日志无Traceback，两组参数/buffer和checkpoint文件均记录不变，无优化器/梯度/step。参考库2048，K64有放回抽样得到63个不同来源是正常情况；参考不含测试被试且当前窗口与参考无重合。缺少的是可选训练摘要，unknown名称不是失败。

A组epoch5骨干仍有变化，但CNS mu和logvar均逐元素跨窗口相同，ANS自身输入影响约1e-12，确认早期信息丢失在中介路径，不必等到骨干完全常量。

B组epoch5在共享噪声下，改变CNS输入使Mc变化RMS=0.0021783、ANS mu变化0.0040606；改变CNS噪声对应1.36478、0.26438。后者ANS均值变化约前者65.1倍。ANS输入替换的mu变化0.0140375，说明ANS自身路径也存在。电流附加项/本征项RMS约5.94，但整体RMS含均值和噪声，不是输入因果贡献比例。

B组真实CNS/ANS输入替换的do概率变化RMS分别2.32e-6、1.29e-5，换CNS噪声为9.06e-5。这是有限样本、特定替换幅度的局部敏感度，不是完整SCM的真实因果效应。

B组单次事实P1 std=0.0080744，同S事实平均std=0.0077796，同S/K前门std=8.9982e-6。固定Ma替换事实参考的概率变化RMS=0.0112791。Head对Ma及参考均有条件响应；参考边缘化后，主要保留的中介输入信号很弱。不能据此宣布K平均错误，也不能把不同幅度替换得到的范数当作全局归因比例或显著性检验。

结论分层：W1支持原耦合L2促进参数退化；E3/E4支持关闭衰减后仍存在输入变化相对噪声弱、结果条件模型学习不足的问题。当前没找到必须更改标签、概率积分或SCM的实现bug。真实潜在中介不可观测，其可识别性并未由这些探针证明。研究目标仍为联合X=(Xc,Xa)、Mc→Ma与真实前门估计；不以增加观察标签do损失、直接Mc→Y、融合替代或改阈值求分数。

## 停止增加探针，先作一次有界学习验证

只把W1 B组从epoch5续训到总计epoch20，新增15轮，保留beta=0.001及10轮warmup，观察warmup之后是否真正学习。wd=0仍是显式实验设置，不改正式默认0.01；不改其他模型、损失、数据、S/K、参考总体或选模协议。新路径保存原5轮副本，原run完全保留；继承的epoch1–5及best需如实标识，不能称为20轮全新从头训练。续训重设RNG，不承诺等同一次未中断20轮；不把它当成新的精确配对A/B实验。

采用已实测的9d698f3fcfbd6b41579673ddb43e3a8074b9d435训练工具，不需要新训练代码。本分支仅存说明，不修改源码。

## 服务器准备与原结果保护

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
git status --short
git worktree list
test "$(git -C /home/fss/NeuroFDAN-wd-control-tools rev-parse HEAD)" = 9d698f3fcfbd6b41579673ddb43e3a8074b9d435
test -z "$(git -C /home/fss/NeuroFDAN-wd-control-tools status --porcelain)"
python - <<'PY'
import hashlib, json, shutil
from pathlib import Path
base = Path('/home/fss/NeuroFDAN/outputs/phase5_controls/DEAP/arousal/seed42')
source = base / 'deap_arousal_fold01_wd0_5ep_001'
destination = base / 'deap_arousal_fold01_wd0_20ep_followup_001'
def fingerprint(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()
expected = '2b31c2034a11e34e2860b0d6ba144fd0d431cc6240c23a3a8126bab1bedd3fe7'
assert fingerprint(source/'fold01/last.pt') == expected, '原last.pt与E3/E4读取版本不同，停止'
if destination.exists():
    raise FileExistsError('新目录已存在；换新编号，不覆盖或自动重新开始')
shutil.copytree(source, destination)
assert fingerprint(source/'fold01/last.pt') == expected
assert fingerprint(destination/'fold01/last.pt') == expected
(destination/'config.json').write_text(json.dumps(dict(
    operation='独立副本续训；前5轮继承W1 B组，新增6–20轮',
    source_run=source.name, source_last_sha256=expected,
    original_epochs=5, target_epochs=20,
    original_run_preserved=True,
    note='复制的旧摘要在续训完成前不是20轮完成凭证；续训不保证逐位等同连续训练'
), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print('已保留原结果，创建独立续训副本：', destination)
PY
)
```

若准备步骤失败，不执行下列训练。不删除原目录或绕过哈希检查。工具目录不存在时先按W1指南获取固定9d698f3版本，不切换原项目分支。继续使用dslenv。

## 服务器完整运行命令

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
FOLLOWUP_DIR=/home/fss/NeuroFDAN/outputs/phase5_controls/DEAP/arousal/seed42/deap_arousal_fold01_wd0_20ep_followup_001
bash /home/fss/NeuroFDAN-wd-control-tools/scripts/phase3_run_logged.sh "$FOLLOWUP_DIR/train.log" \
  python -u /home/fss/NeuroFDAN-wd-control-tools/mainNeuroFDAN.py \
  --dataset DEAP --label arousal --root_dir /home/fss \
  --output_dir /home/fss/NeuroFDAN/outputs/phase5_controls \
  --run_name deap_arousal_fold01_wd0_20ep_followup_001 \
  --held_out_subjects 1 --seed 42 --device cuda:0 \
  --epochs 20 --batch_size 40 --workers 0 --lr 0.0001 --weight_decay 0 \
  --diagnostic_level full --log_detail_interval 1 --fd_mode full \
  --mediator_dim 64 --hidden_dim 128 --dropout 0.3 --ref_dim 0 \
  --S_train 4 --K_train 16 --S_eval 16 --K_eval 64 --N_ref 2048 \
  --reference_chunk_size 4 --beta 0.001 --beta_warmup_epochs 10 \
  --lambda_obs 1 --fire_weight 0 --time_steps 4 --logvar_min=-6 --logvar_max 2 \
  --max_train_batches 0 --max_test_batches 0 \
  --resume "$FOLLOWUP_DIR/fold01/last.pt"
cat "$FOLLOWUP_DIR/train.log.exit"
)
```

首行应显示继续epoch=6；否则停止核查。日志沿用复制的旧日志再追加，JSONL前5轮来自原run，新的summary/fold.json应对应目标20轮。只以最终history/epoch_metrics到20且summary一致作为完成凭证，不看刚复制的旧run_summary就声称完成。原检查点规则、无validation及测试选模偏差保持不变。

## dry-run与结果上传

```bash
cd /home/fss/NeuroFDAN
bash /home/fss/NeuroFDAN-wd-control-tools/scripts/upload_run_diagnostics.sh \
  /home/fss/NeuroFDAN/outputs/phase5_controls/DEAP/arousal/seed42/deap_arousal_fold01_wd0_20ep_followup_001 \
  --log /home/fss/NeuroFDAN/outputs/phase5_controls/DEAP/arousal/seed42/deap_arousal_fold01_wd0_20ep_followup_001/train.log \
  --dry-run
```

看过manifest后正式上传：

```bash
cd /home/fss/NeuroFDAN
bash /home/fss/NeuroFDAN-wd-control-tools/scripts/upload_run_diagnostics.sh \
  /home/fss/NeuroFDAN/outputs/phase5_controls/DEAP/arousal/seed42/deap_arousal_fold01_wd0_20ep_followup_001 \
  --log /home/fss/NeuroFDAN/outputs/phase5_controls/DEAP/arousal/seed42/deap_arousal_fold01_wd0_20ep_followup_001/train.log
```

独立快照追加，不上传权重。错误退出也可上传已有日志，不补造完成记录。回传工具打印的SHA和链接。

## 何时进入正式运行，以及时间估算

本轮只安排这一次有限续训，epoch20必须复盘，不自动扩展。重点观察最后5轮事实训练CE是否持续改善、两类预测是否有实际判别、中介输入变化是否维持、前门概率是否仍接近输入无关。仅概率跨过0.5或测试ACC达到多数类比例不算通过，也不更换判决阈值。若出现有意义学习，再安排一个完整单fold验证稳定性后决定LOSO；若仍接近随机/单类，暂停大规模实验，提出有明确数学依据的中介/条件结果模型学习方案，而不是继续堆探针或同时调多个超参数。

已上传W1日志约11.6–12秒/epoch；新增15轮训练段约3分钟。相同服务器环境可先预留5–15分钟作加载/保存/回传的执行预算，实际随负载变化。诊断能否解决和正式实验何时可用不能保证由这一运行时间决定。最多先做这次有界验证，再依据结果决定是否需要一项针对性修订与验证，不预先保证再做几次就一定成功。

若沿用fold1速度，DEAP单任务32fold×100轮的训练/逐轮评估约10–11小时，尚未计全部数据加载、额外验证及不同fold负载，非实测完整LOSO耗时；HCI不能直接套用该数字。正式结果还必须包含所有预定fold及选模/归一化偏差说明。

## 本轮实际工作范围

仅读取现有服务器产物和静态审查，没有云端模型、CLI、单元测试、训练或GPU实验。本次没有更改源码、SCM或训练协议。文档中的复制和续训尚待服务器执行，不能声称已开始或通过20轮验证。
