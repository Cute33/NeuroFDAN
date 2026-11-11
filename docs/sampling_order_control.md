# 批次审计结论与采样顺序单变量对照

## 1. 本次证据及解释边界

固定快照6f0e0d262c63c750fbf0ace778358058436e4960，目录runs/2026-11-11_103731_unknown_unknown_run_5b5386be。四个载荷字节数和SHA256均匹配，日志无Traceback，skipped为空；纯探针缺少训练摘要不影响结果。审计未运行模型、未训练或改变采样器。

原分组20轮合计5200批，690批单类，占13.27%；每批4个trial；正类比例std的跨轮均值0.25224。虚拟窗口打乱单类批0/5200，平均每批39.324个trial，比例std均值0.07929。前5轮和重启后的6–10轮顺序重复，5200批不是独立重复实验，不能用于夸大统计显著性。

这支持存在批次多样性低、标签相关性强的优化候选，不等于已证实导致学习失败。末批正类比例与测试平均P1的描述性Pearson r≈0.057；末8批平均比例对应r≈-0.626，并不支持简单的“末批正类越多，整轮越倾向预测正类”。不据此声称已解释输出bias或因果机制。

原权重收缩、潜在中介信号相对噪声弱、采样批次相关性应分开报告。目前仍没有证据支持改变SCM、阈值或观察标签do监督。研究目标保持联合X=(Xc,Xa)、Mc→Ma、OutcomeHead(Ma,g(X'))以及唯一前门概率积分；采样对照只检验优化排序，不是物理干预或新的因果图。

## 2. 最小实现与默认兼容

新增--sampling_mode grouped|window，默认grouped，保留原trial整组排列。window复用原GroupedSampler，令group_size=1，使用独立seed42生成器做逐窗口均匀无放回打乱。原预处理、标签、trial下采样、每轮样本集合、drop_last、优化器、CE/KL/fire、warmup、BN、参考总体和选模规则不变；没有类别分层/加权采样。window不是新默认协议，不把窗口打乱等同于真正增加独立受试者或trial总数。

fold.json记录sampling_mode与optimization_group_size，checkpoint arguments保存选择；续训时禁止跨模式切换，旧checkpoint缺少该字段时视为grouped。原window长度和模型的窗口内时间轴不受影响。参考库仍来自全部未平衡训练窗口，改变优化批次不会改p(X')来源。

修改Datasets.make_loaders、mainNeuroFDAN的参数、ModelTrain的调用/日志/来源记录与续训校验；模型、SCM、loss.py及实际train_one_epoch/evaluate/BestCheckpoint完全不变。新增CPU小测试但本轮不在云端执行。

## 3. 一次有界对照，不再追加同类探针

两组同代码、seed42、wd=0、20轮、DEAP arousal fold1、完整训练/测试，均从头初始化：A=grouped，B=window。只计划改变sampling_mode；名称独立。旧20轮在epoch6重启过，不能直接当作新window连续20轮的严格配对控制，因此需要两组都从头运行。不是补跑旧100轮或启动LOSO。

同时保持S_train4/K_train16/S_eval16/K_eval64、N_ref2048、beta0.001、warmup10、batch40、lr1e-4、无validation、probs_do主评价及原测试ACC/F1选模。wd=0沿用当前实验条件，正式默认仍0.01。

按既有约12秒/轮的运行记录，两组训练/逐轮评估合计约8分钟；含读取、保存和上传可先预留10–20分钟，实际依负载而变。不能保证此对照必然提高ACC。第20轮后必须复盘：若两组都未有效学习，停止扩大epoch/fold，转入中介/条件结果模型学习目标审查，不继续堆采样探针。

## 4. 获取代码并先执行小测试

继续复用dslenv，不覆盖原项目分支或结果。

```bash
cd /home/fss/NeuroFDAN
export ORDER_TOOL_DIR=/home/fss/NeuroFDAN-sampling-control-tools
(
set -euo pipefail
git status --short
git worktree list
git fetch origin refs/heads/feat/neurofdan-sampling-order-control:refs/remotes/origin/feat/neurofdan-sampling-order-control
if [ ! -e "$ORDER_TOOL_DIR" ]; then
    git worktree add --detach "$ORDER_TOOL_DIR" origin/feat/neurofdan-sampling-order-control
fi
test "$(git -C "$ORDER_TOOL_DIR" rev-parse HEAD)" = "$(git rev-parse origin/feat/neurofdan-sampling-order-control)" || {
    echo '版本不符，停止，不覆盖。'; exit 1;
}
test -z "$(git -C "$ORDER_TOOL_DIR" status --porcelain)" || {
    echo '工具有本地修改，停止，不覆盖。'; exit 1;
}
(cd "$ORDER_TOOL_DIR" && python -m unittest discover -s tests -p test_sampling_order_control.py -v)
)
```

仅测试子进程切换工具目录以导入新代码。这5项测试只使用小型TensorDataset，不读取真实生理数据或训练模型。失败先回传输出，不继续两组训练。再次运行实验须换新编号，不删除原目录。

## 5. A组完整命令：原分组，连续20轮

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
ORDER_RUN_DIR=/home/fss/NeuroFDAN/outputs/phase5_sampling/DEAP/arousal/seed42/deap_arousal_fold01_grouped20_001
mkdir -p "$(dirname "$ORDER_RUN_DIR")"
mkdir "$ORDER_RUN_DIR"
bash /home/fss/NeuroFDAN-sampling-control-tools/scripts/phase3_run_logged.sh "$ORDER_RUN_DIR/train.log" \
  python -u /home/fss/NeuroFDAN-sampling-control-tools/mainNeuroFDAN.py \
  --dataset DEAP --label arousal --root_dir /home/fss \
  --output_dir /home/fss/NeuroFDAN/outputs/phase5_sampling \
  --run_name deap_arousal_fold01_grouped20_001 \
  --held_out_subjects 1 --seed 42 --device cuda:0 \
  --epochs 20 --batch_size 40 --workers 0 --lr 0.0001 --weight_decay 0 \
  --sampling_mode grouped --diagnostic_level full --log_detail_interval 1 --fd_mode full \
  --mediator_dim 64 --hidden_dim 128 --dropout 0.3 --ref_dim 0 \
  --S_train 4 --K_train 16 --S_eval 16 --K_eval 64 --N_ref 2048 \
  --reference_chunk_size 4 --beta 0.001 --beta_warmup_epochs 10 \
  --lambda_obs 1 --fire_weight 0 --time_steps 4 --logvar_min=-6 --logvar_max 2 \
  --max_train_batches 0 --max_test_batches 0
cat "$ORDER_RUN_DIR/train.log.exit"
)
```

## 6. B组完整命令：窗口打乱，连续20轮

```bash
cd /home/fss/NeuroFDAN
(
set -euo pipefail
ORDER_RUN_DIR=/home/fss/NeuroFDAN/outputs/phase5_sampling/DEAP/arousal/seed42/deap_arousal_fold01_window20_001
mkdir -p "$(dirname "$ORDER_RUN_DIR")"
mkdir "$ORDER_RUN_DIR"
bash /home/fss/NeuroFDAN-sampling-control-tools/scripts/phase3_run_logged.sh "$ORDER_RUN_DIR/train.log" \
  python -u /home/fss/NeuroFDAN-sampling-control-tools/mainNeuroFDAN.py \
  --dataset DEAP --label arousal --root_dir /home/fss \
  --output_dir /home/fss/NeuroFDAN/outputs/phase5_sampling \
  --run_name deap_arousal_fold01_window20_001 \
  --held_out_subjects 1 --seed 42 --device cuda:0 \
  --epochs 20 --batch_size 40 --workers 0 --lr 0.0001 --weight_decay 0 \
  --sampling_mode window --diagnostic_level full --log_detail_interval 1 --fd_mode full \
  --mediator_dim 64 --hidden_dim 128 --dropout 0.3 --ref_dim 0 \
  --S_train 4 --K_train 16 --S_eval 16 --K_eval 64 --N_ref 2048 \
  --reference_chunk_size 4 --beta 0.001 --beta_warmup_epochs 10 \
  --lambda_obs 1 --fire_weight 0 --time_steps 4 --logvar_min=-6 --logvar_max 2 \
  --max_train_batches 0 --max_test_batches 0
cat "$ORDER_RUN_DIR/train.log.exit"
)
```

两组串行运行，不使用--resume，不复制旧checkpoint。应分别显示打乱组大小10和1，从epoch1开始，最后两个退出码均0。若无CUDA，两组都改cpu；不混用设备后声称严格配对。

## 7. dry-run与正式上传

先预览两组：

```bash
cd /home/fss/NeuroFDAN
for ORDER_RUN_NAME in deap_arousal_fold01_grouped20_001 deap_arousal_fold01_window20_001
do
    ORDER_UPLOAD_DIR="/home/fss/NeuroFDAN/outputs/phase5_sampling/DEAP/arousal/seed42/$ORDER_RUN_NAME"
    bash /home/fss/NeuroFDAN-sampling-control-tools/scripts/upload_run_diagnostics.sh \
      "$ORDER_UPLOAD_DIR" --log "$ORDER_UPLOAD_DIR/train.log" --dry-run
done
```

检查manifest后正式上传：

```bash
cd /home/fss/NeuroFDAN
for ORDER_RUN_NAME in deap_arousal_fold01_grouped20_001 deap_arousal_fold01_window20_001
do
    ORDER_UPLOAD_DIR="/home/fss/NeuroFDAN/outputs/phase5_sampling/DEAP/arousal/seed42/$ORDER_RUN_NAME"
    bash /home/fss/NeuroFDAN-sampling-control-tools/scripts/upload_run_diagnostics.sh \
      "$ORDER_UPLOAD_DIR" --log "$ORDER_UPLOAD_DIR/train.log"
done
```

分别回传两条固定SHA/链接。错误退出也保留日志，不能伪造完整记录。上传不包含权重，不覆盖旧快照。

## 8. 预先定义结果解读

先核验两组样本数、参考来源、起始RMS及其余参数相同，均完整到20；RMS不代替逐位权重哈希。原始JSONL和TensorBoard记录应保留，不能只报最高测试ACC。

主要看最后5轮事实训练CE/完整训练矩阵、预测分布、中介变化与参数RMS；前门主输出仍按原协议测试/选模，不改用事实路径挑best。两组都没有独立validation，测试选模偏差仍需报告。

若window显著改善事实学习和前门输入变化，支持优化排序参与当前失败，下一步才验证单fold稳定性；不自动升级为新的正式协议或真实因果识别证明。若只改善事实而前门仍弱，转向中介/结果条件模型的学习依据，不对do直接加观察标签CE。若两组都弱，停止采样方向追加试验，进入学习目标/参数化审查。单折单种子对照不证明跨数据集结论。

本轮没有云端模型、CLI、单元测试或训练；新增测试尚待服务器执行，不能写成通过。代码提交不等于两组实验已运行。没有修改main、创建PR或合并。

已实际通过：四个修改/新增Python文件AST解析、五个文档bash块的bash -n、训练参数名静态核对、git diff --check。AST对比确认原优化更新、评估、选模、参考库、预处理、标签、下采样及GroupedSampler主体未改变；模型/SCM/loss.py文件无差异。上述检查不代替服务器测试和训练。
