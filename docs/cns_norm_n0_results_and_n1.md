# N0结果与真实DEAP固定400窗N1对照

## 原始结果核验

提交becbb9351df891d3d2de6f5c1a5ac1c1bbeb3ebc，目录runs/2026-11-13_124658_unknown_unknown_run_6837fb43。5项载荷长度/SHA256一致，无skipped；missing仅通用run_summary/run_report/config。probe_summary.completed=true，三组均完成200步，X/Y数据一致、初始参数SHA256一致。服务器211项轻量测试通过（15.424秒，skipped=1），评价缺失m_c的问题已修复。

| 条件 | 测试ACC | 正类F1 | do概率MAE | 观察NLL | Brier |
|---|---:|---:|---:|---:|---:|
| 观察监督none | 0.578125 | 0.564516 | 0.165401 | 0.858773 | 0.284629 |
| 观察监督CNS LayerNorm | 0.609375 | 0.596774 | 0.134093 | 0.800284 | 0.274865 |
| Oracle中介参考 | 0.570313 | 0.573643 | 0.153592 | 0.681846 | 0.244392 |

观察两组矩阵分别[[39,29],[25,35]]和[[41,27],[23,37]]，都能预测两类；归一化只多判对4/128，单seed、小样本不能称稳定或显著提高。干预概率误差也下降，但仅16个评价点，直接真值SE约0.00176–0.00361、学习估计MC std约0.02，不能外推真实数据识别精度。Oracle的训练目标/可用信息不同，有限预算下不构成分类ACC上界；它的真实中介坐标恢复明显更好。

归一化观察组NLL仍高于0.5常数预测的log(2)约0.6931，Brier仍高于其0.25；不能说概率已校准或方法已经成熟。幅度反例中归一化输入差为0，而两个do概率MC估计为0.5012/0.8917，确认了归一化会丢掉某些可能有用的幅度信息。该候选只获得进一步进行小规模真实对照的初步支持，不直接放行完整LOSO或声称中介因果识别成功。

## N1只改变一个训练因素

对比已经上传的bfe108e beta0/none 300步结果，本次开启cns_input_norm=layernorm。原自然训练总体、固定400窗、2048参考成员、seed42、lr2e-5、Adam/wd0、fire0.028、LIF256、hidden128、mediator64、S/K=4/16和16/64、BN冻结和原始数据处理全部保留。beta仍为0，是隔离归一化作用的诊断组；正式方法的默认正KL不删除。没有新的模型/损失改动。

只重复拟合同400个训练窗口，最多300个optimizer step，不是完整epoch训练或held-out评价。以ACC/F1为主要分类记录；工程验收为至少50步后3次MC均满足ACC≥0.8、两类最小Recall≥0.6、观察NLL低于该400窗最优常数至少0.05。两类Recall仅防止单类塌缩，不要求论文汇报Balanced Accuracy，且不更改正式test-selected checkpoint规则。

新增`--fit_norm_baseline_summary`与旧KL对照标志互斥。启动前逐项检查参数仅norm/输出位置不同，检查400查询、2048参考成员顺序、标签计数[153,247]及总体步数一致；归一化有意改变前向，不比较首步NLL相等。

为验证初始化，先在隔离CPU随机流内构造none模板，恢复随机流后构造norm模型，比较所有初始参数字节指纹，相同才继续。默认none行为已有服务器兼容测试支持，但历史bfe108e未记录初始参数指纹，不能声称对历史初始权重已逐字节验证。原baseline摘要在末端核对文件哈希不变，不重新训练baseline，也不读取其末端权重作为起点。

即使通过也只保存新目录中的私有末端权重，不自动运行20轮。通过后再交付正KL的N2同因素验证及完整单fold命令；如果仍单类，先读新权重/记录，不自动加轮数或启动LOSO。

## 服务器只执行这一块：测试→N1→配对报告→自动上传

```bash
cd /home/fss/NeuroFDAN
conda activate dslenv
(
set -euo pipefail
if [ -z "${GIT_SSH_COMMAND:-}" ]; then
    export GIT_SSH_COMMAND='ssh -p 443 -o HostName=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=2 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
fi
git status --short
timeout 600 git fetch --no-tags origin feat/neurofdan-cns-norm-n1
N1_SHA="$(git rev-parse FETCH_HEAD)"
N1_TOOLS=/home/fss/NeuroFDAN-cns-norm-n1-tools
if [ ! -e "$N1_TOOLS" ]; then git worktree add --detach "$N1_TOOLS" "$N1_SHA"; fi
test "$(git -C "$N1_TOOLS" rev-parse HEAD)" = "$N1_SHA"
test -z "$(git -C "$N1_TOOLS" status --porcelain)"
git -C "$N1_TOOLS" log -1 --oneline
bash "$N1_TOOLS/scripts/run_cns_norm_n1_and_upload.sh" \
    /home/fss \
    /home/fss/NeuroFDAN/outputs/phase6/cns_input_norm/n1_real \
    deap_arousal_fold02_cnsln_beta0_fit300_001 \
    /home/fss/NeuroFDAN/outputs/phase6/scheme_a_kl_control/DEAP/arousal/seed42/deap_arousal_fold02_scheme_a_beta0_fit300_001/probe_summary.json
)
```

先运行13项CPU测试（原方案A9项、新配对4项），通过后才运行下面完整命令；已由runner调用，不要重复执行：

```bash
python -u /home/fss/NeuroFDAN-cns-norm-n1-tools/scripts/scheme_a_fit_check.py \
  --dataset DEAP --label arousal --root_dir /home/fss \
  --output_dir /home/fss/NeuroFDAN/outputs/phase6/cns_input_norm/n1_real \
  --run_name deap_arousal_fold02_cnsln_beta0_fit300_001 \
  --seed 42 --device cuda:0 --held_out_subjects 2 --epochs 20 \
  --lr 0.00002 --weight_decay 0 --batch_size 400 --eval_batch_size 400 --workers 0 \
  --sampling_mode window --fd_mode full --no_balance --cns_input_norm layernorm \
  --mediator_dim 64 --hidden_dim 128 --lif_neurons 256 --dropout 0.3 --ref_dim 0 \
  --time_steps 4 --logvar_min -6 --logvar_max 2 \
  --S_train 4 --K_train 16 --S_eval 16 --K_eval 64 --N_ref 2048 --reference_chunk_size 4 \
  --classification_objective integrated_observational --lambda_obs 0 --factual_ce_mode all_samples \
  --beta 0 --beta_warmup_epochs 10 --fire_weight 0.028 \
  --diagnostic_level full --log_detail_interval 5 --loss_gradient_interval 5 \
  --max_train_batches 0 --max_test_batches 0 --fit_max_steps 300 --fit_check_every 50 \
  --fit_gate_metric acc --fit_save_checkpoint \
  --fit_norm_baseline_summary /home/fss/NeuroFDAN/outputs/phase6/scheme_a_kl_control/DEAP/arousal/seed42/deap_arousal_fold02_scheme_a_beta0_fit300_001/probe_summary.json
```

epochs20仅保留原拟合入口元数据；实际执行私有step，不调用mainNeuroFDAN.py，不代表训练20轮。训练日志通过phase3_run_logged.sh记录，不覆盖旧目录。

## 结果和自动上传

源目录：`/home/fss/NeuroFDAN/outputs/phase6/cns_input_norm/n1_real/DEAP/arousal/seed42/deap_arousal_fold02_cnsln_beta0_fit300_001`。

probe_summary保存实际步数、每点3次MC矩阵/ACC/F1/NLL、posterior和分项梯度、初始化/来源核对；run_summary为与历史beta0基线的私有拟合比较，不是正式LOSO报告。config记录阶段退出码。train.log、interface_tests.log、summary.log、comparison.log及.exit自动保留。private_fit_last.pt只保存在服务器新目录，不上传。

runner已包含两组动作，不需要另跑上传：先dry-run，再正式上传新run，使用当前确认可用的上传器，--git_timeout 600 --max_file_mib 16 --max_total_mib 64。自动附加训练、测试、摘要和比较日志；独立上传日志/快照在n1_real/upload_logs、n1_real/upload_bundles，保留upload_state.json。旧基线及N0不覆盖、不重复上传，失败也尝试上传已有日志。

回传最后SHA及链接即可。退出3表示预算内未通过拟合，不是自动认定程序错误；上传成功和训练通过分别判断。

## 本轮交付边界

只增加N1协议/运行器/测试与本文，扩展原拟合脚本的可选归一化对照分支；核心模型、损失、数据和选模实现不再改动。N0的211项通过是服务器证据；新4项配对测试尚未在云端执行。云端只读GitHub产物并校验哈希，执行AST/Shell/Git静态检查，没有运行模型、单测、CLI、真实训练或GPU。不修改main、不创建PR或合并。
