# NeuroFDAN

当前开发版本把 Phase 1 前门 SCM 核心接入真实生理骨干和 LOSO train/test 流程。第一阶段备份为 `feat/neurofdan-phase1`，第二阶段稳定版本为 `feat/neurofdan-phase2`。第三阶段 `feat/neurofdan-phase3` 从第二阶段核验提交独立创建，提供服务器操作指南与结果完整性审计。核验时第二阶段已由 PR #3 合并 main；本轮不修改 main。

第四阶段 `feat/neurofdan-phase4` 从 Phase 3 的 `cd844c85d28f2335acf4d41083184d8ad98e5bbd` 创建，仅增强训练诊断。默认每轮完整评估优化训练子集与测试集，打印混淆矩阵；`--diagnostic_level full` 增加中介和裁剪前梯度统计。模型、损失、数据及测试选模协议保持不变。

第五阶段 `feat/neurofdan-phase5-diagnosis` 基于 Phase 4 的真实服务器日志做分级诊断；终端改为每轮紧凑摘要与两组矩阵，完整JSONL/NPZ保留。新增独立 `server-diagnostics` 分支上传工具，不修改模型或训练目标。该阶段没有运行云端测试或模型实验；服务器上传现已用于Phase 4快照和E1结果回传。

E1已在服务器完成。后续独立分支 `feat/neurofdan-e1b-backbone-probe` 提供[骨干逐层只读探针及完整运行/上传命令](docs/e1b_backbone_probe.md)，用于定位输入差异在哪一层消失；不训练，不修改已有checkpoint。

E1b已观察到epoch37卷积核严重收缩与骨干输出变常量。独立分支 `feat/neurofdan-e2-gradient-audit` 提供[E1b证据分析及E2分项梯度/Adam L2只读审计命令](docs/e2_gradient_audit.md)。E2现已在服务器完成，未执行参数更新。

独立分支 `feat/neurofdan-weight-decay-control` 提供[E2结果、局限与5轮耦合L2单变量对照命令](docs/e2_results_and_weight_decay_control.md)。新增 `--weight_decay` 默认仍0.01；显式取0仅供服务器对照，full诊断增加逐参数RMS，禁止用不匹配的衰减配置续训。该入口已用于服务器短程对照，不代表已修复全部退化问题。

W1两组现已在服务器完成5轮：关闭衰减保留了中介参数尺度，但事实分类仍很弱、前门测试输出仍为单类别。独立分支 `feat/neurofdan-mechanism-probe` 提供[W1结果与E3/E4配对机制探针](docs/w1_results_and_mechanism_probe.md)，读取现有权重分离真实输入、中介噪声与参考边缘化，不再训练，不改变核心模型。

E3/E4现已回传：B组输入路径存在，但局部输入影响远弱于采样噪声。参见[结果分析、20轮有限续训与正式实验门槛](docs/e3e4_results_and_training_gate.md)。文档分支 `feat/neurofdan-e3e4-review` 不修改源码；建议在独立副本上复用9d698f3训练工具，新增15轮后复盘，不自动扩展LOSO。

20轮已复盘，尚未达到正式实验推进条件。参见[20轮结果、重复续训报错及训练批次组成审计](docs/epoch20_review_and_batch_audit.md)。独立分支 `feat/neurofdan-batch-composition-audit` 只读标签与原采样器，不运行模型或改变正式训练；优先量化batch40实际只有4个trial的优化影响候选。

批次审计确认分组下690/5200个批次为单类，尚未证明其导致学习失败。独立分支 `feat/neurofdan-sampling-order-control` 提供[采样顺序单变量对照](docs/sampling_order_control.md)：新增 `--sampling_mode`，默认仍grouped，window仅显式改变优化窗口排列；SCM、模型、损失和参考总体不变。两组连续20轮从头对照现已在服务器完成。

采样对照已回传，window减少长期单类轮数但尚未稳定改善分类。用户已选择全局窗口shuffle＋DEAP400/HCI100，见[大batch运行与更新预算说明](docs/large_batch_run.md)。独立分支 `feat/neurofdan-large-batch-run` 增加可选评估分批及批次预算记录；本次命令显式设置训练400/100、评估40，原CLI默认保持兼容。先做GPU smoke，再按独立目录运行，不能将400窗口误称为按被试成批。

四组大batch结果现已回传，工程运行成功但前门判别仍弱。见[结果复盘与固定小批记忆能力检查](docs/large_batch_results_and_capacity_check.md)。独立分支 `feat/neurofdan-fixed-batch-capacity` 只对临时内存模型作有界诊断更新，复用原训练函数，不保存权重、不覆盖正式配置，不将小批记忆等同于因果识别或正式分类效果。

容量检查本次在import torch阶段失败，尚无模型拟合结论。参见[导入错误与不改环境的诊断命令](docs/capacity_import_failure.md)，以及[事实监督、前门推断与正式ACC汇报口径](docs/factual_frontdoor_metrics.md)。分支 `feat/neurofdan-metrics-and-import-audit` 只新增说明与标准库诊断，不改变训练/模型或主指标。

- [第五阶段：真实日志诊断与证据等级](docs/phase5_diagnostic_analysis.md)
- [第五阶段：A/B工程实施与C/D候选及最小服务器探针](docs/phase5_action_plan.md)
- [第五阶段：日志分层、GitHub结果回传与认证说明](docs/phase5_server_upload_guide.md)
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
