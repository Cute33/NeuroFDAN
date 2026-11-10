# Phase 4 实际轻量测试结果

日期：2026-11-10。全部在CPU云环境执行，无真实DEAP/HCI数据。测试夹具、小型随机窗口和oracle数值单测不代表正式实验；本轮没有运行真实数据训练、GPU训练、完整LOSO或完整合成训练。

## 全套单元测试

在现有可用Python环境执行：

```text
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
Ran 71 tests in 35.417s
OK (skipped=1)
```

70项通过，1项CUDA测试因环境无GPU跳过。包含原57项及新增14项；完整测试用例与状态如下。

| 测试用例 | 结果 |
|---|---|
| `test_frontdoor_scm.FrontDoorTests.test_ans_no_cns_features_or_current_input_argument` | 通过 |
| `test_frontdoor_scm.FrontDoorTests.test_bank_heldout_duplicates_and_empty` | 通过 |
| `test_frontdoor_scm.FrontDoorTests.test_bank_pairing_exclusion_and_provenance` | 通过 |
| `test_frontdoor_scm.FrontDoorTests.test_cns_ans_ablation_eliminates_path` | 通过 |
| `test_frontdoor_scm.FrontDoorTests.test_cns_current_reaches_lif_spikes` | 通过 |
| `test_frontdoor_scm.FrontDoorTests.test_complete_gnn_feature_interface` | 通过 |
| `test_frontdoor_scm.FrontDoorTests.test_conditional_ans_and_gradient` | 通过 |
| `test_frontdoor_scm.FrontDoorTests.test_cuda_forward_backward` | 跳过：无CUDA |
| `test_frontdoor_scm.FrontDoorTests.test_do_head_isolated_from_current_input` | 通过 |
| `test_frontdoor_scm.FrontDoorTests.test_full_reference_shapes_and_no_pooling` | 通过 |
| `test_frontdoor_scm.FrontDoorTests.test_hierarchical_shapes_and_clamp` | 通过 |
| `test_frontdoor_scm.FrontDoorTests.test_invalid_shapes_and_singleton_training` | 通过 |
| `test_frontdoor_scm.FrontDoorTests.test_kl_axes_and_oracle_opt_in` | 通过 |
| `test_frontdoor_scm.FrontDoorTests.test_none_never_samples_reference_bank` | 通过 |
| `test_frontdoor_scm.FrontDoorTests.test_outcome_inputs_and_nondegenerate_dependence` | 通过 |
| `test_frontdoor_scm.FrontDoorTests.test_projection_gradient_and_optimizer_step_cpu` | 通过 |
| `test_frontdoor_scm.FrontDoorTests.test_softmax_before_two_means` | 通过 |
| `test_frontdoor_scm.FrontDoorTests.test_tanh_density_matches_transformed_distribution` | 通过 |
| `test_frontdoor_scm.FrontDoorTests.test_tanh_normal_likelihood_and_support` | 通过 |
| `test_frontdoor_scm.OracleTests.test_convergence_in_repeated_runs` | 通过 |
| `test_frontdoor_scm.OracleTests.test_gamma_oracle_direct_agreement` | 通过 |
| `test_frontdoor_scm.OracleTests.test_invalid_point_shapes_are_rejected` | 通过 |
| `test_frontdoor_scm.OracleTests.test_oracle_independent_of_direct_function` | 通过 |
| `test_frontdoor_scm.OracleTests.test_outcome_conditional_independence_moments` | 通过 |
| `test_phase2_integration.DatasetTests.test_balance_keeps_pairing_and_unbalanced_population` | 通过 |
| `test_phase2_integration.DatasetTests.test_deap_balance_uses_task_labels_not_arousal` | 通过 |
| `test_phase2_integration.DatasetTests.test_deap_dimensions_baseline_windows_and_threshold` | 通过 |
| `test_phase2_integration.DatasetTests.test_hci_windows_labels_metadata` | 通过 |
| `test_phase2_integration.DatasetTests.test_loader_single_instance_sources_and_drop_last` | 通过 |
| `test_phase2_integration.DatasetTests.test_normalization_local_axis_and_constant` | 通过 |
| `test_phase2_integration.DatasetTests.test_one_batch_fold_cpu_checkpoint_no_formal_results` | 通过 |
| `test_phase2_integration.DatasetTests.test_reference_population_not_balanced_and_no_test` | 通过 |
| `test_phase2_integration.DatasetTests.test_valence_forwarded_to_compatibility_constructor` | 通过 |
| `test_phase2_integration.ModelTests.test_backbone_preserved_and_batchnorm_frozen` | 通过 |
| `test_phase2_integration.ModelTests.test_both_dataset_feature_dimensions_and_full_output` | 通过 |
| `test_phase2_integration.ModelTests.test_cached_reference_expiry_and_cpu_eval` | 通过 |
| `test_phase2_integration.ModelTests.test_forward_backward_optimizer_step_and_fact_loss` | 通过 |
| `test_phase2_integration.ModelTests.test_legacy_ce_constructor_fixed_without_weight_changes` | 通过 |
| `test_phase2_integration.ModelTests.test_live_reference_backbone_not_detached` | 通过 |
| `test_phase2_integration.ModelTests.test_live_reference_gradient_and_conditional_path` | 通过 |
| `test_phase2_integration.ModelTests.test_no_cns_path_and_none_modes_remain_explicit` | 通过 |
| `test_phase2_integration.ModelTests.test_reference_chunking_preserves_probability_formula` | 通过 |
| `test_phase2_integration.ModelTests.test_seed_before_initialization_and_no_unconditional_cuda` | 通过 |
| `test_phase2_integration.ModelTests.test_single_model_definition_and_no_prototype_path` | 通过 |
| `test_phase2_integration.ProtocolTests.test_checkpoint_acc_gate_and_dataset_specific_f1_tie` | 通过 |
| `test_phase2_integration.ProtocolTests.test_fold_matrix_updates_preserve_other_subjects` | 通过 |
| `test_phase2_integration.ProtocolTests.test_results_ignore_zero_unrun_folds` | 通过 |
| `test_phase3_support.ResultAuditTests.test_bad_probability_and_configuration_mismatch` | 通过 |
| `test_phase3_support.ResultAuditTests.test_cli_require_complete_exit_and_no_stale_export` | 通过 |
| `test_phase3_support.ResultAuditTests.test_missing_fold_and_incomplete_are_distinct` | 通过 |
| `test_phase3_support.ResultAuditTests.test_no_best_not_silently_averaged_as_full` | 通过 |
| `test_phase3_support.ResultAuditTests.test_resume_stale_summary_is_not_completed` | 通过 |
| `test_phase3_support.ResultAuditTests.test_smoke_and_reference_leak_excluded` | 通过 |
| `test_phase3_support.ResultAuditTests.test_specs_match_loader_and_complete_export` | 通过 |
| `test_phase3_support.RuntimeSupportTests.test_documented_cli_commands_parse` | 通过 |
| `test_phase3_support.RuntimeSupportTests.test_logger_preserves_python_and_tee_failures` | 通过 |
| `test_phase3_support.RuntimeSupportTests.test_primary_probability_guard_preserves_valid_gradient` | 通过 |
| `test_phase4_diagnostics.EvaluationTests.test_basic_and_full_have_same_primary_outputs_and_loss_update` | 通过 |
| `test_phase4_diagnostics.EvaluationTests.test_full_fold_files_and_legacy_summary_history_compatibility` | 通过 |
| `test_phase4_diagnostics.EvaluationTests.test_full_gradient_diagnostics_before_step_and_no_structure_change` | 通过 |
| `test_phase4_diagnostics.EvaluationTests.test_repeated_cpu_eval_and_extra_eval_do_not_change_test_prediction` | 通过 |
| `test_phase4_diagnostics.EvaluationTests.test_train_eval_parameters_bn_optimizer_modes_rng_unchanged` | 通过 |
| `test_phase4_diagnostics.MediatorTests.test_full_stream_statistics_not_batch_average_and_class_distance` | 通过 |
| `test_phase4_diagnostics.MediatorTests.test_mean_variation_separate_from_posterior_and_sampling` | 通过 |
| `test_phase4_diagnostics.MediatorTests.test_single_sample_and_single_draw_are_finite_not_evidence_of_collapse` | 通过 |
| `test_phase4_diagnostics.MetricTests.test_checkpoint_rule_unchanged_by_train_and_diagnostics` | 通过 |
| `test_phase4_diagnostics.MetricTests.test_extreme_imbalance_and_factual_disagreement` | 通过 |
| `test_phase4_diagnostics.MetricTests.test_matrix_order_and_both_positive_classes` | 通过 |
| `test_phase4_diagnostics.MetricTests.test_single_class_zero_and_one_and_reported_candidates` | 通过 |
| `test_phase4_diagnostics.PersistenceTests.test_jsonl_npz_resume_streak_and_no_mutation` | 通过 |
| `test_phase4_diagnostics.PersistenceTests.test_server_guide_cli_and_bash_syntax` | 通过 |

## 额外核验

- 最后一次诊断代码与文档调整后，新增14项再次运行：Ran 14 tests in 1.755s，OK。
- 只读加载Git中Phase 3原ModelTrain.evaluate，在相同5窗口CPU夹具上与Phase 4 full比较：全部旧指标、逐窗口概率、混淆矩阵完全一致。
- Python编译、核心模块import、两个CLI --help、git diff --check通过。
- 两阶段服务器指南中的Bash代码块通过语法检查；Phase 4四条训练命令通过真实argparse解析。没有在云端执行这些服务器命令。
- 对照Phase 3，模型、前门核心、数据、损失定义、结果审计与依赖文件内容保持不变；CLI原参数默认值保持不变，唯一新增参数为diagnostic_level=basic。

新增测试先发现并纠正了测试夹具未补齐factual指标的问题；也纠正了仅凭surrogate返回None就认为阈值无梯度的静态推断，复位项仍有梯度路径。最终全套通过，不据此推断真实服务器异常已解决。

未验证：服务器实际标签与预测分布、GPU/驱动、正式显存与耗时、真实塌缩原因、完整训练及LOSO结果。需按服务器指南运行并反馈。
