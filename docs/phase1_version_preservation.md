# 第一阶段版本保护记录

开始第二阶段时，本地分支为 feat/front-door-scm，HEAD 为 19bf18f052b5c4b9b7395f99c0e8fff7f3ca4264，远端只有 main 和 fix/configurable-paths。第一阶段新增源码当时未被 Git 跟踪，NeuroFDAN.py 的接口导出未提交。

第一阶段源码、测试、实现报告、实验配置及本次测试核验日志现纳入 feat/neurofdan-phase1。原实现报告记录的是第一阶段结束时的历史状态，其中“未提交/推送”不表示此保护分支的发布状态。完整代码以此分支提交为准。

重新运行 unittest 共24项，23项通过，1项因无CUDA跳过，日志见 phase1_test_recheck.txt。没有重新运行完整合成训练。

第一阶段 ZIP 为本地结果归档，包含CSV、图表、生成参数及报告，不包含独有源代码；继续保留在云端，不上传Git。results/ 仍由既有 .gitignore 忽略。必要实验配置已另存 phase1_experiment_config.json；该配置属于历史合成实验，不是第二阶段真实数据训练配置。
