# 验证记录

2026-11-06 的256指标修正见 `metric_fix_verification.json`：δ阈值恢复为1.05、1.10、1.25，
新字段为 `delta_1.05`、`delta_1.10`、`delta_1.25`（比例）；最终报告schema升级为v2。
旧256的 `delta_1/2/3` 为1.25、1.5625、1.953125，不能改列名冒充新结果。
本次保留所有历史报告、训练contract和来源快照。需用原预测与GT重新评分的目录及缺失项见该验证记录；
离线重算命令见主README，不需要重新训练，也不改透明MAE选模。

本次修正前的发布状态见 `publication_verification.json`、`publication_cpu_pytest.txt` 和
`../provenance/publication_source_snapshot.json`。这不是Stage1预测seal，也不宣称真实GPU训练已通过。

`cleargrasp256_acceptance.json`及其引用的source snapshot、测试日志保存的是先前固定末轮策略下的数据检查与GPU阻塞证据。
它的 `latest_user_protocol` 是当时协议，不是当前协议。当前代码已改为每轮real-val透明MAE选 `best.pt`，旧报告不会被重写成新GPU验收。
选模策略/准备源码改变后，应重新生成audit/roster/Stage2 cache；不要修改旧证据哈希。

`transcg256_acceptance.json`保存先前transCG数据验收结果。
其他E58、240×320、80轮曲线和数值属于历史实验，不能作为新256训练成绩。

原始数据及数据可视化PNG不随本次代码发布；审计保留帧身份、统计、文件摘要与缺标定清单。
迁移后按两份协议文档重建smoke数据图；真实预测可视化只有真实Stage1+Stage2运行成功后才能生成。
