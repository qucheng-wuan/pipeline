# 评测指标与测试指南

当前256评测使用 **δ1.05、δ1.10、δ1.25**，采用严格小于 `<`。
本指南同时说明代码回归测试、已有模型评分和旧结果重算；所有命令从仓库根目录执行。

## 1. 指标定义与输出位置

在每张图的每个有效区域 V 上，令 p 为预测深度、g 为GT、r=max(p/g,g/p)：

| JSON字段 | 计算 | 单位 / 方向 |
|---|---|---|
| `rmse_m` | sqrt(mean_V((p-g)^2)) | 米，越低越好 |
| `abs_rel` | mean_V(abs(p-g)/g)，即REL | 无量纲，越低越好 |
| `mae_m` | mean_V(abs(p-g)) | 米，越低越好 |
| `delta_1.05` | mean_V(r < 1.05) | [0,1]比例，越高越好 |
| `delta_1.10` | mean_V(r < 1.10) | [0,1]比例，越高越好 |
| `delta_1.25` | mean_V(r < 1.25) | [0,1]比例，越高越好 |

δ展示为百分数时才乘100。例如GT=1、预测=1.2，三个δ分别为0、0、1。
MAE/RMSE与旧原生评测的毫米值相比需乘1000；REL无需换算。

实现为 [`evaluate256.py`](../src/transparent_depth/evaluate256.py)，报告schema为
`fullfield256_evaluation_v2`，附带 `delta_thresholds`、`delta_comparison`、`delta_storage`。
报告中的 `overall`、`groups`、`images` 分别保存总体、场景/相机分组和逐图指标；
每层分别包含 `P256` / `Stage2` 的 `all` / `transparent` / `background`。
逐图访问路径为 `images[i].metrics.Stage2.transparent`；总体为 `overall.Stage2.transparent`。
训练验证中的 `metrics256` 使用相同字段；选模继续使用 `transparent_mae_m`。

先计算每图指标，再对非空图等权平均，**RMSE也是每图开根号后再平均**。
空区域跳过并计数，不填零；预测无效时报错，不通过剔除失败像素改善指标。
256网格、GT面积插值、coverage阈值、深度范围和mask规则未改变，
所以公式一致不代表与原生分辨率的整个评测协议等价。

## 2. CPU回归测试

使用已经按[主README](../README.md)安装的两个环境。无需真实数据、模型权重或GPU；
仍需要各环境的torch、NumPy、OpenCV、Pillow、OpenEXR和pytest等依赖。
5090现有环境示例（其他机器替换解释器路径，并按其安装文档设置runtime profile）：

```bash
export S1=/home/QuchengZou/miniconda3/envs/rgbfusion-stage1/bin/python
export S2=/home/QuchengZou/miniconda3/envs/rgbfusion-stage2-5090/bin/python
export TD_RUNTIME_PROFILE=blackwell-cu130
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

# 若环境尚未安装pytest，先补齐测试依赖
"$S1" -m pip install pytest
"$S2" -m pip install pytest

# 推荐：分别在对应环境运行，脚本自动禁用GPU并设置PYTHONPATH
bash scripts/test_cpu.sh "$S1" "$S2"

# 仅运行本次指标修复相关测试
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src "$S2" -m pytest -q \
  tests/test_evaluate256_metrics.py tests/test_rescore256.py

# 在Stage2环境运行整个CPU测试集
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src "$S2" -m pytest -q tests
```

本次版本：指标与重算测试47项；完整测试175项。
`test_cpu.sh` 将其中15项放在Stage1环境运行，另外160项放在Stage2环境运行。

| 测试文件 | 核查内容 |
|---|---|
| [`test_evaluate256_metrics.py`](test_evaluate256_metrics.py) | 六项指标手算和独立参考；三个阈值及相邻浮点数；双向深度比；无效GT/预测；空mask；不同像素数的逐图平均；原native实现对照；训练验证和最终报告输出 |
| [`test_rescore256.py`](test_rescore256.py) | 从旧数组重新计算；身份与manifest哈希；不按文件顺序配对；原误差/像素数不变；拒绝缺失、重复、额外数组；不覆盖原报告 |

2026-11-06在5090验证：Stage2环境全175项通过，并在Stage1环境复核15项通过，
不是190个不同测试。证据：[全量日志](../reports/metric_fix_cpu_pytest.txt)、
[Stage1日志](../reports/metric_fix_stage1_pytest.txt)、[核查记录](../reports/metric_fix_verification.json)。
上述推荐的双环境脚本也已实际执行，通过15+160项，见[脚本运行日志](../reports/metric_fix_split_env_pytest.txt)。
核查记录是服务器修复时的快照，其中 `github_pushed=false` 表示当时尚未发布，不表示当前远端状态。

这些是数学与流程回归，不等于真实GPU端到端验收，也不代表已经重算真实实验成绩。

## 3. 用已有权重生成新评测报告

需要真实checkpoint、已封存的Stage1缓存及对应输入/标签清单。设置 `WORK` 为该实验目录，
确保来源哈希匹配；不修改checkpoint合同或关闭检查。

```bash
export WORK=/path/to/experiment
PYTHONPATH=src "$S2" -m transparent_depth.evaluate256 \
  --checkpoint "$WORK/training/best.pt" \
  --input-manifest "$WORK/prepared/manifests/input_manifest.jsonl" \
  --label-manifest "$WORK/prepared/manifests/label_manifest.jsonl" \
  --stage1-root "$WORK/prepared/stage1" \
  --dataset cleargrasp --role evaluation --device cuda:0 \
  --output "$WORK/evaluation-delta105-110-125"
```

结果在 `evaluation-delta105-110-125/metrics.json`，预测和监督分别保存在
`predictions/`、`supervision/`；输出目录必须不存在或为空。
检查 `overall.Stage2.transparent` 等区域的三个明确δ字段。
如评估transCG内部验证集，使用 `--dataset transcg --role val` 和其对应清单。
ClearGrasp仍保留测试集选模披露，不能将结果称为独立盲测。

## 4. 重算旧256报告，不重跑模型

旧 `delta_1/2/3` 使用1.25、1.5625、1.953125。只有旧 `delta_1` 对应现在的 `delta_1.25`；
不能把三列直接改名，也不能从三个旧汇总值反推δ1.05和δ1.10。
有完整的旧 `metrics.json`、`predictions/*.npz`、`supervision/*.npz` 时：

```bash
PYTHONPATH=src "$S2" -m transparent_depth.rescore256 \
  --input /path/to/old-evaluation \
  --input-manifest /path/to/original/input_manifest.jsonl \
  --output /path/to/new-evaluation-delta105-110-125
```

旧报告无cache_id，必须提供与原报告SHA256匹配的input manifest；新报告有cache_id时可省略。
新目录必须不存在，旧文件不变。工具核验每图及分组的MAE/RMSE/REL、像素数和身份，
记录来源报告、数组和源码哈希，并输出新的 `metrics.json`。旧数组最初未封存哈希的限制会明确记录。

本次服务器未找到可重算的真实256评测数组，原报告保留；其他机器上的旧结果需按上述方法重算。
若只剩汇总数字，需要找回原数组或用原权重重新推理，不能补造结果。
仅修正δ不需要重训，也不改变MAE选出的best；旧训练续跑仍须满足原始源码contract，不能绕过校验。
