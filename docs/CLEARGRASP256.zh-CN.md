# ClearGrasp 256×256：train训练，real-val测试集选最优权重

按最新要求，**不使用 synthetic-val，也不从训练集另划 val**。正式配置
`configs/cleargrasp256_stage2_60.json` 只训练 ClearGrasp train；每轮对 real-val 做纯 P 评估，
在全部第1–60轮中按 **256网格、逐图平均透明区域 MAE（米）最小**保存 `best.pt`。
严格改善才更新，分数相同保留较早轮次；`last.pt` 用于断点续训。
**这是按测试集选权重（test-set selection），real-val 分数不能称为独立未见测试性能。**
不固定交付第60轮，不用 real-val 的 GT/mask 控制模型输入/query，也不用其分数调整学习率。
原有 transCG256 与历史 ClearGrasp 240×320 配置保留，独立目录运行。

最新发布检查见 `reports/publication_verification.json`。`reports/cleargrasp256_acceptance.json` 是选模策略修改前的数据/阻塞证据，
其中旧 final.pt 描述已被本协议取代，不能把 CPU 单元测试当作 GPU 跑通。

本机存在三个具体阻塞：13656张训练图的JSON同时缺少 `image` 和 `camera.field_of_view`；
Stage1缺DINOv3的model/config/preprocessor三个文件；5090现有cu130环境在官方SPVNAS拓扑初始化
稀疏卷积中发生CUDA非法内存访问。没有用历史默认FOV补K，也没有替换权重或绕过源码/缓存检查。
目标A100需按下述参考工具链重新构建并验收；当前结果不证明A100已通过。

## 数据、几何与监督

训练数据按相对目录身份＋数字帧号严格匹配 RGB JPG、rectified depth EXR、segmentation mask PNG、
逐帧 masks JSON。四个集合必须相同，缺文件/损坏会记录错误，不按排序拼接或默默丢弃。
JSON 的 `image.width_px/height_px` 必须与 RGB/GT/mask 一致；
`fx=W/(2*tan(fov_x/2))`、`fy=H/(2*tan(fov_y/2))`，FOV 为弧度。
渲染器边界中心 `(W/2,H/2)` 转为 integer pixel centres：`cx=(W-1)/2, cy=(H-1)/2`。
不使用固定原始尺寸、固定 K 或缺失 metadata 时的默认 FOV。

real-val 递归按目录＋帧号匹配 transparent RGB、opaque EXR 和 mask，读取对应 YAML 并核验尺寸；
不读 transparent-depth 或 output-depth。两类 EXR 均已是米制 optical-Z，不做单位缩放或 radial-to-Z。
mask 必须为灰度 0/255。原始数据只读，rosters 仅引用路径，不复制全套 GT。

冻结 Stage1 读取原始高分辨率 RGB，沿用原内部预处理和真实专家/V2/Laplacian E08 权重，生成
同尺寸完整视野 P，长期保留 native P。Stage2 完整视野 resize 到 **256×256**，无裁剪/补边：

* RGB：`INTER_AREA`；P：`INTER_LINEAR`；mask：像素中心 nearest-exact 后转 bool。
* GT：`V=isfinite(GT)&(GT>0)`，`coverage=AREA(V)`，`GT256=AREA(where(V,GT,0))/max(coverage,1e-8)`。
  有效区为 coverage≥0.5，再与预先固定的 **[0.1,2.0] 米闭区间**相交。训练和最终评估相同，
  不按 real-val 结果调整。保存 float32 coverage；无效区不参与任何监督，不填孔、不逐图归一化。
* `sx=256/W, sy=256/H`；`fx'=sx*fx, fy'=sy*fy`；
  `cx'=sx*(cx+0.5)-0.5, cy'=sy*(cy+0.5)-0.5`。ray、点云、法线、体素共用 K256，只缩放一次。
* 法线继续使用原 LIDF 损失：相机 x右/y下/z前，由 GT256/K256 得到点云，取 `dx×dy`；
  当前、右、下邻域均有效才监督，屏蔽图像边界和退化向量并归一化。可选 cameraNormals EXR 不作为
  新损失输入，不用重采样其向量代替原有语义。网络、损失公式/权重、SPVNAS **0.05 m voxel** 不变。

审计逐帧解码 RGB/mask，检查 EXR 尺寸、R 通道、chunk 完整性并记录各文件 SHA256；同时检查
压缩文件和解码像素两种 RGB 重复，跨 train/real-val 重复阻止生成正式 roster。
`audit --decode-all` 额外解压全部 GT 像素；默认审计不声称完成所有 GT 压缩载荷的像素解码。
smoke 始终完整解码其 GT，正式 Stage2 label cache 构建也必须完整解码每张 GT，错误即停止。
带错误的全量 audit 不能生成正式 roster；显式 `roster --smoke` 仅可选已核验帧，并保留全量错误状态。

历史名单训练数45453与当前45454的差异是
`square-plastic-bottle-train/000004501-rgb.jpg`，不会沿用历史数字自动删除该帧。
该RGB的SHA256与历史 synthetic-val 的 `square-plastic-bottle-val/000000114-rgb.jpg` 相同，
历史 `provenance/stage1_historical/tomnet_full_v1.json` 明确记录了这对重复；该训练帧当前解码和标定正常。
本次完全不使用synthetic-val，保留这张训练图，仍严格检查train与real-val之间的RGB重复。
数量从实际 manifest 推导，逐帧结果和历史名单差异保存在 audit 报告中。
缺标定分布为 heart-bath-bomb 4461、square-plastic-bottle 4501、stemless-plastic-champagne-glass 4694。
必须补齐有来源的逐帧标定后重新审计，正式roster不会自动舍弃这些帧。
冻结 Stage1 历史训练/选模接触过 ClearGrasp，尤其 real-val 的173帧与历史验证名单存在 SHA256 交集；
本实验的 real-val 不进入梯度训练，但用于每轮选模；不能称整个 pipeline 对 real-val 完全未见。

## 环境、权重和迁移

当前服务器使用 `conda rgbfusion-stage1` 及仓库 `.setup/activate-stage2.sh`；不停止其他任务。
A100 上安装两个独立环境，复制仓库源文件、vendor、provenance 和数据引用；不依赖当前 `.setup/`、
旧项目、临时目录、JIT cache 或5090的 `.so`。完整步骤也见 `TRANSCG256_A100.zh-CN.md`。

```bash
# 仓库根目录，Stage1：Python3.11 / torch2.8.0+cu128 / torchvision0.23.0+cu128
bash scripts/setup_stage1.sh "$PWD/.venv-stage1"
export S1="$PWD/.venv-stage1/bin/python"
"$S1" -m pip install --no-deps -e .

# Stage2 A100：Python3.10 / torch2.1.2+cu121 / torchvision0.16.2+cu121
conda create -n rgbfusion-stage2-a100 python=3.10 -y
conda activate rgbfusion-stage2-a100
export S2="$CONDA_PREFIX/bin/python" TD_RUNTIME_PROFILE=reference
export CUDA_HOME=/usr/local/cuda-12.1 CC=gcc-11 CXX=g++-11
export TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=2
export TORCH_EXTENSIONS_DIR="$PWD/outputs/runtime/stage2-cu121-sm80"
bash scripts/install_stage2.sh
"$S2" -m pip install pytest
"$S2" -m pytest -q tests

export ASSETS=/path/to/stage1-assets INIT=/path/to/stage2-initialization
# 先完成已获授权的 Hugging Face 登录，补齐 gated DINOv3；不替换变体/版本。
"$S1" scripts/fetch_assets.py --manifest provenance/release_assets.json --component stage1 --asset-root "$ASSETS"
"$S2" scripts/fetch_stage2_initialization.py --output "$INIT"
"$S2" scripts/verify_transfer_assets.py --asset-root "$ASSETS" --init-root "$INIT"
```

完整依赖在 `requirements-stage1.txt` 和 `requirements/stage2.txt`。
TorchSparse 固定 commit `385f5ce8718fcae93540511b7f5832f4e71fd835`，安装器针对目标架构源码编译；
LIDF C++/CUDA 扩展从仓库源码 JIT 构建；scatter 使用匹配 PyTorch/CUDA 的官方包。
启动检查 ABI、GPU 架构、nvcc/ninja 与源码摘要，实际 kernel 仍需在目标机执行 smoke。
换架构/工具链要重新编译并使用新 JIT 目录，旧 DDP 限制仍保留，本任务只验单 GPU。

可单独复现真实初始化兼容性检查（官方构造器含拓扑dummy forward，不是pipeline smoke）：

```bash
CUDA_LAUNCH_BLOCKING=1 "$S2" scripts/check_stage2_construction.py \
  --assets-root "$INIT" --output /path/to/new-stage2-construction-check
```

Stage2 fresh 初始化为已核验 ImageNet ResNet34 + 官方 SPVNAS20G，LIDF heads/adapters 新初始化；
不加载旧 ClearGrasp Stage2 最佳权重。`--resume` 才恢复本次模型、Adam、StepLR、RNG、更新计数。

## audit → roster → native Stage1 predict/seal → Stage2 cache → smoke

```bash
export CLEARGRASP_TRAIN_ROOT=/path/to/cleargrasp-dataset-train
export CLEARGRASP_REAL_VAL_ROOT=/path/to/cleargrasp-dataset-test-val/real-val
export PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=0
export WORK=/path/to/cleargrasp256-smoke AUDIT=/path/to/cleargrasp256-audit

"$S2" -m transparent_depth.prepare_cleargrasp256 audit --output "$AUDIT" --workers 2
# 需要检查所有GT压缩载荷时，在新AUDIT目录追加 --decode-all。
"$S2" -m transparent_depth.prepare_cleargrasp256 roster --audit-root "$AUDIT" \
  --config configs/cleargrasp256_stage2_smoke.json --smoke --output "$WORK/rosters"

export CUDA_HOME=/usr/local/cuda-12.8  # Stage1工具链，按目标机器实际路径修改
"$S1" -m transparent_depth.stage1_batch predict --manifest "$WORK/rosters/rgb.jsonl" \
  --asset-root "$ASSETS" --output "$WORK/native-P" --cache-root "$WORK/stage1-jit-sm80"
"$S1" -m transparent_depth.stage1_batch seal --manifest "$WORK/rosters/rgb.jsonl" \
  --asset-root "$ASSETS" --output "$WORK/native-P"
export CUDA_HOME=/usr/local/cuda-12.1
"$S2" -m transparent_depth.prepare_stage2_data --config configs/cleargrasp256_stage2_smoke.json \
  --input-roster "$WORK/rosters/inputs.jsonl" --label-roster "$WORK/rosters/labels.jsonl" \
  --predictions "$WORK/native-P/predictions.jsonl" --output "$WORK/prepared"
"$S2" scripts/run_transfer_smoke.py --config configs/cleargrasp256_stage2_smoke.json \
  --formal-config configs/cleargrasp256_stage2_60.json --prepared-root "$WORK/prepared" \
  --assets-root "$INIT" --output "$WORK/acceptance"
```

新目录避免覆盖其它实验。RGB-only、RGB/K、监督各自独立 roster；只有真实 native P/seal 能进入 cache。
4帧 smoke：两张训练图覆盖1920×1080与1024×576，两张 real-val 图；没有 synthetic-val。
独立4轮配置只做8次 optimizer 更新，alpha 为 `1,1,1,2/3,1/3,0,0,0`；第2轮保存后由新进程
恢复并继续，校验模型/优化器/调度/RNG精确恢复、输出/梯度有限及各可训练参数组实际变化。
同时断言正式20/20/20边界。每轮纯P评估 real-val 并选 best；结束后对 best 做详细256评估/可视化，成功才写 acceptance.json。
数据图在 `rosters/visualizations`，真实预测图在 `acceptance/cleargrasp-evaluation/visualizations`。

## 正式 train / resume / 最终 real-val（本次不运行）

正式阶段在新的 WORK 下使用正式配置，去掉 roster 的 `--smoke`，再执行上述 predict/seal/cache。
这是迁移后启动正式实验时才生成的全量 P，当前只准备少量真实样本。

```bash
export WORK=/path/to/cleargrasp256-full
"$S2" -m transparent_depth.prepare_cleargrasp256 roster --audit-root "$AUDIT" \
  --config configs/cleargrasp256_stage2_60.json --output "$WORK/rosters"
# 对本WORK执行上节predict/seal/cache；cache配置改为cleargrasp256_stage2_60.json。
"$S2" -m transparent_depth.train_stage2 --config configs/cleargrasp256_stage2_60.json \
  --input-manifest "$WORK/prepared/manifests/input_manifest.jsonl" \
  --label-manifest "$WORK/prepared/manifests/label_manifest.jsonl" \
  --stage1-root "$WORK/prepared/stage1" --label-root "$WORK/prepared/labels" \
  --assets-root "$INIT" --output "$WORK/training"
# 恢复：同一条训练命令追加 --resume，保持代码/配置/manifest/world size/ABI不变。

"$S2" -m transparent_depth.evaluate256 --checkpoint "$WORK/training/best.pt" \
  --input-manifest "$WORK/prepared/manifests/input_manifest.jsonl" \
  --label-manifest "$WORK/prepared/manifests/label_manifest.jsonl" \
  --stage1-root "$WORK/prepared/stage1" --dataset cleargrasp --role evaluation \
  --output "$WORK/final-real-val"
```

第1–20轮有效非透明背景用 GT，第21–40轮按已提交 optimizer updates 连续 alpha=1→0（含端点），
第41–60轮纯P。透明区/无效GT始终P；无随机几何增强。batch=2、accumulation=5、单GPU有效batch=10。
Adam基础LR=1e-3，保留组倍率 lidf/spvnas=0.1、adapters=1；StepLR每20轮衰减0.1，
第1–20/21–40/41–60轮基础LR分别1e-3/1e-4/1e-5。此调度不读取测试分数。
保存每轮、`last.pt` 和指标最优 `best.pt`。即使某轮训练使用 GT 背景，该轮选模推理仍始终纯P。
全部60轮参与选模；checkpoint、contract和评估报告记录 `test_set_selection=true`，公开最佳epoch及指标。

评估直接在256网格同时比较P256与D256；全有效/透明/背景各报MAE/RMSE（米）、AbsRel、
δ1/δ2/δ3（`max(P/G,G/P)<1.25^k`）。逐图先计算再对非空图等权平均，RMSE也是逐图RMSE平均；
空区不记零误差，显式统计空图和有效像素，按真实可用目录场景/相机分组。两模型使用相同GT/mask/有效区，
不作GT尺度拟合或原图上采样指标。GT/mask只在模型完成纯P全图query预测后用于评分。

## 缓存、bundle 与迁移核验

选模策略和准备源码已改变，旧审计/roster/cache不能直接继续；先在新目录重新audit/roster，再prepare。
清单路径用 `${CLEARGRASP_TRAIN_ROOT}` / `${CLEARGRASP_REAL_VAL_ROOT}`；迁移重绑定环境变量。
审计/划分、源图和标定SHA、原始尺寸、预处理协议、Stage1权重和producer seal共同绑定缓存。
本次新增独立 `rgb_fusion_cleargrasp_fullfield256_v1`，不能与transCG或240×320缓存混用。
公共源码变动会令旧Stage2 cache/checkpoint/bundle校验失败；用原真实native P重新prepare到新目录，
不要改/删安全检查。Stage1源/权重改变则重新predict/seal。

可调用 `transparent_depth.stage2.export_stage2_bundle(best_checkpoint, output, selection_metadata)`，
selection_metadata说明“real-val测试集逐轮选模，透明MAE最小，非独立测试”；bundle保留训练contract、分辨率/预处理和源/权重哈希。
`python -m transparent_depth.stage2 --help` 的文件入口根据bundle选择网格；新bundle输出D256.npy，
旧240×320入口保留历史行为。目标GPU实际推理、反向和恢复必须以新的真实smoke报告为准。
