# transCG 256 → ClearGrasp real-val，A100 迁移

本路径默认只训练 Stage2，正式配置为 `configs/transcg_stage2_60.json`；小样本配置为
`configs/transcg_stage2_smoke.json`。本次按用户后续指示不占用已有 GPU 任务，未启动真实
Stage1/Stage2 GPU smoke，也未启动正式训练。实际验收见 `reports/transcg256_acceptance.json`。
旧 `configs/stage2_train.json` / `stage2_train_8gpu.json` 继续表示历史 ClearGrasp 240×320 协议。

## 固定协议

* 原始 RGB（本次样本 720×1280）→ 原有冻结专家/V2/Laplacian E08 → 同尺寸完整视野 P。
  Stage1 的内部预处理、数值精度及权重不变。持久保存 native P，随后完整视野缩到 256×256。
* RGB：OpenCV `INTER_AREA`；P：`INTER_LINEAR`；透明 mask：按像素中心的 nearest-exact，转 bool。
  不裁剪、不补边、不随机几何增强，不对深度乘图像缩放系数或逐图归一化。
* GT 为 float32 米制 optical-Z。`V=isfinite(GT)&(GT>0)`，分别 area resize `V` 与
  `where(V,GT,0)`，相除得到 GT256，保存 float32 coverage。有效区为 coverage≥0.5，
  再与固定 **[0.1,2.0] 米闭区间**求交。该范围在本次小样本结果之前固定，训练/验证/最终评估共用。
  未达覆盖阈值的深度数值保留供审计，但不进入任何监督；不填补 GT 孔洞。
* 原始 K 是 integer pixel centres；`fx'=sx*fx`、`fy'=sy*fy`、
  `cx'=sx*(cx+0.5)-0.5`、`cy'=sy*(cy+0.5)-0.5`。只缩放一次；ray、点云、法线、体素共用 K256。
* 法线沿用原 LIDF 损失：相机坐标 x 向右、y 向下、z 向前；由 GT256/K256 的点云作
  `dx × dy`，监督需当前/右/下三个像素均有效，并屏蔽边界和退化法线。
  网络、损失公式、权重及 SPVNAS **0.05 m voxel** 未改；LIDF 自身的原有 volume/grid 也未改。

transCG 同时支持官方 `scene*/` 平铺目录和 `transcg-*/scene*/` 分卷目录；重复scene身份会报错。
transCG 使用每个 scene metadata 的 D435/L515 有效视角列表，camera1=D435，camera2=L515，
各相机是独立样本；仅 `rgb{c}.png` 是图像输入。监督为 `depth{c}-gt.png`，分别除以
1000/4000；mask 必须为 0/1。Stage1 访问守卫额外拒绝 `depth1.png`、`depth2.png` 和 GT PNG。
官方采集器将深度对齐到 color；GT renderer 使用对应相机 K；归档 K 与采集器中的两份 K
逐元素一致。来源、归档 SHA256、源代码 SHA256 位于 `data/transcg_info/source.json`。
原始数据目录只读，补充 metadata 放在仓库中。

官方 train 78 场景经 Python `Random(23).shuffle(sorted(train))`，取 round(10%)=8 个场景为
内部 val；具体名单及每个 scene metadata 哈希保存于 `data/transcg_info/split_seed23.json`。
完整枚举得到 train **30,899** / val **3,292** / 官方 test **23,524** 个相机视图。
这些数字仅是本次枚举结果；训练代码从实际 manifest 推导数量、loader 长度和 optimizer steps。
官方 test 的 52 个场景不进入训练/选模，可用 `--include-official-test` 纳入独立 evaluation roster。

ClearGrasp real-val 递归按“相对目录+数字帧号”严格匹配 transparent RGB、opaque GT 和 mask；
对应 YAML 的 xres/yres 必须吻合原图。GT EXR 已是米制 optical-Z，不再次换算；mask 校验 0/255。
本机共 173 帧，全部只作本次 Stage2 最终评估。**冻结 V2/Laplacian 历史选模使用的 ClearGrasp
validation 包含这些 real-val 图像**（历史 704-image validation，real-val 173 帧）；因此不得将
最终结果称为整个 pipeline 完全未接触过的 held-out test。

参考实现已在 CVPR commit `1c280d11df64f534591947c1f7724bcff5be88ee` 核对，见
`provenance/transcg256_references.json`。采用严格配对、物理单位、半像素 K 和有效覆盖积分思路；
本协议没有采用其它项目的 Unity 编码、中心裁剪、bilinear GT 或内参。

## A100 环境与权重

复制整个仓库（包括 `third_party/`、`src/v2_lidf/vendor/`、`data/transcg_info/`、`provenance/`）。
不要复制 `.setup/`、`.deps/`、旧环境、JIT cache 或编译好的 `.so` 作为安装方案。
系统驱动需支持两个独立环境的 CUDA runtime。Stage1 用 Python3.11 / torch2.8.0+cu128，
Stage2 A100 用 Python3.10 / torch2.1.2+cu121，绝不能混在同一个 Python 环境里。

```bash
# 在迁移后的仓库根目录；先准备 Python3.11 与 CUDA12.8 工具链
bash scripts/setup_stage1.sh "$PWD/.venv-stage1"
export S1="$PWD/.venv-stage1/bin/python"
"$S1" -m pip install --no-deps -e .

conda create -n rgbfusion-stage2-a100 python=3.10 -y
conda activate rgbfusion-stage2-a100
export S2="$CONDA_PREFIX/bin/python"
export TD_RUNTIME_PROFILE=reference
export CUDA_HOME=/usr/local/cuda-12.1   # 改成目标机的 toolkit 路径
export CC=gcc-11 CXX=g++-11 TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=2
export TORCH_EXTENSIONS_DIR="$PWD/outputs/runtime/stage2-cu121-sm80"
bash scripts/install_stage2.sh
"$S2" -m pip install pytest
"$S2" -m pytest -q tests
```

安装器固定 sparsehash / TorchSparse commit，针对 `TORCH_CUDA_ARCH_LIST=8.0` 本地编译 TorchSparse；
LIDF ray_aabb / pcl_aabb 从随仓库分发的 C++/CUDA 源码 JIT 编译。torch-scatter 使用匹配
PyTorch/CUDA 的官方 wheel（不是 5090 环境里的 `.so`）。依赖见 `requirements/stage2.txt`。
启动模型会校验 torch/torchvision/scatter ABI、TorchSparse 源码摘要、GPU 架构、nvcc 和 ninja；
实际 CUDA 算子还必须通过下面的 smoke。更换 GPU/toolkit 时使用新的 JIT/cache 目录并重新编译。
可选 CUDA13 profile 的依赖在 `requirements/stage2-cu130.txt`，不作为本次 A100 默认方案。
旧 DDP 版本/单 bucket 安全检查保留；本交付只支持验收单 GPU，不声称已验证多 GPU。

```bash
export ASSETS=/path/to/model-assets
export INIT=/path/to/stage2-initialization
# Stage1 包含 gated DINOv3：先在服务器完成已获授权账户的 Hugging Face 登录。
# 不替换为其它 DINO 变体或其它 revision；全部文件逐项核验 SHA256。
"$S1" scripts/fetch_assets.py --manifest provenance/release_assets.json \
  --component stage1 --asset-root "$ASSETS"
"$S2" scripts/fetch_stage2_initialization.py --output "$INIT"
"$S2" scripts/verify_transfer_assets.py --asset-root "$ASSETS" --init-root "$INIT"
```

本机已下载并核验其余 Stage1 文件和官方 Stage2 初始化文件；DINOv3 的 model/config/preprocessor
三项仍缺失，下载返回 401。可以拷贝已下载的 `artifacts/stage1/`、`artifacts/stage2_init/`，
在目标机补齐缺项并再次核验。文件清单与固定哈希在 `provenance/stage1_assets.json`。
Stage2 fresh 初始化为官方 ImageNet ResNet34 + SPVNAS20G，LIDF heads/adapter 新初始化；
本路径不载入历史 ClearGrasp Stage2 checkpoint。`--resume` 才恢复模型、Adam、StepLR、RNG 和更新计数。

## 六帧 smoke：准备 → native P/seal → 256 cache → 更新/恢复/评估

```bash
export TRANSCG_ROOT=/path/to/transcg-dataset
export CLEARGRASP_ROOT=/path/to/cleargrasp-dataset-test-val/real-val
export WORK=/path/to/transcg256-smoke
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD/src"

"$S2" -m transparent_depth.prepare_transfer_data \
  --config configs/transcg_stage2_smoke.json --smoke --output "$WORK/rosters"

# Stage1 必须读取原始分辨率 RGB；若需要 nvcc，使用该环境对应的12.8工具链
export CUDA_HOME=/usr/local/cuda-12.8
"$S1" -m transparent_depth.stage1_batch predict \
  --manifest "$WORK/rosters/rgb.jsonl" --asset-root "$ASSETS" \
  --output "$WORK/native-P" --cache-root "$WORK/stage1-jit-sm80"
"$S1" -m transparent_depth.stage1_batch seal \
  --manifest "$WORK/rosters/rgb.jsonl" --asset-root "$ASSETS" --output "$WORK/native-P"

export CUDA_HOME=/usr/local/cuda-12.1
"$S2" -m transparent_depth.prepare_stage2_data \
  --config configs/transcg_stage2_smoke.json \
  --input-roster "$WORK/rosters/inputs.jsonl" --label-roster "$WORK/rosters/labels.jsonl" \
  --predictions "$WORK/native-P/predictions.jsonl" --output "$WORK/prepared"
"$S2" scripts/run_transfer_smoke.py --prepared-root "$WORK/prepared" \
  --assets-root "$INIT" --output "$WORK/acceptance"
```

准备阶段输出独立 RGB-only / RGB+K / supervision rosters。smoke 取 train/val 各两台相机共四个视图，
另加 real-val 两帧；原生 P 缓存仅六帧。脚本仅运行四个 diagnostic epochs、八次 optimizer 更新，
alpha 为 `1,1,1,2/3,1/3,0,0,0`，不修改正式配置。它在第2轮完整保存后启动新进程恢复，
逐项比对模型/优化器/调度/RNG，并继续更新；检查各参数组实际变化、loss/output/gradient 有限，
测试正式 20/20/20 边界，最后分别评估内部 val 与 real-val。成功才写 `acceptance.json`；
使用按内部val选出的 `best.pt` 做流程评估，不根据 smoke 分数调整配置或宣称正式精度。
Stage1 权重、seal、GPU 任一缺失时应停在错误处，不用随机 P、GT 或伪造 seal 替代。

数据审计图在 `rosters/visualizations/`，coverage 在 `rosters/supervision_audit/`。
真实推理图在 `acceptance/{transcg-val,cleargrasp-evaluation}/visualizations/`；后两者本次未生成。

## 正式 60 epochs 与 resume（本次未运行）

在新的 `WORK` 下去掉准备命令的 `--smoke`，使用 `configs/transcg_stage2_60.json`，
重复上述 native P/seal 和 cache 命令；仅在目标机正式运行时生成全量缓存。
Stage1 先完整冻结导出，训练进程只读 P，Stage1 参数不在 optimizer 中。

```bash
export WORK=/path/to/transcg256-full
"$S2" -m transparent_depth.prepare_transfer_data \
  --config configs/transcg_stage2_60.json --output "$WORK/rosters"
# 按上节 predict/seal/cache 命令准备这个 WORK；cache --config 改用正式配置。

"$S2" -m transparent_depth.train_stage2 --config configs/transcg_stage2_60.json \
  --input-manifest "$WORK/prepared/manifests/input_manifest.jsonl" \
  --label-manifest "$WORK/prepared/manifests/label_manifest.jsonl" \
  --stage1-root "$WORK/prepared/stage1" --label-root "$WORK/prepared/labels" \
  --assets-root "$INIT" --output "$WORK/training"
# 恢复：同一命令末尾追加 --resume。保持 config、代码、manifest 和 world size 不变。
```

默认单 GPU，保留原每卡 batch=2、accumulation=5（effective batch=10），沿用原 Adam 和参数组 LR 倍率（lidf/spvnas=0.1，adapters=1）。
基础 LR=1e-3；每轮训练完成后 StepLR(step_size=20,gamma=0.1)，第1–20轮 1e-3、21–40轮 1e-4、
41–60轮 1e-5，再乘参数组倍率。第60轮后的 decay 仅保存在调度状态中。
前20轮有效背景为 GT；中20轮按 committed optimizer updates 连续从1降至0（含两个端点）；
后20轮全部 P。透明区域和无效 GT 区域始终 P。验证始终纯 P，全图 query；不存在标签混合验证。
只在41–60轮按内部 transCG val 的 **256-grid、逐图平均、透明区 MAE（米）**更新 best。
保存每个完整 epoch、`last.pt`、正式 `best.pt`，恢复从最后完整 epoch 的已提交状态继续。

## ClearGrasp real-val 最终评估

```bash
"$S2" -m transparent_depth.evaluate256 --checkpoint "$WORK/training/best.pt" \
  --input-manifest "$WORK/prepared/manifests/input_manifest.jsonl" \
  --label-manifest "$WORK/prepared/manifests/label_manifest.jsonl" \
  --stage1-root "$WORK/prepared/stage1" --dataset cleargrasp --role evaluation \
  --output "$WORK/final-real-val"
```

只在256网格上同时报告 P256 和 D256：all/transparent/background 的 MAE、RMSE（米）、AbsRel（REL）、
`delta_1.05`、`delta_1.10`、`delta_1.25`。每个δ统计 `max(pred/GT,GT/pred)<阈值` 的有效像素比例，
采用严格小于；保存值为[0,1]，展示百分数时乘100。先算每图每区域，再对非空图等权平均；
RMSE 也是逐图 RMSE 的平均。空区域不记为零误差，明确统计 nonempty/empty images 与 valid_pixels。
按 dataset/scene/camera 分组并输出逐图明细。本 real-val 目录只提供 d435 文件夹，因此该目录作为
可用 scene 分组，不猜测额外物理场景。两模型使用相同 GT、mask、有效区，不做 GT 尺度对齐或原尺寸上采样。
单独保存预测和监督/coverage，图使用固定米制色标。

旧256报告的 `delta_1/2/3` 阈值为1.25、1.5625、1.953125，只有旧 `delta_1` 对应当前 `delta_1.25`。
不得把三列直接改名；δ1.05和δ1.10需用保存的预测、GT及相同有效掩码重新统计，不能从旧聚合值恢复。
本次阈值修正不改变MAE/RMSE/AbsRel、训练损失或透明MAE选模，无需重新训练。

若旧评估目录仍包含 `predictions/`、`supervision/` 和 `metrics.json`，可重算到新目录，不加载模型或重新推理：

```bash
"$S2" -m transparent_depth.rescore256 --input "$WORK/final-real-val" \
  --input-manifest "$WORK/prepared/manifests/input_manifest.jsonl" \
  --output "$WORK/final-real-val-rescored"
```

旧报告缺少cache_id，必须用与原报告SHA256一致的原始input manifest匹配预测与监督；新报告包含cache_id。
旧报告保留不变。评测器源码参与训练contract，修正后不能直接resume旧训练；需续训时使用原始代码，
不得修改contract或关闭哈希检查。已有checkpoint的最终评估仍执行原有模型源码检查，修正δ无需重训。

导出使用 `transparent_depth.stage2.export_stage2_bundle(checkpoint, output, selection_metadata)`，
其中 selection_metadata 必须说明上述内部 val 选模规则。bundle 保存 `input_hw`、preprocessing、
训练 contract、权重与源码 SHA256；加载会核验。`python -m transparent_depth.stage2 --help` 的单图
入口接收原始 RGB/native P/native K，256 bundle 输出 D256.npy；历史240×320 bundle保留原输出行为。

## 路径、缓存及迁移核验

数据路径在 rosters 中以 `${TRANSCG_ROOT}` / `${CLEARGRASP_ROOT}` 表达，迁移仅重新绑定环境变量；
ASSETS、INIT、WORK、output、config 均由 CLI 控制。拷贝缓存和 checkpoint 时保留文件字节和内部目录结构，
无需旧服务器绝对路径。native P 路径和 seal 是 manifest-relative；portable labels 用内容 SHA256 而不是
旧机器 mtime 识别原始文件。源码、P/权重、划分、K、分辨率、插值、单位/有效范围变更都会令相关
Stage2 cache/训练 contract 不兼容；必须使用新目录重新准备，不编辑校验或将新旧 cache 混用。
原始高分辨率 P 在生产者不变时可用于重新准备 Stage2 cache，无需再次运行 Stage1。
新 checkpoint 记录 runtime ABI；resume 拒绝不同 PyTorch/CUDA ABI，迁移 GPU 本身不改变同 ABI 的训练状态。

`provenance/stage1_runtime_sources.json` 是更新过的源码清单，明确 `execution_verified=false`，
不是预测完成 seal。`provenance/transcg256_source_amendment.json` 记录 data.py 的前后摘要；
数值 core 的旧哈希测试仍然保留。不要因源码变化自行声称旧 release 的数值精度或 GPU 验收继续有效。
