# RGB Fusion + Learned Laplacian + LIDF/SPVNAS

用 **transCG 或 ClearGrasp 分别训练** RGB-only 深度估计 pipeline。两个实验独立，默认不混合数据。
新训练入口统一 **256×256、60 epochs、单 GPU**；Stage1 始终冻结。

```text
原始高分辨率 RGB
  → 冻结专家 + V2 融合 + Learned Laplacian E08
  → 同分辨率、完整视野的米制深度先验 P（持久缓存 + producer seal）
  → RGB256 + P256 + K256
  → Stage2 LIDF/SPVNAS → D256
```

推理只需 RGB 和标定 K；GT/mask 只用于训练监督、训练 curriculum 和预测完成后的评分。
不使用传感器原始深度作为模型输入。网络、原有损失和 **0.05 m voxel** 保持不变。

| 实验 | Stage2 训练数据 | `best.pt` 选取 | real-val 的用途 |
|---|---|---|---|
| `transcg256` | 官方 train 中的70个场景 | 内部val：seed23划出的8个场景；第41–60轮透明MAE最小 | 对同一个best做最终评估 |
| `cleargrasp256` | ClearGrasp train；不使用 synthetic-val、不另划val | **全部第1–60轮**在real-val上纯P评估，透明MAE最小 | **测试集选模**及最终详细报告 |

透明MAE是256网格上逐图透明有效区域MAE的等权平均（米）；空区域跳过并计数。
严格改善才更新 `best.pt`，并列保留较早轮次；`last.pt` 用于续训。**不固定交付第60轮 `final.pt`。**
ClearGrasp的规则按本项目实验要求设置，属于 test-set selection，不能报告为独立未见测试精度。
此外，冻结Stage1历史训练/选模已接触ClearGrasp，历史验证包含real-val的173帧；两套实验均需披露。

## 当前验证状态

代码、数据协议、隔离与checkpoint逻辑已做CPU回归测试；**新256实验尚未完成真实GPU端到端验收，也没有60轮训练结果**。
当前服务器的数据与运行阻塞如下，clone后不能跳过这些前置条件：

- ClearGrasp train实际45454张，real-val173张。训练中13656份JSON缺图像尺寸及FOV，正式roster会拒绝生成；须补齐有官方依据的标定，不能猜K或静默删样本。
- 45454与历史45453的差异，是一张与历史synthetic-val重复的完整训练图。本实验不用synthetic-val，故保留它；详情见[ClearGrasp协议](docs/CLEARGRASP256.zh-CN.md)。
- Stage1的20个资产中，本机已核验17个；DINOv3权重及两份配置需通过官方授权下载。
- 当前5090/cu130环境在官方SPVNAS构造器的稀疏卷积中出现CUDA非法内存访问，尚未到真实训练前向。A100必须重新编译并通过真实smoke，不能据此声称已验证A100。

最新记录见[发布检查](reports/publication_verification.json)。`reports/cleargrasp256_acceptance.json`等保留了此前的数据检查与阻塞；其中旧末轮交付策略已由上述best策略替代。CPU测试的合成fixture不算真实P/seal或GPU验收。

## 1. Clone、官方数据和权重

```bash
git clone https://github.com/qucheng-wuan/pipeline.git
cd pipeline
```

必须先阅读[下载清单](docs/DOWNLOADS.md)：包含官方仓库、数据直链、每个所需权重/配置的固定版本、下载地址、落盘路径及SHA256。

| 内容 | 官方入口 / 固定来源 | 本实验需要什么 |
|---|---|---|
| transCG | [官方数据页](https://graspnet.net/transcg)、[官方代码](https://github.com/Galaxies99/TransCG) | `transcg-data-1.zip`至`13.zip`，以及`transcg-info.zip` |
| ClearGrasp | [官方数据页](https://sites.google.com/view/cleargrasp/data)、[官方代码](https://github.com/Shreeyak/cleargrasp) | `cleargrasp-dataset-train.tar`和`cleargrasp-dataset-test-val.tar`中的real-val；不使用synthetic-val |
| TransNormal | [官方仓库](https://github.com/longxiang-ai/TransNormal)、[官方模型](https://huggingface.co/Longxiang-ai/TransNormal) | 固定TransNormal第一版UNet/VAE/projector及配置，不能换成TransNormal2 |
| DINOv3 | [官方仓库](https://github.com/facebookresearch/dinov3)、[ViT-H/16+模型](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m) | 固定revision的模型及config/preprocessor；需同意许可并获得下载授权 |
| 项目训练的Stage1 | [原项目v0.1.0 Release](https://github.com/cyberpunk6668/RGBOnly-Fusion-Laplacian-LIDF-SPVNAS/releases/tag/v0.1.0) | V2、Laplacian E08、R5/O2/O4/FAR、微调MoGe、TOM及校准、FlexGEMM配置 |
| Stage2初始化 | [torchvision](https://github.com/pytorch/vision)、[官方SPVNAS](https://github.com/mit-han-lab/spvnas) | `resnet34-333f7ec4.pth`、SPVNAS `SemanticKITTI_val_SPVNAS@20GMACs/{net.config,init}` |

Stage1项目训练权重与官方基础权重不是同一组文件，不能相互替换。新256训练**不加载历史E58 Stage2 bundle**，也不要求完整LIDF预训练checkpoint；LIDF heads/adapters按现有代码初始化。
仓库包含必要vendor源码和许可证，不包含数据、模型大文件、缓存、编译产物或私有环境。

## 2. A100参考环境

Linux；准备git、Python3.11及开发头文件/venv、conda、GCC/G++11、CUDA12.1与12.8工具链。驱动需支持所用CUDA runtime。
两个Stage必须分别安装，完整固定依赖见[Stage1](requirements-stage1.txt)和[Stage2](requirements/stage2.txt)。

```bash
# Stage1: Python3.11, torch2.8.0+cu128, torchvision0.23.0+cu128
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=2
bash scripts/setup_stage1.sh "$PWD/.venv-stage1"
export S1="$PWD/.venv-stage1/bin/python"
"$S1" -m pip install --no-deps -e .

# Stage2: Python3.10, torch2.1.2+cu121, torchvision0.16.2+cu121
conda create -n rgbfusion-stage2-a100 python=3.10 -y
conda activate rgbfusion-stage2-a100
export S2="$CONDA_PREFIX/bin/python" TD_RUNTIME_PROFILE=reference
export CUDA_HOME=/usr/local/cuda-12.1 CC=gcc-11 CXX=g++-11
export TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=2
export TORCH_EXTENSIONS_DIR="$PWD/outputs/runtime/stage2-cu121-sm80"
bash scripts/install_stage2.sh
"$S2" -m pip install pytest
CUDA_VISIBLE_DEVICES='' "$S2" -m pytest -q tests
```

安装器将TorchSparse固定到 `385f5ce8718fcae93540511b7f5832f4e71fd835` 并源码编译；LIDF CUDA扩展从仓库源码JIT构建，scatter使用匹配torch/CUDA的包。启动时检查源码、ABI和架构，目标机仍需执行真实smoke。
其他GPU应修改架构/工具链并重新编译；不要复制5090的 `.so`、`.setup`、`.deps`或JIT缓存。当前blackwell-cu130配置不是A100默认环境，多GPU未验收，原安全检查保留。

```bash
# 改成自己的路径；不要把权重提交到git
export ASSETS=/path/to/stage1-assets INIT=/path/to/stage2-initialization
# 获得DINOv3模型访问权限后在Stage1环境登录：
"$PWD/.venv-stage1/bin/huggingface-cli" login
"$S1" scripts/fetch_assets.py --manifest provenance/release_assets.json \
  --component stage1 --asset-root "$ASSETS"
"$S2" scripts/fetch_stage2_initialization.py --output "$INIT"
"$S2" scripts/verify_transfer_assets.py --asset-root "$ASSETS" --init-root "$INIT"
```

所有文件校验大小/SHA256。缺权重时停止，不能用GT、随机P或假seal代替。

## 3. 选择一个实验，audit → roster

所有命令在clone的仓库根目录执行。路径通过环境变量/CLI覆盖，数据只读。
完整协议与独立命令分别见[transCG/A100](docs/TRANSCG256_A100.zh-CN.md)和[ClearGrasp256](docs/CLEARGRASP256.zh-CN.md)。
先选以下**其中一组**；它们使用不同WORK目录，互不覆盖。

```bash
# A: transCG；支持 ROOT/scene*/ 或 ROOT/transcg-*/scene*/，两者出现同一scene会报错
export TRANSCG_ROOT=/path/to/transcg-dataset
export CLEARGRASP_ROOT=/path/to/cleargrasp-dataset-test-val/real-val
export CONFIG=configs/transcg_stage2_smoke.json
export FORMAL=configs/transcg_stage2_60.json
export WORK=/path/to/transcg256-smoke
"$S2" -m transparent_depth.prepare_transfer_data --config "$CONFIG" \
  --smoke --output "$WORK/rosters"
# 此入口一起生成audit.json、seed23场景划分及rosters；可选 --include-official-test。
```

```bash
# B: ClearGrasp；逐目录/帧号严格配对，不能仅按文件排序
export CLEARGRASP_TRAIN_ROOT=/path/to/cleargrasp-dataset-train
export CLEARGRASP_REAL_VAL_ROOT=/path/to/cleargrasp-dataset-test-val/real-val
export CONFIG=configs/cleargrasp256_stage2_smoke.json
export FORMAL=configs/cleargrasp256_stage2_60.json
export WORK=/path/to/cleargrasp256-smoke AUDIT=/path/to/cleargrasp256-audit
"$S2" -m transparent_depth.prepare_cleargrasp256 audit --output "$AUDIT" --workers 2
"$S2" -m transparent_depth.prepare_cleargrasp256 roster --audit-root "$AUDIT" \
  --config "$CONFIG" --smoke --output "$WORK/rosters"
```

ClearGrasp要求每个对象目录包含 `rgb-imgs`、`depth-imgs-rectified`、`segmentation-masks`、`json-files`。
输入必须是**已校正的米制optical-Z EXR**；官方原始数据若未生成rectified目录，不能直接将radial depth改名当GT。
缺失逐帧标定须获得可信来源后重审计。当前全量audit有标定错误，正式roster拒绝继续；显式smoke只选已核验帧并保留阻塞状态。
默认audit完整解码RGB/mask、校验EXR结构/尺寸/哈希和重复RGB；`--decode-all`额外解压全部GT，smoke及label cache始终完整解码对应GT。

## 4. 原生Stage1 predict/seal → Stage2 cache → 真实smoke

```bash
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda-12.8
"$S1" -m transparent_depth.stage1_batch predict --manifest "$WORK/rosters/rgb.jsonl" \
  --asset-root "$ASSETS" --output "$WORK/native-P" --cache-root "$WORK/stage1-jit-sm80"
"$S1" -m transparent_depth.stage1_batch seal --manifest "$WORK/rosters/rgb.jsonl" \
  --asset-root "$ASSETS" --output "$WORK/native-P"

export CUDA_HOME=/usr/local/cuda-12.1
"$S2" -m transparent_depth.prepare_stage2_data --config "$CONFIG" \
  --input-roster "$WORK/rosters/inputs.jsonl" --label-roster "$WORK/rosters/labels.jsonl" \
  --predictions "$WORK/native-P/predictions.jsonl" --output "$WORK/prepared"
"$S2" scripts/run_transfer_smoke.py --config "$CONFIG" --formal-config "$FORMAL" \
  --prepared-root "$WORK/prepared" --assets-root "$INIT" --output "$WORK/acceptance"
```

RGB-only、RGB/K输入和监督是独立清单。Stage1先读原始高分辨率RGB生成完整P，绝不先缩RGB到256。
transCG smoke六帧覆盖两台相机/train/val/real-val；ClearGrasp四帧覆盖两种原图尺寸和real-val。
独立4轮smoke只执行8次optimizer更新，覆盖alpha=1/中间值/0及正式边界；第2轮后从新进程恢复，核验模型/优化器/调度/RNG、有限输出/梯度和实际参数变化，最后评估/可视化。
只有全部成功才产生 `acceptance/acceptance.json`。不启动60轮、不生成全量P缓存、不据此声称正式精度。

## 5. 正式60轮 train / resume（迁移验收通过后自行运行）

换**新的**WORK（例如 `/path/to/transcg256-full` 或 `/path/to/cleargrasp256-full`），设置 `CONFIG="$FORMAL"`；
重做第3节对应roster命令并去掉 `--smoke`，然后执行第4节predict/seal/cache三步，**不执行smoke脚本**。
ClearGrasp正式roster要求全量audit无错误。这样正式与smoke划分身份及缓存互不混用。

```bash
"$S2" -m transparent_depth.train_stage2 --config "$CONFIG" \
  --input-manifest "$WORK/prepared/manifests/input_manifest.jsonl" \
  --label-manifest "$WORK/prepared/manifests/label_manifest.jsonl" \
  --stage1-root "$WORK/prepared/stage1" --label-root "$WORK/prepared/labels" \
  --assets-root "$INIT" --output "$WORK/training"
# 断点续训：上述同一命令末尾追加 --resume，从last.pt恢复；不能拿best.pt当fresh初始化。
```

1–20轮：有效非透明背景用GT；21–40轮：按已提交optimizer更新连续alpha=1→0；41–60轮纯P。
透明区及无效GT始终用P，验证/测试始终纯P全图query；无随机几何增强。
Adam基础LR=1e-3，lidf/spvnas倍率0.1、adapters倍率1；StepLR每20轮乘0.1，所以三阶段基础LR为1e-3/1e-4/1e-5。
数量及steps从manifest推导。源码、配置、划分、预处理或ABI变化必须用新实验目录，不能删除检查强行resume。

```bash
# 两套实验均使用各自训练得到的best；ClearGrasp报告必须保留test_set_selection标记
"$S2" -m transparent_depth.evaluate256 --checkpoint "$WORK/training/best.pt" \
  --input-manifest "$WORK/prepared/manifests/input_manifest.jsonl" \
  --label-manifest "$WORK/prepared/manifests/label_manifest.jsonl" \
  --stage1-root "$WORK/prepared/stage1" --dataset cleargrasp --role evaluation \
  --output "$WORK/best-real-val"
```

直接在256网格同时报告P256与D256：全有效/透明/背景的MAE、RMSE（米）、AbsRel、δ1/δ2/δ3（阈值1.25^k），
逐图计算后对非空图等权平均；报告空区域和有效像素数，按可用场景/相机分组。不做GT尺度对齐或原生尺寸上采样指标。

## 几何、缓存与历史兼容

整图resize，不裁剪/补边：RGB AREA、P LINEAR、mask nearest-exact；深度float32米制optical-Z。
GT用有效正深度加权AREA，保存coverage，coverage≥0.5且深度在固定[0.1,2.0]m内才监督，不填GT孔洞。
K统一integer pixel centres后只变换一次：`f'=s*f`、`c'=s*(c+0.5)-0.5`，所有ray/点云/法线/体素使用K256。

缓存、checkpoint和bundle绑定源码、模型来源、权重、分辨率、插值、K、单位及划分。
本次选模/准备代码改变后，旧audit/roster/Stage2 cache须重新生成；Stage1 producer未变时可复用真实native P并重新prepare。
Stage1源码/权重变更须重新predict/seal。详细导出/加载见两份协议文档；禁止修改manifest哈希或删除验证来兼容旧产物。

历史240×320入口保留在 `configs/stage2_train*.json` 及[历史运行文档](docs/STAGE2_RUNTIME.md)。
其80轮、历史样本数量、E58权重与报告不属于新256实验，旧bundle若不满足当前源码检查，应按原版本加载或按当前流程重新导出。
上游源码/权重分别遵循[第三方说明](THIRD_PARTY.md)，本仓库没有把所有组件重新许可为同一种协议。
