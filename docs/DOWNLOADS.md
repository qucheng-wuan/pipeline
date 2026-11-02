# 官方数据、模型与固定下载清单

本页与 [机器可读清单](../provenance/download_catalog.json) 对应。下载时仍由脚本逐文件核验大小/SHA256；链接检查不等于下载并运行模型。
本次核对了官方数据页、固定HFrevision及项目Release的文件大小/摘要；DINOv3的3个文件未经授权返回401，不能声称已下载验证。
历史路径仅用于溯源，下面所有落盘路径都相对 `$ASSETS` 或 `$INIT`，可由CLI更改。

## 数据

[transCG官方数据页](https://graspnet.net/transcg) / [官方模型代码](https://github.com/Galaxies99/TransCG) / [官方采集器](https://github.com/Galaxies99/TransCG-data-collector)。需要info和全部13个data压缩包。DFNet权重不用于本pipeline。

| 文件 | Google官方链接 | 百度官方链接 / 提取码 |
|---|---|---|
| `transcg-info.zip` | [下载](https://drive.google.com/file/d/18LkbelKNTURF-8f8N-ykzs79FJ013knH/view?usp=sharing) | [下载](https://pan.baidu.com/s/1IddfXYOGOhuqw4CjXS0wfg) / `ncj0` |
| `transcg-data-1.zip` | [下载](https://drive.google.com/file/d/1rDgbMrzyO8GvTwEyPDbyx6pS9TVVDJLI/view?usp=sharing) | [下载](https://pan.baidu.com/s/1911NmQwLDN8zGvd_km7obg) / `3n0d` |
| `transcg-data-2.zip` | [下载](https://drive.google.com/file/d/10etZWFV5r_wJRCR3Q38eq9WaRY8qWBFp/view?usp=sharing) | [下载](https://pan.baidu.com/s/14PGEaJCjHJewt_Uy7UiSdQ) / `umim` |
| `transcg-data-3.zip` | [下载](https://drive.google.com/file/d/19tXZ9lzpW2gUk1ibkij76I_-oknCbqyP/view?usp=sharing) | [下载](https://pan.baidu.com/s/1nidz-kfQdxQJZL7RSG1iSg) / `g63o` |
| `transcg-data-4.zip` | [下载](https://drive.google.com/file/d/13JFpSXd8d7vkVJvAUT1E6JOL6ZGC9B5f/view?usp=sharing) | [下载](https://pan.baidu.com/s/1ME48e44CxueCq7yWLqr2iQ) / `6tk3` |
| `transcg-data-5.zip` | [下载](https://drive.google.com/file/d/1sMt8pbiVTU4J2YCdN31ZV3BY6m4ywX9O/view?usp=sharing) | [下载](https://pan.baidu.com/s/1B8LORKmsJeJ6KqbMBuE88Q) / `kp14` |
| `transcg-data-6.zip` | [下载](https://drive.google.com/file/d/1Uc2UDALZRG2NDZJt9CXMRLw1y6gpa_sT/view?usp=sharing) | [下载](https://pan.baidu.com/s/1vlg_FoUwE92cmUkJHhtofQ) / `jmsu` |
| `transcg-data-7.zip` | [下载](https://drive.google.com/file/d/1NXCGDzC2MrWktR9HcUddtRpajeg2JMEf/view?usp=sharing) | [下载](https://pan.baidu.com/s/1Hv6xVpeobgETDz29l3Y_3A) / `fqv9` |
| `transcg-data-8.zip` | [下载](https://drive.google.com/file/d/11MK-7_eX5oRNTdJPcOUXmvMaL3_cmFVh/view?usp=sharing) | [下载](https://pan.baidu.com/s/1nDFZqbdbUm9aDuR8wkc_eA) / `equb` |
| `transcg-data-9.zip` | [下载](https://drive.google.com/file/d/1oSMxQ0O9UD38Tu8w16BOhi8sPKMp6UQf/view?usp=sharing) | [下载](https://pan.baidu.com/s/1qP_i1_j9S0Md-646F2h4Ng) / `msc7` |
| `transcg-data-10.zip` | [下载](https://drive.google.com/file/d/1usAWlEJ0PE9gdgQb7CKo9CgkWOFr-Ms0/view?usp=sharing) | [下载](https://pan.baidu.com/s/115DubyGxkXU7L2gijp0zpQ) / `ncg0` |
| `transcg-data-11.zip` | [下载](https://drive.google.com/file/d/1M-QQcEGFcTw2SJrr-Z_Gg1-fyA0PZrcb/view?usp=sharing) | [下载](https://pan.baidu.com/s/1iVHnQ_u0PUyahY4Y8uFxQw) / `wntp` |
| `transcg-data-12.zip` | [下载](https://drive.google.com/file/d/1KxpeNNOEOlv0t6PSNYWg0PoQ37OizDhR/view?usp=sharing) | [下载](https://pan.baidu.com/s/1pU9hWS7Kl8Q_re_gCHw_SQ) / `in9v` |
| `transcg-data-13.zip` | [下载](https://drive.google.com/file/d/1FaZX6eWGYuMKu6riX-su3es3_XzoTawC/view?usp=sharing) | [下载](https://pan.baidu.com/s/12s6njiQa-MRdzbltbLdcWA) / `rmed` |

解压后支持 `TRANSCG_ROOT/scene*/` 或 `TRANSCG_ROOT/transcg-*/scene*/`；同一scene不可出现两次。
本仓库 [data/transcg_info](../data/transcg_info) 只带所需metadata、两份官方K及seed23划分，不含数据图像。
[来源记录](../data/transcg_info/source.json) 披露本机曾因官方Drive配额限制经第三方HF镜像获得info归档，并与官方采集器交叉核对K；该镜像不冒充官方发布。

[ClearGrasp官方数据页](https://sites.google.com/view/cleargrasp/data) / [官方代码](https://github.com/Shreeyak/cleargrasp)：

- [cleargrasp-dataset-train.tar](https://storage.googleapis.com/cleargrasp/cleargrasp-dataset-train.tar)：合成训练集。
- [cleargrasp-dataset-test-val.tar](https://storage.googleapis.com/cleargrasp/cleargrasp-dataset-test-val.tar)：本实验只取 `real-val`，不使用synthetic-val。

官方ClearGrasp的normal/edge/mask网络checkpoints不用于本pipeline，不需要下载。
训练适配要求 `depth-imgs-rectified/*-depth-rectified.exr` 为已校正米制optical-Z；如果原始压缩包只含未校正depth，不能改名代替。须用真实逐帧标定按官方流程生成相应数据并审计，再交给本适配器；适配器不会再次radial-to-Z。
当前数据有13656份JSON缺尺寸和FOV，官方数据链接本身不解决标定缺失。尚未找到可认证的这些帧补充标定，因此不能声称当前全量ClearGrasp可一键训练。

## Stage1：必须使用这组资产

自动下载：`python scripts/fetch_assets.py --manifest provenance/release_assets.json --component stage1 --asset-root "$ASSETS"`。
`--component stage1` 很重要：原manifest还记录旧240×320的Stage2 E58 bundle，它不是新256训练初始化或本实验结果。

- 项目自行训练的V2/Laplacian/R5/O2/O4/FAR/MoGe/TOM从[原项目v0.1.0 Release](https://github.com/cyberpunk6668/RGBOnly-Fusion-Laplacian-LIDF-SPVNAS/releases/tag/v0.1.0)下载。它们不是上游基础权重；微调MoGe不能换成[MoGe官方仓库](https://github.com/microsoft/MoGe)发布的其他checkpoint。
- [TransNormal官方源码](https://github.com/longxiang-ai/TransNormal)、[官方HF](https://huggingface.co/Longxiang-ai/TransNormal)：固定revision `796f97ad3d82ea2e455bed88afe3c33aca8e115f`，第一版模型，包含3份权重和4份配置。
- [DINOv3官方源码](https://github.com/facebookresearch/dinov3)、[ViT-H/16+官方HF](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m)：固定revision `c807c9eeea853df70aec4069e6f56b28ddc82acc`。登录HF并获得Meta模型许可后再下载；不要在代码/仓库中写入token。
- FlexGEMM JSON是冻结运行配置，复制到每个predictor私有可写cache；它不是可移植CUDA二进制。[FlexGEMM源码](https://github.com/JeffreyXiang/FlexGEMM)版本由requirements固定。

下表路径均相对 `$ASSETS`。完整SHA用于精确复现，不能用同名不同内容文件代替。

| 文件 / 落盘路径 | 固定下载地址 | 字节数 | SHA256 |
|---|---|---:|---|
| `stage1/v2_best_eligible.pt` | [v2](https://github.com/cyberpunk6668/RGBOnly-Fusion-Laplacian-LIDF-SPVNAS/releases/download/v0.1.0/trained-stage1--v2_best_eligible.pt) | 2996781 | `32faa8cae83a3acd536507df21e986f28a79c9ea70165c1998fed31babdb4e7d` |
| `stage1/laplacian_epoch08.pt` | [laplacian_e08](https://github.com/cyberpunk6668/RGBOnly-Fusion-Laplacian-LIDF-SPVNAS/releases/download/v0.1.0/trained-stage1--laplacian_epoch08.pt) | 181185 | `db5f9b61e767dea8bd9ad93c45779327f640dfb8c51bcfeccfee11c333f74256` |
| `stage1/r5.pt` | [r5](https://github.com/cyberpunk6668/RGBOnly-Fusion-Laplacian-LIDF-SPVNAS/releases/download/v0.1.0/trained-stage1--r5.pt) | 16552391 | `70e5b33f7bf2c7f0a65b427b58688506b8bf8bcd2366eaf89d788e4bd7dce648` |
| `stage1/o2.pt` | [o2](https://github.com/cyberpunk6668/RGBOnly-Fusion-Laplacian-LIDF-SPVNAS/releases/download/v0.1.0/trained-stage1--o2.pt) | 5604147 | `cbd0ca2258e870708c0325884010879a362963af49f578f1c8bbeca979451de1` |
| `stage1/o4.pt` | [o4](https://github.com/cyberpunk6668/RGBOnly-Fusion-Laplacian-LIDF-SPVNAS/releases/download/v0.1.0/trained-stage1--o4.pt) | 6921087 | `ce135c1853973ed777a5fab204b28fc6d7d2d76adca4fd600fdc67e13abf5629` |
| `stage1/far_dpt.pt` | [far](https://github.com/cyberpunk6668/RGBOnly-Fusion-Laplacian-LIDF-SPVNAS/releases/download/v0.1.0/trained-stage1--far_dpt.pt) | 38625663 | `221daf822f081685277fc6b9b5c03db88b2e33e89a10b2e3cca72b53244e6d3f` |
| `stage1/moge_step0299.pt` | [moge](https://github.com/cyberpunk6668/RGBOnly-Fusion-Laplacian-LIDF-SPVNAS/releases/download/v0.1.0/trained-stage1--moge_step0299.pt) | 1481381121 | `7f293312b7f388edfb5bb00fc13758b0cd005fd634f402d207f289127ca6e133` |
| `stage1/tomnet.pt` | [tom](https://github.com/cyberpunk6668/RGBOnly-Fusion-Laplacian-LIDF-SPVNAS/releases/download/v0.1.0/trained-stage1--tomnet.pt) | 35697799 | `a1541a8d14a9bffd211376821f198a0d1950d91d3d595be3d35510cab2496b02` |
| `stage1/tom_calibration.json` | [tom_calibration](https://github.com/cyberpunk6668/RGBOnly-Fusion-Laplacian-LIDF-SPVNAS/releases/download/v0.1.0/trained-stage1--tom_calibration.json) | 5673 | `e88062e11415f4bad39311a359753573935e0d7ab4d4a4c6b90d37fe82e4e66e` |
| `stage1/transnormal/unet/diffusion_pytorch_model.safetensors` | [transnormal_unet](https://huggingface.co/Longxiang-ai/TransNormal/resolve/796f97ad3d82ea2e455bed88afe3c33aca8e115f/unet/diffusion_pytorch_model.safetensors) | 3470311272 | `4d7f8e2935af42a18c6a7b2351b84c4f4dae042e08e133ef19bde6cc82584bf1` |
| `stage1/transnormal/vae/diffusion_pytorch_model.safetensors` | [transnormal_vae](https://huggingface.co/Longxiang-ai/TransNormal/resolve/796f97ad3d82ea2e455bed88afe3c33aca8e115f/vae/diffusion_pytorch_model.safetensors) | 334643276 | `a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815` |
| `stage1/transnormal/cross_attention_projector.pt` | [transnormal_projector](https://huggingface.co/Longxiang-ai/TransNormal/resolve/796f97ad3d82ea2e455bed88afe3c33aca8e115f/cross_attention_projector.pt) | 2625220 | `d8d607cc08ece6620e1caa7af375a7656596e190e9070bb28c87eef077aa2386` |
| `stage1/dinov3/model.safetensors` | [dinov3](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m/resolve/c807c9eeea853df70aec4069e6f56b28ddc82acc/model.safetensors) | 3362432800 | `3e1d4d18b9bfa9f28fad8e9de6a783f1313532d3460efa4cd0b12521d81d1a4d` |
| `stage1/transnormal/model_index.json` | [transnormal_model_index_json](https://huggingface.co/Longxiang-ai/TransNormal/resolve/796f97ad3d82ea2e455bed88afe3c33aca8e115f/model_index.json) | 377 | `cb42f168f368a312924233ac6d0d82b8512f90b30966f560578908cc6382624c` |
| `stage1/transnormal/unet/config.json` | [transnormal_unet_config_json](https://huggingface.co/Longxiang-ai/TransNormal/resolve/796f97ad3d82ea2e455bed88afe3c33aca8e115f/unet/config.json) | 1894 | `f25c31a2af50385156282b0002237fc56a36a4134fa68c3cd306f2725049a20a` |
| `stage1/transnormal/vae/config.json` | [transnormal_vae_config_json](https://huggingface.co/Longxiang-ai/TransNormal/resolve/796f97ad3d82ea2e455bed88afe3c33aca8e115f/vae/config.json) | 716 | `6b194a1bad5f6ab0431cc254088949b814f75d0c3230483ad8fc6be2cc1495a0` |
| `stage1/transnormal/scheduler/scheduler_config.json` | [transnormal_scheduler_scheduler_config_json](https://huggingface.co/Longxiang-ai/TransNormal/resolve/796f97ad3d82ea2e455bed88afe3c33aca8e115f/scheduler/scheduler_config.json) | 308 | `d38e01c6b084c8db8c16411d955f17974e6c3d83ef6dc9c6b7b576329d6907a9` |
| `stage1/dinov3/config.json` | [dinov3_config_json](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m/resolve/c807c9eeea853df70aec4069e6f56b28ddc82acc/config.json) | 744 | `35770e98e425c9383534bcbfa7f3b7a3cb1d943f6d26a37d7d19fea0735eab57` |
| `stage1/dinov3/preprocessor_config.json` | [dinov3_preprocessor_config_json](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m/resolve/c807c9eeea853df70aec4069e6f56b28ddc82acc/preprocessor_config.json) | 585 | `960c41d1f3a7778b936365769a2d90550b318a6c0a53a0296957adacfe5e0dd7` |
| `stage1/flex_gemm_autotune_cache.json` | [flex_gemm_autotune_cache](https://github.com/cyberpunk6668/RGBOnly-Fusion-Laplacian-LIDF-SPVNAS/releases/download/v0.1.0/runtime-stage1--flex_gemm_autotune_cache.json) | 6963 | `e40097daa6f432754cc1a27762ae5910a114df5034d9d218a61109f4021a39cd` |

## Stage2：fresh训练初始化

[torchvision官方源码](https://github.com/pytorch/vision)的旧ResNet34 ImageNet权重与[SPVNAS官方源码](https://github.com/mit-han-lab/spvnas)的20GMACs专用拓扑/初始化。拓扑net.config也必须精确匹配。
[LIDF官方源码](https://github.com/NVlabs/implicit_depth)已按原项目集成；新训练不加载另一个完整LIDF模型权重，heads/adapters按代码初始化。

运行 `python scripts/fetch_stage2_initialization.py --output "$INIT"`，下表落盘路径相对 `$INIT`：

| 文件 / 落盘路径 | 官方下载地址 | 字节数 | SHA256 |
|---|---|---:|---|
| `resnet34-333f7ec4.pth` | [下载](https://download.pytorch.org/models/resnet34-333f7ec4.pth) | 87306240 | `333f7ec4c6338da2cbed37f1fc0445f9624f1355633fa1d7eab79a91084c6cef` |
| `SemanticKITTI_val_SPVNAS@20GMACs/net.config` | [下载](https://hanlab18.mit.edu/files/SPVNAS/spvnas_specialized/SemanticKITTI_val_SPVNAS@20GMACs/net.config) | 3225 | `3bedf00c36f4ba926504fcc05cc00734b01d74f0803a4563583e315ec4f3c72d` |
| `SemanticKITTI_val_SPVNAS@20GMACs/init` | [下载](https://hanlab18.mit.edu/files/SPVNAS/spvnas_specialized/SemanticKITTI_val_SPVNAS@20GMACs/init) | 13937690 | `d8f8e53ab5774f610c33a9d7d023beb82fdc3f846276162b205906112a27af85` |

`--resume`恢复本次 `last.pt` 的模型/Adam/StepLR/RNG/更新计数，不是重新加载上述backbone。评估使用本次指标选出的best.pt。
移机后重新运行 `scripts/verify_transfer_assets.py`；缓存、checkpoint及bundle保留源码、分辨率、预处理、划分和模型来源用于核验。
当前仓库不重发官方受限权重；各组件许可和出处见 [THIRD_PARTY.md](../THIRD_PARTY.md)。
