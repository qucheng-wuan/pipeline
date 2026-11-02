# 本次执行任务 prompt

将现有 `/home/QuchengZou/projects/RGBOnly-Fusion-Laplacian-LIDF-SPVNAS` 整理为可迁移、可从clone复现的项目，发布到 `https://github.com/qucheng-wuan/pipeline`。

1. 提供独立的transcg256与cleargrasp256训练入口、60轮正式配置和小样本smoke配置；不自动混合数据、不覆盖已有输出，保留旧240×320路径。冻结原Stage1专家/V2/Laplacian E08，Stage2网络、损失、5 cm voxel不变。
2. 原始高分辨率RGB先生成同尺寸真实P并seal，再做完整视野256×256变换。RGB AREA、P LINEAR、GT有效像素加权AREA并保存coverage、mask nearest；米制optical-Z、integer centres、半像素K缩放一次。标签/模型输入分离，真实seal、源码、权重及缓存校验不得绕过。
3. transCG按官方metadata有效视角、相机内参与各自PNG单位读取，seed23场景划分内部val，默认仅第41–60轮按内部val透明MAE选best；官方test独立。ClearGrasp严格按目录/帧号配对，逐帧尺寸/FOV和real-val YAML验证，无synthetic-val、不另划val。
4. ClearGrasp不再固定交付第60轮final.pt：每轮对real-val作纯P全图预测，在第1–60轮中按256网格逐图平均透明区域MAE最小保存best.pt；并列保留较早轮次。last.pt用于resume。明确披露这是测试集选模，并披露Stage1历史验证集接触情况；GT/mask只能在预测完成后评分，不能影响模型输入/query。
5. 保持20/20/20 curriculum：有效背景GT→连续alpha1到0→纯P，透明/无效区域始终P；StepLR在20、40轮后有效衰减，数量从manifest推导，清楚区分官方backbone初始化和断点续训。
6. README给出clone、两环境安装、官方数据下载、路径布局以及audit→roster→Stage1 predict/seal→Stage2 cache→smoke→train/resume→best real-val evaluate命令。实际查证官方模型仓库、所需权重/配置直链、固定revision、落盘位置、大小与SHA256；项目自行训练的Stage1资产单列Release来源，不用官方基础权重冒充。说明DINOv3授权要求。
7. 迁移到A100时按目标GPU/工具链重编译CUDA扩展，保留vendor/许可证，不依赖本机临时文件、旧项目或5090专用.so；路径可覆盖。报告已完成CPU/数据检查和真实GPU验收状态，缺权重/标定/资源时明确阻塞，绝不伪造通过。
8. 不执行60轮、不生成全量P缓存、不停止已有GPU任务。保留1280/1920等原图尺寸审计、45454与45453数量差异说明。完成必要回归检查后提交并正常push目标仓库，不上传数据、模型大文件、缓存、密钥或本机环境，不强推覆盖远端历史。

交付代码、配置、复现README、下载清单、任务prompt和可核验的测试/阻塞记录；最终提供仓库与commit链接。
