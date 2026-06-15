# LGM NPU 亲和性分析

生成日期：2026-06-15

## 分析结论

LGM 在 Ascend950PR、`torch-npu 2.10.0` 环境下具备中等 NPU 亲和性。MVDream text-to-multiview、ImageDream image-to-multiview、LGM 多视图到 Gaussian 参数预测和 PLY 导出这几条推理主干路径已经在 NPU 上跑通。WebUI 可以完成 text-only 和 image-only 输入，并在同一进程内连续生成 Multi-view Image、PLY 和 MP4。

但该项目不能归类为“完整 NPU 原生支持”。本轮实测风险集中在两类路径：第一类是 MVDream/ImageDream attention 原始依赖 `xformers`，当前通过 PyTorch fp32 attention fallback 跑通；第二类是最终 Video 渲染依赖 CUDA Gaussian rasterizer，当前环境缺少对应 NPU 原生实现，只能走点投影 fallback。

因此，本次结论是：LGM 的 NPU 推理主干可用，Multi-view Image 和 PLY 可以作为当前功能验收与粗粒度回归依据；Video 输出只能证明 WebUI 链路可跑通，不能作为精度验收。真正的 A5 亲和性缺口集中在 3D Gaussian rasterization，而不是 text/image 多视图生成或 LGM 主模型 forward。

## 已验证的 NPU 计算路径

本轮实测的 NPU 友好路径集中在卷积 U-Net、Transformer attention、多视图张量重排和 Gaussian 参数激活这些算子模式上。LGM 主模型的 `core/unet.py` 主要由 `Conv2d`、`GroupNorm`、`SiLU`、nearest upsample、stride convolution downsample、multi-view self-attention 和 skip connection 组成；MVDream/ImageDream 动态 UNet 还包含 `LayerNorm`、`Linear`、`GELU`、cross-attention、VAE encode/decode 相关卷积和 bilinear resize。下表按 NPU 功能和算子模式归纳，不把 WebUI 功能列表当作支持矩阵。

| NPU 功能/算子模式 | 实测状态 | 验证方式 | 边界说明 |
| --- | --- | --- | --- |
| 设备隔离、NPU device 选择与 fp16 autocast | 支持 | `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7` 下 `torch.npu.is_available()==True`，`device_count==4`，`npu:0` float16 matmul 通过 | 该项只用于确认设备选择、隔离和基础 dtype 路径正确。 |
| 2D 卷积主干：`Conv2d`、1x1 conv、3x3 conv、stride conv | 支持 | isolated smoke 中 `conv2d_groupnorm_silu_residual`、`stride_conv_downsample` 通过；完整 LGM forward 已生成 PLY | 覆盖 LGM U-Net、最后 1x1 Gaussian head、MVDream/ImageDream UNet/VAE 的主要卷积模式。 |
| 归一化与激活：`GroupNorm`、`LayerNorm`、`SiLU`、`GELU` | 支持 | `GroupNorm+SiLU+Conv2d`、`LayerNorm+Linear+GELU` smoke 通过；text/image pipeline 完整推理通过 | xformers 不可用时 attention 走 PyTorch fallback，归一化和激活本身未形成阻塞。 |
| Attention 基础算子：`Linear` QKV、reshape/permute、`matmul`、`softmax`、再 `matmul` | 支持 | `attention_matmul_softmax_matmul` smoke 通过；MVDream text 30 steps 和 ImageDream 30 steps 均通过 | 这是替代 xformers 的 fp32 attention fallback，功能可用但性能不等价于 CUDA memory-efficient attention。 |
| Feed-forward / projection：`Linear`、`GELU`、dropout-disabled inference projection | 支持 | `linear_layernorm_gelu_ffn` smoke 通过；CLIP/text encoder 与动态 UNet 推理通过 | 覆盖当前 WebUI 推理中的 text encoder、image encoder 和 UNet projection。 |
| 多视图张量布局：`cat`、`stack`、`permute`、`reshape`、`contiguous` | 支持 | `cat_stack_permute_reshape_contiguous` smoke 通过；LGM 将 `[B,4,9,H,W]` 重排为 U-Net 输入并输出 Gaussian | 这类操作对 LGM 多视图拼接、skip connection、Gaussian pack 很关键，当前未观察到设备错误。 |
| 上下采样与预处理：nearest upsample、bilinear `interpolate` | 支持 | `nearest_upsample_plus_conv`、`bilinear_interpolate_preprocess` smoke 通过；WebUI image/text 路径完整通过 | 覆盖当前 WebUI 输入 resize、VAE/image latent resize 和 LGM U-Net upsample。 |
| Gaussian 参数激活：`clamp`、`sigmoid`、`softplus`、`normalize`、`tanh`、`cat` | 支持 | `gaussian_activation_pack` smoke 通过，形状 `[1,65536,14]`，对应 big 配置 4 视角 128×128 splats | 这是 PLY 输出前最核心的 LGM Gaussian 参数生成路径。 |
| camera/video 辅助张量：小矩阵 `inverse`、matmul、CPU transfer | 功能支持 | `batched_4x4_inverse_matmul` 和 `cpu_transfer_for_numpy_imageio_ply` smoke 通过；WebUI 能生成 MP4 | 这些只是 Video 辅助路径，不代表 CUDA Gaussian rasterizer 已支持。 |
| fallback 点投影聚合：`index_add_`、scatter-style accumulation | 功能支持，但非等价渲染 | `index_add_point_accumulation_cpu_style` smoke 通过；WebUI MP4 可生成 | 只能证明 fallback preview 可运行，不能作为 3DGS alpha-blending 精度依据。 |

在算子级 smoke 之外，本轮还跑了三条完整 WebUI API：text-only、image-only、text-after-image。`a hamburger` text-only 30 steps 生成 52509 点 PLY，`bird_rgba.png` image-only 30 steps 生成 41828 点 PLY，image pipeline 加载后再次 text-only 仍可生成 32511 点 PLY。这说明上述算子不只是 isolated smoke 通过，也已经被完整推理链路覆盖。

从日志与 API 补测看，MVDream text 30 steps 约 `20 it/s`，ImageDream 30 steps 约 `13-14 it/s`；三条 API 暖启动端到端耗时约 5-6 秒。`npu-smi` 显示物理 NPU 4 上有 WebUI 推理进程，进程显存约 12092 MB；物理 NPU 0-3 无项目进程，该证据只用于确认设备隔离有效。

当前 Multi-view Image 通过人工视觉验收。对在线 demo 下载的 `bird_rgba` PLY 与 A5 输出做粗比较，在线 demo PLY 为 41217 个 vertex，A5 输出为 41828 个 vertex，点数差约 1.48%；抽样几何近邻均值约为参考 bbox 对角线的 4.5%，p95 约为 12.6% - 12.8%。该结果可作为宽松回归参考，但不应写成严格数值精度对齐。

## fallback 与性能风险

本轮观察到的 fallback 与性能风险比较集中。功能跑通不代表所有路径都已经具备原生 NPU 性能。

| 路径 | 观察 | 影响 |
| --- | --- | --- |
| xformers attention | 当前环境无 xformers，动态 UNet 改为 PyTorch fp32 attention fallback | text/image 多视图可运行，但性能、显存和 CUDA+xformers 不等价。 |
| diffusers 动态模块 | text/image 两套 `mv_unet.py` 原本会被同名缓存，导致 `ResBlock.forward()` 缺少 `emb` 参数 | 已通过 text 专用动态模块和跨模块类判断修复；升级 diffusers 后需重新验证。 |
| rembg 背景去除 | 输入图像背景去除作为前处理执行 | 影响端到端耗时，不代表 LGM 主模型 NPU 性能。 |
| CUDA Gaussian rasterizer | `diff_gaussian_rasterization` 在当前环境不可用 | WebUI Video 进入点投影 fallback，稀疏/方向异常不能作为模型精度问题判断。 |

最重要的风险是最终 Gaussian render。原项目 Video 依赖修改版 CUDA `diff-gaussian-rasterization`，而当前 NPU 环境中该模块不可导入。现有 PyTorch fallback 只做粗略点投影和颜色累积，没有完整 3DGS rasterization 所需的投影半径、tile 相交、深度排序、alpha blending、遮挡关系和剪枝逻辑。因此 Video 稀疏或上下方向异常是 renderer 兼容性问题，不应归因于 MVDream/ImageDream 或 LGM Gaussian prediction 失败。

## 阻塞项分析

阻塞项按当前 WebUI 推理闭环中仍未原生支持的 NPU 功能归纳如下。已经通过适配修复的问题不写入阻塞表。

| 阻塞的 NPU 功能/项目路径 | 实测现象 | 影响范围 | 判断 |
| --- | --- | --- | --- |
| 3D Gaussian rasterization | `diff_gaussian_rasterization` 不可用，WebUI Video 走点投影 fallback | MP4 视频质量、高质量 novel-view 渲染、渲染性能 | 当前最大亲和性缺口；需要原生 NPU rasterizer 或等价 AscendC 算子链。 |

3D Gaussian rasterization 是最明确的结构性阻塞。完整 3DGS 渲染不是单点投影，而是需要将 3D Gaussian 投影到 2D、计算 Gaussian 与 tile 的相交关系、按深度排序、再按像素进行 alpha blending。当前 LGM 上游实现把这些能力封装在 CUDA rasterizer 中，本次环境没有 NPU 等价实现，所以只能保留低质量 fallback 作为 WebUI 链路 smoke。

xformers 的问题相对次要。它影响 MVDream/ImageDream attention 的性能和显存，但通过 fp32 attention fallback 后功能路径已经跑通。后续如果要做性能报告，需要单独测 attention fallback 的耗时、HBM 占用和是否存在 AiCPU fallback。

diffusers 动态模块缓存问题已经解决，但需要在后续版本升级时保留回归测试。最小回归序列应包含：先跑 text-only，再跑 image-only，再跑 text-only，确保 text/image 两套动态 UNet 不再互相污染。

## 适配建议

后续如果要把 LGM 推进到更完整的 A5 交付，优先级应围绕项目真实瓶颈展开，而不是继续调点投影 Video 的观感。

第一，建立分阶段 profile。建议在 `process()` 中分别记录背景去除、MVDream/ImageDream diffusion、LGM Gaussian prediction、PLY 保存、Video render 的耗时，并结合 `npu-smi` 或 msprof 采集 NPU 利用率和 HBM 占用。当前端到端 5-6 秒只能证明链路可用，不能定位 NPU 性能瓶颈。

第二，把 Video 从当前精度验收项中降级。现阶段可验收项应是 Multi-view Image 的视觉形状、PLY 字段完整性、PLY 点数、bbox、宽松几何距离和固定输入回归。MP4 只能作为服务链路是否产物完整的 smoke 指标。

第三，若业务必须验收高质量 Video，需要做原生 3DGS rasterizer 适配。适配目标应覆盖 3D Gaussian 投影、tile 相交/剔除、tile 内深度排序、alpha blending 和必要的反向路径。简单扩大 Python fallback 的点半径只能改善观感，不能等价替代 CUDA rasterizer。

第四，固化 text/image 动态模块隔离。启动时应继续分别维护 text 与 image 的动态 `mv_unet.py`，并清理 diffusers 本地动态模块缓存。每次升级 diffusers、transformers 或模型目录后，都应跑 text-image-text 的同进程回归。

第五，建立严格 reference 环境。在线 demo 下载的 PLY 只能作为粗参考；严格精度对比需要在同一 commit、同一 checkpoint、同一依赖版本、同一 seed 和同一输入下生成 CUDA reference，再与 A5 输出比较。

综上，本项目当前 NPU 亲和性应表述为：推理主干已跑通，text/image 多视图与 LGM Gaussian prediction 可用；最终 Gaussian rasterization 是主要不亲和点，Video 质量问题应归入渲染算子缺口，而不是模型主干失败。
