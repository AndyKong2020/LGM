# LGM NPU 部署报告

生成日期：2026-06-15

## 任务概述

本次部署任务是在 LGM 代码基础上，将项目同步到 Ascend NPU 服务器的目标容器内，建立可复用 Python 运行环境，并完成一轮以 WebUI 推理功能和 NPU 亲和性边界为主的复现验证。LGM（Large Multi-View Gaussian Model）是一个高分辨率 3D 内容生成项目，推理链路由三部分组成：MVDream/ImageDream 生成多视图图像，LGM 主模型从多视图预测 3D Gaussians，最后通过 Gaussian rasterizer 渲染视频并导出 PLY。

本次验证环境为 Ascend950PR，`npu-smi` 版本 `25.7.rc1`，每张 NPU HBM 约 114688 MB。运行时按任务约束只暴露物理 NPU `4,5,6,7`，通过 `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7` 隔离设备；`npu-smi` 验证物理 NPU 0-3 无项目进程，WebUI 进程落在物理 NPU 4。

## 部署过程

代码通过文件同步工具同步到远端容器工作目录，并在容器内启动 WebUI。远端源码目录保留项目代码、适配脚本和报告文件。

运行环境采用 Python 3.11 虚拟环境。当前 Ascend 运行栈使用 `torch 2.10.0+cpu` 与 `torch-npu 2.10.0`，并安装 LGM 推理所需的 `diffusers`、`transformers`、`accelerate`、`gradio`、`rembg`、`kiui`、`safetensors`、`plyfile`、`imageio` 等依赖。关键包版本如下：

| 组件 | 版本 |
| --- | --- |
| Python | 3.11.6 |
| torch | 2.10.0+cpu |
| torch-npu | 2.10.0 |
| torchvision | 0.25.0+cpu |
| diffusers | 0.38.0 |
| transformers | 5.12.0 |
| accelerate | 1.14.0 |
| gradio | 6.18.0 |
| rembg | 2.0.76 |
| kiui | 0.3.3 |
| safetensors | 0.8.0 |
| plyfile | 1.1.4 |
| imageio / imageio-ffmpeg | 2.37.3 / 0.6.0 |

原项目默认依赖 CUDA 生态，部署时做了必要的 NPU bringup 适配。主要改动如下：

| 模块 | 适配内容 | 作用 |
| --- | --- | --- |
| `core/device.py` | 新增统一 device 选择和 autocast 上下文 | 在 `torch_npu` 可用时选择 NPU，保留 CUDA/CPU fallback。 |
| `app.py`、`infer.py` | 移除硬编码 CUDA，lazy load text/image pipeline，禁用 LPIPS | 降低启动内存和 CUDA 依赖，支持 WebUI 在 NPU 上推理。 |
| `mvdream/pipeline_mvdream.py` | pipeline 调用显式传入目标 device | 让 text encoder、UNet、VAE、image encoder 迁移到 NPU。 |
| `mvdream/mv_unet.py`、`mvdream/mv_unet_text.py` | 替换 xformers 强依赖，增加 fp32 attention fallback | 支持 MVDream/ImageDream 动态 UNet 在 NPU 环境加载和执行。 |
| `core/gs.py` | CUDA Gaussian rasterizer 不可用时进入 PyTorch 点投影 fallback | 保证 WebUI 可产出 MP4 和 PLY，但 Video 只作为 smoke preview。 |
| `scripts/run_npu_webui.sh` | 固定 NPU 隔离、模型目录和启动命令 | 形成可复用启动入口。 |

本次使用的主要模型文件与哈希如下：

| 模型文件 | SHA256 |
| --- | --- |
| LGM `model_fp16_fixrot.safetensors` | `744d6324656342c64f871308e73db97f0eb51858d94329b30090e986a6d050ab` |
| MVDream UNet | `9ff839dd8c11591c2faa8efca41ac8145be8878b8ebf7cc92255fdcab0e09e53` |
| ImageDream UNet | `28d8b241a54125fa0a041c1818a5dcdb717e6f5270eea1268172acd3ab0238e0` |
| ImageDream image encoder | `2a56cfd4ffcf40be097c430324ec184cc37187f6dafef128ef9225438a3c03c4` |

环境 smoke 验证结果为：`torch.npu.is_available() == True`；在设置 `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7` 后，`torch.npu.device_count()` 返回 4；`npu:0` 上 float16 矩阵乘法通过。WebUI 进程环境中确认存在 `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7`。

## 功能验证结果

本次验证不是只拉起 WebUI 页面，而是在同一个 WebUI 进程内通过 API 连续执行三条推理路径：text-only、image-only，以及 image pipeline 加载后再次执行 text-only。这样可以覆盖 MVDream、ImageDream、LGM 主模型、PLY 导出、MP4 输出，以及 diffusers 动态模块缓存污染问题。

| 用例 | 输入 | 结果 | 端到端耗时 | PLY 点数 | Multi-view 输出 | MP4 输出 | Video 首帧非白比例 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `text_hamburger_30_seed11` | prompt `a hamburger` | 通过 | 5.69 s | 52509 | 17240 B | 7008495 B | 27.0% |
| `image_bird_30_seed0` | `bird_rgba.png` | 通过 | 5.09 s | 41828 | 9538 B | 3463266 B | 13.1% |
| `text_hamburger_after_image_30_seed12` | prompt `a hamburger` | 通过 | 5.04 s | 32511 | 10104 B | 1879572 B | 9.5% |

验证过程中确认 text-only prompt 已支持。此前 text 路径曾出现 `ResBlock.forward() missing 1 required positional argument: 'emb'`，原因是 text/image 两套 diffusers 动态 `mv_unet.py` 同名缓存污染，导致 `CondSequential` 中的 `isinstance(layer, ResBlock)` 对跨模块类对象失效。当前通过 text 专用动态模块和类名兼容判断修复，text-only、image-only、text-after-image 均可连续运行。

`npu-smi info` 显示 WebUI 运行时物理 NPU 0-3 无运行进程，物理 NPU 4 有 WebUI 推理进程，进程显存约 12092 MB。该结果用于确认设备隔离没有误用前四张卡。

本次还将在线 demo 下载的 `bird_rgba` PLY 与 A5 输出做了粗粒度比较。在线 demo PLY 为 41217 个 vertex，A5 输出为 41828 个 vertex，点数差 611 个，约 1.48%。抽样几何近邻均值约为参考 bbox 对角线的 4.5%，p95 约为 12.6% - 12.8%。该结果适合作为视觉回归参考，不适合作为严格数值精度金标准。

需要特别说明的是，WebUI Video 当前不能作为精度验收依据。原项目依赖 CUDA `diff-gaussian-rasterization` 生成高质量 novel-view video，而当前 NPU 环境中该模块不可用，Video 走 PyTorch 点投影 fallback。因此 MP4 文件可以证明服务链路可跑通，但画面稀疏或方向异常应归入 renderer 兼容性缺口。

## 部署结论

LGM 已在 Ascend NPU 容器内完成 WebUI 复现。text-only 和 image-only 推理均能生成 Multi-view Image、PLY 和 MP4，并确认没有使用物理 NPU 0-3。

当前部署可以支持 LGM 推理主干的 NPU 亲和性分析：MVDream/ImageDream 多视图生成和 LGM Gaussian prediction 已经跑通，PLY 导出可用于粗粒度结构回归；但最终 Video 渲染依赖的 CUDA Gaussian rasterizer 未完成 NPU 原生移植，当前只能作为 smoke preview。后续若要声明完整视觉渲染精度或性能，需要补充原生 3D Gaussian rasterization 适配，并对投影、tile 相交、深度排序和 alpha blending 做专项验证。
