# LGM — 高分辨率 3D 内容生成 NPU 部署及亲和性报告

| 项 | 内容 |
|---|---|
| 任务编号 | LGM |
| 任务用途 | text/image 到多视图图像、3D Gaussian、PLY 与 WebUI 预览视频 |
| 仓库 | https://github.com/AndyKong2020/LGM |
| 版本 / commit | main / 2f3a668 |
| 报告人 | - |
| 日期 | 2026-06-16 |
| 硬件 | Ascend 950PR ×8 / npu-smi 25.7.rc1 |
| 软件 | torch 2.10.0+cpu / torch-npu 2.10.0 / Python 3.11.6 |

---

## 1. 技术栈梳理
- 主语言:Python / PyTorch。
- ML 框架:torch、torch-npu、diffusers、transformers、accelerate、Gradio。
- 模型结构:MVDream/ImageDream 生成 4 视角图像,LGM U-Net 从 `[B,4,9,256,256]` 预测 3D Gaussian 参数,再导出 PLY 或渲染 WebUI 预览视频。
- CUDA 依赖:原项目默认依赖 CUDA 生态。`xformers` 为 attention 加速依赖,已用 PyTorch attention fallback 跑通;`diff-gaussian-rasterization` 用于高质量 3DGS video renderer,当前 NPU 环境不可用。
- 自定义核(.cu / C++ 扩展):上游 Gaussian rasterizer 为 CUDA 扩展路径;当前未移植 AscendC,Video 走 PyTorch 点投影 fallback。
- 第三方库:diffusers 0.38.0、transformers 5.12.0、gradio 6.18.0、rembg 2.0.76、kiui 0.3.3、safetensors 0.8.0、plyfile 1.1.4、imageio 2.37.3。
- 模型权重 / 来源:LGM `model_fp16_fixrot.safetensors`,MVDream diffusers 权重,ImageDream diffusers 权重。已记录主要权重 SHA256,用于复现一致性核对。

## 2. 部署步骤
- [x] 依赖安装:Python 3.11 虚拟环境,安装 torch-npu、diffusers、transformers、accelerate、gradio、rembg、kiui、safetensors、plyfile、imageio 等推理依赖。
- [x] 编译 / 构建:无本地 CUDA/C++ 扩展编译;CUDA Gaussian rasterizer 未编译,由 PyTorch fallback 兜底 WebUI preview。
- [x] 权重获取:LGM、MVDream、ImageDream 权重放在本地模型目录,WebUI 通过环境变量指定。
- [x] NPU 适配改动(device、torch_npu、禁用 CUDA 核等):
  - `core/device.py` 新增 NPU/CUDA/CPU device 选择与 autocast 上下文。
  - `app.py`、`infer.py` 移除硬编码 CUDA,支持 NPU autocast,并 lazy load text/image pipeline。
  - `mvdream/pipeline_mvdream.py` 显式传入目标 device,确保 text encoder、UNet、VAE、image encoder 迁移到 NPU。
  - `mvdream/mv_unet.py` 增加 xformers 缺失时的 fp32 attention fallback。
  - `mvdream/mv_unet_text.py` 隔离 text/image 两套动态 UNet,修复同进程 text-image-text 缓存污染。
  - `core/gs.py` 在 `diff_gaussian_rasterization` 不可用时走 PyTorch 点投影 fallback。
- 命令:
```bash
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
LGM_DATA_DIR=<data_dir> \
LGM_VENV=<venv_dir> \
LGM_CHECKPOINT=<model_fp16_fixrot.safetensors> \
scripts/run_npu_webui.sh
```

## 3. 验证用例
- 输入数据:
  - text-only:`a hamburger`
  - image-only:`bird_rgba.png`
  - text-after-image:`a hamburger`
- 运行方式:同一个 WebUI 进程内通过 API 连续执行 text-only、image-only、text-after-image,覆盖 MVDream、ImageDream、LGM forward、PLY 导出、MP4 输出和动态模块缓存稳定性。
- 期望输出:Multi-view Image、PLY、MP4 三类文件均生成;text/image 两条 pipeline 可在同进程连续运行。
- 实测输出:

| 用例 | 输入 | 结果 | 端到端耗时 | PLY 点数 | Multi-view 输出 | MP4 输出 | Video 首帧非白比例 |
|---|---|---|---:|---:|---:|---:|---:|
| text_hamburger_30_seed11 | prompt `a hamburger` | 通过 | 5.69s | 52509 | 17240B | 7008495B | 27.0% |
| image_bird_30_seed0 | `bird_rgba.png` | 通过 | 5.09s | 41828 | 9538B | 3463266B | 13.1% |
| text_hamburger_after_image_30_seed12 | prompt `a hamburger` | 通过 | 5.04s | 32511 | 10104B | 1879572B | 9.5% |

- 与 CPU/GPU 基准对比(误差/一致性):未建立同 commit、同 checkpoint、同 seed 的 CUDA reference。在线 demo 下载的 `bird_rgba` PLY 为 41217 个 vertex,950PR 输出为 41828 个 vertex,点数差 611 个,约 1.48%;抽样几何近邻均值约为参考 bbox 对角线的 4.5%,p95 约为 12.6% - 12.8%。该结果可作粗回归参考,不能写成严格精度对齐。
- 视觉验收:Multi-view Image 通过人工视觉验收;Video 因 renderer fallback 稀疏/方向不可靠,仅作为链路 smoke,不作为精度验收。

## 4. NPU 亲和性
口径:Ascend 950PR、`__NPU_ARCH__=3510`、FP16 主干推理。当前验证为单请求推理;MVDream/ImageDream 多视图生成与 LGM Gaussian prediction 已在 NPU 跑通;高质量 Video renderer 仍是主要缺口。

| 指标 | 数值 |
|---|---|
| 能否在 NPU 跑通 | 能。text-only、image-only、text-after-image 三条 WebUI API 均通过 |
| NPU 利用率 (npu-smi) | `npu-smi` 仅用于进程与 HBM 核对;短 kernel 峰值以 profiler `op_statistic/step_trace` 为准 |
| HBM 占用 | WebUI 推理进程约 12092MB |
| 关键算子是否回退 CPU | 代表性 NPU profile 未出现 AICPU 行;PLY/MP4 导出含 host/CPU 边界;原生 CUDA 3DGS rasterizer 不可用,Video 走 PyTorch fallback |
| 性能(吞吐/时延) | text 30 steps 约 20 it/s,image 30 steps 约 13-14 it/s;API 暖启动端到端约 5-6s |

**Profiler 取证**

| 项 | 结果 |
|---|---|
| 采集方式 | `torch_npu.profiler` 注入采集,`ExperimentalConfig(Level1 + PipeUtilization)`,预热在 profiler 外完成,active=20,循环后额外 `prof.step()` 收尾 |
| 覆盖路径 | `Conv2d+GroupNorm+SiLU`、SDPA attention、`reshape/permute/contiguous/interpolate/cat`、Gaussian activation pack |
| 产物验收 | `kernel_details.csv` 49 列 / 540 行;`op_statistic.csv`、`api_statistic.csv`、`operator_details.csv`、`step_trace_time.csv` 均生成;`trace_view.json` 合法 |
| step trace | 20 step;平均 Computing 191.197 us;Communication 总计 0 us |
| CPU fallback | `kernel_details.csv` 未检出 AICPU 行 |

**计算分布**(基于 profiler `op_statistic.csv`):

| 单元 | 主要算子 / 路径 | 压力 | 判定 |
|---|---|---|---|
| 向量(Vector,归一/激活/布局) | Transpose、Slice、ConcatD、GroupNormSilu、ResizeBilinearV2、SoftplusV2、Cast、RealDiv 等 | AI_VECTOR_CORE 2581.721 us,67.514% | 代表窗口由 Vector/layout 主导,不是纯 Cube-bound |
| 算力(Cube,卷积) | Conv2DV2 | AI_CORE 862.554 us,22.557% | 卷积主干亲和,但占比被 layout 与 Gaussian pack 稀释 |
| 融合 attention | FlashAttentionScore | MIX_AIC 379.671 us,9.929% | SDPA 可映射到 NPU fused attention,是替代 xformers fallback 的优先方向 |
| 通信(communication) | 无 collective | step trace Communication 0 us | 当前单请求推理无通信压力 |
| 调度(host/head) | diffusion step 循环、Gradio、PLY/MP4 导出 | profiler 窗口外仍有 Python 与文件导出开销 | 对端到端时延有影响,但不改变主干 NPU 可跑结论 |

**Profiler Top OP**

| OP Type | Core Type | Count | Avg Time(us) | Ratio |
|---|---|---:|---:|---:|
| Conv2DV2 | AI_CORE | 20 | 43.127 | 22.557% |
| Transpose | AI_VECTOR_CORE | 40 | 12.694 | 13.279% |
| Slice | AI_VECTOR_CORE | 100 | 4.005 | 10.475% |
| FlashAttentionScore | MIX_AIC | 20 | 18.983 | 9.929% |
| ConcatD | AI_VECTOR_CORE | 40 | 6.296 | 6.587% |
| GroupNormSilu | AI_VECTOR_CORE | 20 | 11.078 | 5.794% |
| ResizeBilinearV2 | AI_VECTOR_CORE | 20 | 9.744 | 5.097% |
| SoftplusV2 | AI_VECTOR_CORE | 20 | 8.723 | 4.563% |
| Cast | AI_VECTOR_CORE | 60 | 2.547 | 3.997% |
| RealDiv | AI_VECTOR_CORE | 20 | 6.644 | 3.475% |

**子路径 microbench**

| 子路径 | Shape / 说明 | 平均耗时 | 结论 |
|---|---|---:|---|
| Conv / norm / activation | `Conv2d(256,256,3)+GroupNorm(32)+SiLU`,`[1,256,64,64]` | 0.0596 ms | 主干卷积块可高效落 NPU |
| Manual attention fallback | `matmul+softmax+matmul`,`[1,8,1024,64]` | 0.2375 ms | 功能可用,但不是最佳 NPU 路径 |
| SDPA attention | `F.scaled_dot_product_attention`,`[1,8,1024,64]` | 0.0323 ms | 映射 `FlashAttentionScore`;max abs diff 0.000244,mean abs diff 0.00000773 |
| 多视图 layout | `reshape/permute/contiguous/interpolate/cat`,`[1,4,9,256,256]` | 0.2296 ms | Vector/layout 成本需要重点控制 |
| Gaussian pack | `clamp/sigmoid/softplus/normalize/tanh/cat`,`[1,65536,14]` | 0.0777 ms | 参数后处理可跑,Vector 占比高 |
| 点投影 fallback | `index_add_` 累积到 `512x512` | 0.0502 ms | 只覆盖 preview fallback,不等价于 3DGS rasterizer |
| 小矩阵逆 | batched `4x4 inverse`,64 组 | 0.0769 ms | 小 batch 易受调度开销影响 |

**判断依据**:Ascend 950PR FP16 Cube 平衡点约 270 FLOP/Byte。高 channel 3x3 Conv 和 GEMM/QKV 投影接近或高于该平衡点时偏 Cube 友好;低 channel 输入/输出 conv、1x1 Gaussian head、GroupNorm/LayerNorm/Softmax/Gaussian activation 属 Vector/MTE/head 辅助路径,不能用 Cube 峰值乐观估算。950PR 小包判断按 L2 `512B cacheline + 4×128B sector` 口径,不套用 A2 GM 512B 对齐经验。

**已覆盖算子路径**:

| 路径 | 实测证据 | 结论 |
|---|---|---|
| Conv / norm / activation | `Conv2DV2`、`GroupNormSilu` 进入 profiler Top OP,WebUI 链路通过 | LGM 与 diffusion U-Net 主干可跑 |
| Attention | Manual fallback 与 SDPA 均实测;SDPA 触发 `FlashAttentionScore` | xformers 缺失不阻断功能,SDPA 是后续性能优化方向 |
| Layout / preprocess | `Transpose/Slice/ConcatD/ResizeBilinearV2` 进入 profiler Top OP | 多视图拼接和输入 resize 可跑,但为主要 Vector/layout 成本 |
| Gaussian pack | `SoftplusV2/Cast/RealDiv/LpNormV2/Tanh/Sigmoid` 等已覆盖,形状 `[1,65536,14]` | PLY 前核心 Gaussian 参数路径可跑 |
| Video fallback | `index_add_` 点累积 smoke 通过,WebUI MP4 可生成 | 只能证明 preview 链路可跑,不等价于 3DGS rasterizer |

**3DGS renderer 指令级判断**:

| 子步骤 | 3510 路径 | 判定 |
|---|---|---|
| 投影、协方差/半径估计、SH/RGB | 规整 Vector | 亲和较好 |
| tile binning / 可见性列表 | scatter、atomic、动态列表 | 高风险,需要 AscendC 原生实现后评估 |
| 深度排序 / top-k | tile-local sort/top-k | 高风险,需要与 CUDA reference 做视觉/数值回归 |
| alpha blending / early-exit | 分支、逐像素状态更新、早退 | 高风险 |

原生 3DGS rasterizer 不应继续用 Python 点投影扩大半径来替代。更合适的方向是 `SIMD projection + SIMT/mixed tile binning + tile-local sort/top-k + Vector alpha blending`,并将 Gaussian 从 `[N,14]` AoS 尽量改为 SoA physical layout,减少小字段 scatter。采用 top-k 或固定 K 近似时,必须用 CUDA reference 做视觉和数值阈值验收。

## 5. 阻塞项
| 阻塞点 | 原因 | 是否硬阻塞 | CANN/AscendC 替代方案 | 兜底 |
|---|---|---|---|---|
| 高质量 3D Gaussian rasterization | `diff_gaussian_rasterization` 为 CUDA rasterizer,当前无 NPU 原生实现 | 对高质量 Video 是硬阻塞;对 Multi-view Image/PLY 不是阻塞 | AscendC/torch-npu custom path:SIMD projection + SIMT/mixed tile binning + tile-local sort/top-k + Vector alpha blending | PyTorch 点投影 fallback,仅作 WebUI preview smoke |
| xformers memory efficient attention | NPU 环境无 xformers,当前改为 PyTorch fp32 attention fallback | 非硬阻塞,功能已跑通 | SDPA 已实测映射 `FlashAttentionScore`,建议替换 fallback 后做 text/image 回归 | 保留当前 fallback 可继续跑 text/image pipeline |
| 严格精度基线 | 当前只有在线 demo PLY 粗参考,无同 commit CUDA reference | 非硬阻塞,影响精度定量结论 | 固定 CUDA reference 环境,同 seed/输入/checkpoint 对齐 Multi-view、PLY、Video | 当前以 Multi-view 视觉与 PLY 粗几何回归验收 |

## 6. 结论
- 运行方案:NPU 为主、host/renderer fallback 混合部署。MVDream/ImageDream 多视图生成与 LGM Gaussian prediction 已在 Ascend 950PR 上跑通,WebUI 可连续完成 text-only、image-only、text-after-image。
- 亲和性结论:主干卷积/GEMM 可用且亲和;代表窗口实际由 Vector/layout/Gaussian 后处理占主导,需要减少 `transpose/slice/concat/resize` 链式搬运。SDPA attention 已映射 `FlashAttentionScore`,优先级高于继续维护纯手写 attention fallback。
- 验收口径:当前以 Multi-view Image 与 PLY 为主;MP4 只证明服务链路产物完整,不能作为精度验收。
- 剩余风险:高质量 3DGS rasterizer 需要新 NPU 原生实现与 CUDA reference 回归;严格精度结论需要同 commit、同 checkpoint、同 seed 的 CUDA 对照环境。
