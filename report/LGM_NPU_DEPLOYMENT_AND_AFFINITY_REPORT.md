# LGM — 高分辨率 3D 内容生成 NPU 部署及亲和性报告

| 项 | 内容 |
|---|---|
| 任务编号 | LGM |
| 任务用途 | text/image 到多视图图像、3D Gaussian、PLY 与 WebUI 预览视频 |
| 仓库 | https://github.com/AndyKong2020/LGM |
| 版本 / commit | main / 2f3a668 |
| 报告人 | - |
| 日期 | 2026-06-16 |
| 硬件 | Ascend 950PR ×8 / CANN 9.0.0 |
| 软件 | torch 2.10.0+cpu / torch_npu 2.10.0 / Python 3.11.6 |

---

## 1. 技术栈梳理
- 主语言:Python / PyTorch。推理链为 MVDream/ImageDream 生成 4 视角图像,LGM U-Net 从 `[B,4,9,256,256]` 预测 3D Gaussian 参数,再导出 PLY 或渲染 WebUI 预览视频。
- ML 框架:torch、torch_npu、diffusers、transformers、accelerate、Gradio。
- CUDA 依赖(必需/可选):`xformers` 为 attention 加速依赖,已用 PyTorch attention fallback 跑通;`diff_gaussian_rasterization` 为高质量 3DGS video renderer,无 NPU 原生实现。
- 自定义核(.cu / C++ 扩展):上游 Gaussian rasterizer 为 CUDA 扩展路径,无 AscendC 移植;Video 走 PyTorch 点投影 fallback。
- 第三方库:diffusers 0.38.0、transformers 5.12.0、gradio 6.18.0、rembg 2.0.76、kiui 0.3.3、safetensors 0.8.0、plyfile 1.1.4、imageio 2.37.3。
- 模型权重 / 来源:LGM `model_fp16_fixrot.safetensors`、MVDream diffusers 权重、ImageDream diffusers 权重,均离线放置并做哈希核对。

## 2. 部署步骤
- [x] 依赖安装:Python 3.11 虚拟环境,安装 torch_npu、diffusers、transformers、accelerate、gradio、rembg、kiui、safetensors、plyfile、imageio。
- [x] 编译 / 构建:无 CUDA/C++ 扩展编译;CUDA Gaussian rasterizer 未编译,NPU 环境用 PyTorch fallback 保留 MP4 产物生成能力。
- [x] 权重获取:LGM、MVDream、ImageDream 权重放在离线模型目录,WebUI 通过环境变量指定。
- [x] NPU 适配改动(device、torch_npu、禁用 CUDA 核等):统一 device/autocast;`app.py`、`infer.py` 移除硬编码 CUDA 并 lazy load pipeline;`mvdream/pipeline_mvdream.py` 显式迁移 text encoder、UNet、VAE、image encoder 到目标 device;`mvdream/mv_unet*.py` 增加 xformers 缺失 fallback 并隔离 text/image 动态模块;`core/gs.py` 在 CUDA rasterizer 缺失时走点投影 fallback。
- 命令:
```bash
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
LGM_DATA_DIR={data_dir} \
LGM_CHECKPOINT={model_fp16_fixrot.safetensors} \
scripts/run_npu_webui.sh
```

## 3. 验证用例
- 输入数据:text-only `a hamburger`;image-only `bird_rgba.png`;text-after-image `a hamburger`。
- 运行命令:同一个 WebUI 进程内通过 API 连续执行三条路径,覆盖 MVDream、ImageDream、LGM forward、PLY 导出、MP4 输出和动态模块缓存稳定性。
- 期望输出:Multi-view Image、PLY、MP4 三类文件均生成;text/image 两条 pipeline 可在同进程连续运行。
- 实测输出:

| 用例 | 输入 | 结果 | 端到端耗时 | PLY 点数 | Multi-view | MP4 | Video 首帧非白比例 |
|---|---|---|---:|---:|---:|---:|---:|
| text_hamburger_30_seed11 | prompt `a hamburger` | 通过 | 5.69s | 52509 | 17240B | 7008495B | 27.0% |
| image_bird_30_seed0 | `bird_rgba.png` | 通过 | 5.09s | 41828 | 9538B | 3463266B | 13.1% |
| text_hamburger_after_image_30_seed12 | prompt `a hamburger` | 通过 | 5.04s | 32511 | 10104B | 1879572B | 9.5% |

- 与 CPU/GPU 基准对比(误差/一致性):未建立同 commit、同 checkpoint、同 seed 的 CUDA reference。在线 demo `bird_rgba` PLY 为 41217 个 vertex,950PR 输出为 41828 个 vertex,点数差约 1.48%;抽样几何近邻均值约为参考 bbox 对角线的 4.5%,p95 约 12.6% - 12.8%。该结果用于粗粒度回归参考。
- 验证方式:Multi-view Image 目视通过;PLY 字段和点数通过;MP4 因 renderer fallback 稀疏/方向不可靠,仅验证链路产物生成。

## 4. NPU 亲和性
口径:Ascend 950PR、`__NPU_ARCH__=3510`、FP16、单请求推理。MVDream/ImageDream 多视图生成与 LGM Gaussian prediction 已在 NPU 跑通;高质量 Video renderer 是主要缺口。

| 指标 | 数值 |
|---|---|
| 能否在 NPU 跑通 | 能。text-only、image-only、text-after-image 三条 WebUI API 均通过 |
| NPU 利用率 (npu-smi) | `npu-smi` 用于进程与 HBM 核对;短 kernel 峰值以 profiler `op_statistic/step_trace` 为准 |
| HBM 占用 | WebUI 推理进程约 12092MB |
| 关键算子是否回退 CPU | 代表性 NPU profile 未出现 AICPU 行;PLY/MP4 导出含 host 边界;原生 CUDA 3DGS rasterizer 不可用 |
| 性能(吞吐/时延) | text 30 steps 约 20 it/s,image 30 steps 约 13-14 it/s;API 暖启动端到端约 5-6s |

- 算子回退清单:
  - MVDream/ImageDream 与 LGM 主干未检出 AICPU 行。
  - `diff_gaussian_rasterization` 无 NPU 原生实现,Video 使用点投影 fallback,不等价于高质量 3DGS rasterizer。
  - `xformers` 不可用时功能走 PyTorch attention fallback;SDPA 已实测可映射 `FlashAttentionScore`,建议替换并回归。
- profiler 摘要:
  - `torch_npu.profiler` 注入采集,`ExperimentalConfig(Level1 + PipeUtilization)`,预热在 profiler 外完成,active=20,额外 `prof.step()` 收尾。
  - 覆盖 `Conv2d+GroupNorm+SiLU`、SDPA attention、`reshape/permute/contiguous/interpolate/cat`、Gaussian activation pack。
  - `kernel_details.csv` 49 列 / 540 行;`op_statistic.csv`、`api_statistic.csv`、`operator_details.csv`、`step_trace_time.csv` 均生成;`trace_view.json` 合法。
  - 20 step 平均 Computing 191.197us,Communication 总计 0us。

**计算分布**(实测,Level1 加 PipeUtilization):

| 单元 | 主要算子 | 占比 | 说明 |
|---|---|--:|---|
| 算力(Cube,卷积) | Conv2DV2 | 22.6% | 卷积主干亲和,但被 layout 与 Gaussian pack 稀释 |
| 向量(Vector,归一激活) | GroupNormSilu、SoftplusV2、Cast、RealDiv、LpNormV2、Tanh、Sigmoid | 约 26% | Gaussian 后处理与归一激活占比较高 |
| 搬运(MTE/FixPipe,布局) | Transpose、Slice、ConcatD、ResizeBilinearV2 | 约 35% | 多视图重排、切片、拼接为代表窗口主成本 |
| 融合 attention | FlashAttentionScore | 9.9% | SDPA 可触发 NPU fused attention |
| 通信(communication) | 无 | 0 | 单请求推理无 collective |
| 调度(host/head) | diffusion step、Gradio、PLY/MP4 导出 | 偏高 | 对端到端有影响,不改变主干可跑结论 |

**热点算子明细**:

| 算子 | 次数 | 平均耗时 | 占比 |
|---|--:|--:|--:|
| Conv2DV2 | 20 | 43.127us | 22.557% |
| Transpose | 40 | 12.694us | 13.279% |
| Slice | 100 | 4.005us | 10.475% |
| FlashAttentionScore | 20 | 18.983us | 9.929% |
| ConcatD | 40 | 6.296us | 6.587% |
| GroupNormSilu | 20 | 11.078us | 5.794% |
| ResizeBilinearV2 | 20 | 9.744us | 5.097% |
| SoftplusV2 | 20 | 8.723us | 4.563% |

**子路径实测**:

| 子路径 | 形状 / 说明 | 平均耗时 | 判定 |
|---|---|---:|---|
| Conv / norm / activation | `Conv2d(256,256,3)+GroupNorm+SiLU`,`[1,256,64,64]` | 0.0596ms | 亲和 |
| Manual attention fallback | `matmul+softmax+matmul`,`[1,8,1024,64]` | 0.2375ms | 功能可用,非最佳路径 |
| SDPA attention | `F.scaled_dot_product_attention`,`[1,8,1024,64]` | 0.0323ms | 触发 `FlashAttentionScore`,max abs diff 0.000244 |
| 多视图 layout | `reshape/permute/contiguous/interpolate/cat`,`[1,4,9,256,256]` | 0.2296ms | 搬运主导 |
| Gaussian pack | `clamp/sigmoid/softplus/normalize/tanh/cat`,`[1,65536,14]` | 0.0777ms | Vector 主导 |
| 点投影 fallback | `index_add_` 累积到 `512x512` | 0.0502ms | 验证 WebUI MP4 链路可运行 |

平衡点为计算瓶颈与访存瓶颈分界,950PR FP16 约 270 FLOP/Byte。LGM 的卷积/GEMM 主干具备 NPU 亲和性;代表窗口实际由 layout、Vector 后处理与小算子链稀释 Cube 占比。最终 3DGS rasterizer 涉及 tile binning、深度排序、alpha blending 与动态状态更新,不是 PyTorch 点投影 fallback 可等价覆盖的路径。

## 5. 阻塞项
| 阻塞点 | 原因 | 是否硬阻塞 | CANN/AscendC 替代方案 | 兜底 |
|---|---|---|---|---|
| 高质量 3D Gaussian rasterization | `diff_gaussian_rasterization` 为 CUDA rasterizer,无 NPU 原生实现 | 对高质量 Video 是硬阻塞;对 Multi-view Image/PLY 不是阻塞 | AscendC/torch_npu custom path:SIMD projection + SIMT/mixed tile binning + tile-local sort/top-k + Vector alpha blending | PyTorch 点投影 fallback,验证 WebUI MP4 产物生成 |
| xformers memory efficient attention | NPU 环境无 xformers,采用 PyTorch fp32 attention fallback | 非硬阻塞 | SDPA 已实测映射 `FlashAttentionScore`,替换后做 text/image 回归 | 保留 fallback 可继续跑 text/image pipeline |
| 严格精度基线 | 仅有在线 demo PLY 粗参考,无同 commit CUDA reference | 非硬阻塞 | 固定 CUDA reference 环境,同 seed/输入/checkpoint 对齐 Multi-view、PLY、Video | 以 Multi-view 视觉与 PLY 粗几何回归验收 |

## 6. 结论
- 运行方案(NPU / NPU+CPU / CPU):NPU+CPU。MVDream/ImageDream 多视图生成与 LGM Gaussian prediction 上 NPU;PLY/MP4 导出与 renderer fallback 含 host 边界。
- NPU 亲和性:主干卷积/GEMM/SDPA 可用且亲和;性能风险集中在多视图 layout、Vector 后处理、小算子调度和 3DGS renderer。
- 风险与建议:高质量 3DGS rasterizer 需原生 NPU 实现与 CUDA reference 回归;attention 建议从手写 fallback 切到 SDPA 后做 text/image 回归;严格精度结论需同 commit、同 checkpoint、同 seed 的 CUDA 对照。
