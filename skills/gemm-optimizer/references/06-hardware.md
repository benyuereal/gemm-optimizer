# 06 · 硬件参数：所有"换硬件/换机器会变"的数字都在这

> 本文件集中所有**换硬件/换机器/换 DTK 会变**的参数。01-05 篇里出现的机器相关数字
> （峰值带宽、L2/LDS 大小、kernel VA、编码锚点、库路径）以本文件为准。
> 当前所有值均为本机（Hygon DCU gfx938 + DTK clang 18.0.0）实测。

## 换硬件流程（固定顺序）

```
改本文件参数表 → 按 05 篇"换机器字节比对"流程重新验证编码/VA → 更新本文件实测锚点
```

换硬件 ≠ 把 01-05 抄过去重读一遍：01-05 讲方法与判据，本文件讲数字。数字全以本文件当前值
为准；01-05 中出现的旧数字与本文件不一致时，以本文件为准并回头修旧文。

## 当前硬件参数表（全部本机实测）

| 参数 | 值 | 来源/备注 |
|------|-----|-----------|
| GPU | Hygon DCU gfx938（=CDNA3 架构），8 卡，`HIP_VISIBLE_DEVICES` 0-7 | HCU 利用率用 `hy-smi` 看 |
| HBM 峰值带宽 | **1.8 TB/s**（硬顶） | 01 篇 roofline 与利用率公式均按此口径 |
| FP8 算力峰值 | 560 TFLOPS | 01 篇双利用率公式用 |
| L2 | 8 MB | slc 判据：权重 > 8MB 才有效（见 02 篇）；gate_up 262MB 远大于此 |
| LDS | 64 KB/CU | 大 tile 只能 1 WG/CU，split-K 无空间（02 篇死路清单） |
| CU | 64 | tile 数 ≪ 64 时并行度不足是独立瓶颈（01 篇） |
| VGPR | 256 /wave | 大 tile 炸寄存器，选型上限约束 |
| 计时纪律 | **必须独占空闲卡**；**不锁频** | 共享卡数据作废（本项目踩过："M=64 断崖"是争抢假象）；禁止 `hy-smi` 锁频，多轮中位数代替 |
| 带宽利用率口径 | `GB/s = 权重字节数 / us`；利用率 = 实测 GB/s ÷ 1.8 TB/s | 01 篇"双利用率公式"同一口径 |

## 库文件与默认加载（本机实测）

`/opt/dtk/lib/hipblaslt/` 下多套 `library_*` 库：

| 目录 | 内容 | 谁走这套 |
|------|------|----------|
| `library_gfx938/` | `TensileLibrary_gfx938.co`（774MB）+ Custom_Cijk 系列 | **bf16/fp16 默认走这套** |
| `library_bw_nmz/` | `TensileLibrary` 128MB + `FusedMatmulKernel_gfx938.co` 24MB | **fp8 F8BS 生产 GEMM 走这套** |
| `library_bw/`、`library_gfx936/` 等 | 其它变体/旧卡型 | 别拿错卡型的库 |

- 默认加载哪套由 hipblaslt 内部**按数据类型**决定，不靠我们选。
- **确认方法**：`hipblaslt-bench --print_kernel_info` 看 kernel 名前缀——
  `Custom_Cijk`/`Cijk` = Tensile 库（library_gfx938）；`FusedMatmul` = bw_nmz。
  重定向（`HIPBLASLT_TENSILE_LIBPATH`）后第一步就是核对 winner 名与默认路径一致（03 篇诊断 6）。
- 库目录定位：`find /opt/dtk/lib -name '*Library*.co'`（目录名随 DTK 变）。

### VA 与 file_off

- **`file_off = VA - 0x1000`**：本机 DTK 的 .text 段基址差。换 DTK 先
  `llvm-readelf -S <.co>` 实测 .text 的 vaddr/file_off 重算，别沿用 0x1000。
- VA 换机器/换 DTK 必变，**永远按 kernel 名字解析**（01 篇"按名定位 kernel"）。

### 关键 VA 实测锚点（本机）

| kernel | 所在库 | VA | 区间终点（下一 FUNC VA） |
|--------|--------|----|--------------------------|
| `Custom_Cijk_Alik_Bljk_BBS_BH_Bias_AS_MT128x32x128_SE_K1_thread384_ccyBlockSize384_gfx938` | `library_gfx938/TensileLibrary_gfx938.co` | **0x1abdb400** | 0x1abe3f00 |
| `Custom_Cijk_Alik_Bljk_BBS_BH_Bias_AS_MT128x128x64_SE_K1_thread384_ccyBlockSize384_gfx938` | `library_gfx938/TensileLibrary_gfx938.co` | **0x78886750** | —（M=64 实测，终点按需 readelf 补） |

（bf16 gate_up **M=32** 与 **M=64** 的 winner kernel，**两个不同 kernel**（tile 不同：MT128×32×128 vs MT128×128×64），但权重都是流式 A-side load，slc 各自在 K-loop idxen 组上打（M=32 -6.2% / M=64 -6.14%）。file_off = VA - 0x1000。）

## 编码锚点（本机 DTK clang 18.0.0 实测，换 DTK 必须重验）

| 指令/位 | 本机编码 | 说明 |
|---------|----------|------|
| slc evict-first 位 | word0 **bit17（0x20000）** | 两种 lds type 同位 |
| lds 型 `buffer_load_dwordx4` | `E05D1000 → E05F1000` | prologue 直接写 LDS（fp8 FusedMatmul 主战场） |
| idxen 型 `buffer_load_dwordx4` | `E05C3000 → E05E3000` | K-loop 流式（bf16 Tensile Cijk 主战场） |
| `vmcnt(N)` 低 word | `0x0f70 \| N` | vmcnt(0)=0x0f70，vmcnt(4)=0x0f74 |
| `s_endpgm` | 0xbf810000 | 最小验证片段用 |

- **验证方法**：跑 05 篇"最小可复现汇编验证"片段（~1 分钟），objdump 对照本表；
  不一致 = 该机编码不同，走 05 篇字节比对流程全量重做。
- **编码锚点是否被硬件响应 ≠ 编码对不对**：编码对但硬件不响应 evict-first 时 patch 白做。
  用 03 篇"诊断"第 4 步的对照组实验确认（12 条 vs 201 条 slc 计时差异；无差异 = slc 路线该机判死）。

## 换硬件 checklist（换机器 / 换 DTK / 换卡型，按序做）

1. **改本文件参数表**：峰值带宽 / FP8 算力 / L2 / LDS / CU / VGPR，逐项按新硬件填，来源标注。
2. **跑 05 篇最小验证片段**确认新 DTK 汇编器的 vmcnt/s_endpgm 编码，更新本文件编码锚点表。
3. **重定位库目录**（`find /opt/dtk/lib -name '*Library*.co'`）、按 03 篇机械步骤**重新按名取 VA**，
   并重算 file_off 基址差（`llvm-readelf -S`）。
4. **字节比对**：交付的旧 `*_original.s` 在新机汇编后与新机原 .co 对应区逐字节 cmp（05 篇流程）。
   0 diff = 编码一致可沿用旧 patch；**有 diff = 全部重新反汇编、重新确认编码、重新 patch**。
5. **重跑 01 篇基线表**（同 shape 口径），新数字回填 01 篇与 02 篇的达标线/实测案例。
6. **对照组实验**确认新硬件是否响应 cache hint（03 篇诊断第 4 步：12 条 vs 201 条 slc 计时）。
   不响应 = slc 路线该机判死，直接进 02 篇死路清单，别再堆变体。
