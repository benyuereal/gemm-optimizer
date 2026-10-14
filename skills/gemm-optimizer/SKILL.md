---
name: gemm-optimizer
description: 优化 Hygon DCU (gfx938) 上的 GEMM 算子（两大 .co 家族: bf16/fp16 Tensile Cijk + FP8 FusedMatmul F8BS）。
  当用户要求优化 matmul/GEMM、提升 decode 带宽利用率、给 hipBLASLt 的 .co kernel 打 patch
  （slc / Layout A / vmcnt）、与基线 A/B、或诊断"patch 了但不提速"时使用此技能。
  slc 已确认收益在 bf16 Tensile Cijk (M=32/64 gate_up ~-6%)；bf16 一键闭环 scripts/slc_ab.sh。
---

# GEMM 算子优化（DCU gfx938）

> 定位：Claude Code 本身已能编译/跑分/迭代。本 skill 只提供**领域知识**（什么有效、什么是死路、
> 配对铁律、诊断顺序）+ **流程纪律**（防自嗨）。不重复造样板。
> 覆盖两个 hipBLASLt `.co` 家族：**bf16/fp16 Tensile Cijk**（`library_gfx938/TensileLibrary_gfx938.co`，
> slc 已确认收益 ~-6%）+ **FP8 FusedMatmul F8BS**（`library_bw_nmz/FusedMatmulKernel*.co`）。
> 方法论通用（同一套 缓存提示 / 位级 patch / A/B 纪律），kernel 名 / 编码 / operand 角色按家族各查各的。

## 何时用哪个 phase（按需读对应 reference，别全读）

| 你在做什么 | 读 |
|-----------|-----|
| 建基线、定位 kernel、算利用率、判断瓶颈 | `references/01-profiling.md` |
| 选优化手段（slc/Layout A/vmcnt）、判断适不适用、避死路 | `references/02-tuning.md` |
| 真正改汇编 / 拼进 .co / 权重重排 / 验证 / 诊断"没提速" | `references/03-patching.md` |
| 记 perf-log、出报告、判定达标 | `references/04-reporting.md` |
| **bf16/fp16 Tensile slc 一键闭环**（解析→patch→A/B→norm，不改生产库） | `scripts/slc_ab.sh`（判据见 02 篇） |
| 理解 slc 为什么有效（缓存污染 / 权重过路 vs 激活常驻 / bit 位） | `references/07-optimization-principles.md` |
| tiling 怎么切、激活复用 N 次 vs M 次（slc 打在哪的推导） | `references/08-gemm-activation-reuse.md` |

## 铁律（跨所有 phase，违反 = 失败）

1. **基线同机同口径**：优化前在同一张卡/同一会话跑基线。禁止拿别的机器或生产口径的数字
   对比隔离 bench（本项目踩过：把机器差算成 patch 收益）。
2. **正确性达标才算数**：slc/vmcnt 是纯策略位 → 期望 **bit-exact**（norm_error 与 clean 同量级，
   bf16 ~1e-5）；Layout A 动累加顺序 → 容差按精度（FP8 1 ULP，非 FP16 的 1e-2）。`slc_ab.sh` 已自动判 norm。
3. **Layout A 配对**：kernel patch 与权重重排**必须同时生效**，缺一个 = 算错。slc 可单独用。
4. **不锁频**：本环境禁止 `hy-smi` 锁频；用多轮中位数代替。
5. **按名定位 kernel**：VA 换机器会变，永远按 kernel **名字**解析 VA；`file_off = VA - 0x1000`
   （此偏移随 DTK 变，换 DTK 先实测）。
6. **全 kernel 报告**：达标判断看所有相关 kernel，禁止只报变好的那个。
7. **不启动 serving**：只在空闲卡（0-5）跑 bench，绝不碰生产服务/生产卡。
8. **失败回滚**：任一验证步失败，恢复原 `.co` 并 md5 校验。

## 硬件天花板（决定优化只能在"调度/缓存提示/访问模式"做文章）

- HBM 峰值 **1.8 TB/s**（硬顶）；L2 **8MB**；LDS **64KB/CU → 1 WG/CU**（split-K 无空间）；
  VGPR 256/wave（大 tile 炸寄存器）。

## 环境

- 工具链 DTK（`/opt/dtk`），汇编器/读符/反汇编用 DTK 自带（05 篇有路径）。
- **两个 .co 家族**（目录名随 DTK 变，用 `find /opt/dtk -name '*.co'` 定位）：
  - **bf16/fp16 Tensile Cijk**：`/opt/dtk/lib/hipblaslt/library_gfx938/TensileLibrary_gfx938.co`
    → slc 主战场（bf16 gate_up M=32/64 已确认 ~-6%）。**一键闭环 `scripts/slc_ab.sh`**。
  - **FP8 FusedMatmul**：`/opt/dtk/lib/hipblaslt/library_bw_nmz/FusedMatmulKernel*.co`
    → slc + Layout A + vmcnt 主战场（02 篇 fp8 案例）。
- 复用脚本在 `scripts/`：
  - `slc_ab.sh` — **bf16/fp16 Tensile slc 一键闭环**（解析 VA → 数 load 组 → 打 slc → 重定向目录 →
    交替 A/B → norm 校验 → 写 perf-log；不改生产库）。`CARD=4 bash scripts/slc_ab.sh` 默认跑 bf16 gate_up M=32。
  - `patch_co.sh` — fp8 FusedMatmul 拼+验证+回滚全流程。
  - `bench.py`（单 kernel 计时）、`diagnose.py`（patch 是否在位）、`report.py`（汇总）。
  **脚本是可选加速器**，核心流程 Claude 手工也能做（按 03-patching 的机械步骤）。
