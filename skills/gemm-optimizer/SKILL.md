---
name: gemm-optimizer
description: 优化 Hygon DCU (gfx938) 上的 GEMM 算子（首个精度变体 = hipBLASLt FP8 F8BS channelwise）。
  当用户要求优化 matmul/GEMM、提升 decode 带宽利用率、给 hipBLASLt 的 .co kernel 打 patch
  （slc / Layout A / vmcnt）、与基线 A/B、或诊断"patch 了但不提速"时使用此技能。
---

# GEMM 算子优化（DCU gfx938）

> 定位：Claude Code 本身已能编译/跑分/迭代。本 skill 只提供**领域知识**（什么有效、什么是死路、
> 配对铁律、诊断顺序）+ **流程纪律**（防自嗨）。不重复造样板。
> 首个精度变体是 FP8（hipBLASLt F8BS channelwise `.co` patching）；结构对其它精度/GEMM 家族可复用。

## 何时用哪个 phase（按需读对应 reference，别全读）

| 你在做什么 | 读 |
|-----------|-----|
| 建基线、定位 kernel、算利用率、判断瓶颈 | `references/01-profiling.md` |
| 选优化手段（slc/Layout A/vmcnt）、判断适不适用、避死路 | `references/02-tuning.md` |
| 真正改汇编 / 拼进 .co / 权重重排 / 验证 / 诊断"没提速" | `references/03-patching.md` |
| 记 perf-log、出报告、判定达标 | `references/04-reporting.md` |

## 铁律（跨所有 phase，违反 = 失败）

1. **基线同机同口径**：优化前在同一张卡/同一会话跑基线。禁止拿别的机器或生产口径的数字
   对比隔离 bench（本项目踩过：把机器差算成 patch 收益）。
2. **正确性 ≤ 1 ULP**：FP8 累加顺序舍入，容差 1 ULP，不是 FP16 的 1e-2。
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

- 工具链 DTK（`/opt/dtk`），kernel 在 `/opt/dtk/lib/hipblaslt/library_bw_nmz/FusedMatmulKernel*.co`
  （目录名随 DTK 变，用 `find` 定位）。汇编器/读符/反汇编用 DTK 自带。
- 复用脚本在 `scripts/`：`patch_co.sh`（拼+验证+回滚全流程）、`bench.py`（单 kernel 计时）、
  `diagnose.py`（patch 是否在位）、`report.py`（汇总）。**脚本是可选加速器**，核心流程 Claude
  手工也能做（按 03-patching 的机械步骤）。
