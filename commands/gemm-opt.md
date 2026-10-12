---
description: 启动 GEMM kernel 优化全流程 (DCU gfx938 / hipBLASLt FP8)
argument-hint: <kernel名或shape> [目标, 如 "gate_up M=64 利用率提到70%"]
---

使用 gemm-optimizer skill，对 $ARGUMENTS 执行完整优化流程：

1. **profiling**（读 references/01-profiling.md）：同一张空闲卡（0-5，别碰生产卡）建基线，
   按名解析 kernel VA，算当前有效带宽/利用率。
2. **tuning**（读 references/02-tuning.md）：按 weight 字节数/operand/shape 选手段
   （slc / Layout A / vmcnt），避开死路清单。
3. **patching**（读 references/03-patching.md）：改汇编 → 用 scripts/patch_co.sh 拼装+验证
   （byte-exact + ≤1 ULP 正确性 + 同机 A/B 计时），失败自动回滚。
4. **reporting**（读 references/04-reporting.md）：写 perf-log.md，全 kernel 报告，
   达标或确认后出 report.md。

开始前先向用户确认：目标 shape/精度、目标性能、用哪张空闲卡。
**不要启动 serving**，只在空闲卡跑 bench。基线必须同机同口径（铁律1）。
