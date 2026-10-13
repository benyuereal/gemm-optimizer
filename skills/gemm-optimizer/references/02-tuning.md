# Tuning phase:选手段、判适用、避死路

> 优化对象是 **hipBLASLt 已生成的 .co kernel**(反汇编后改汇编),不是从零写 kernel。
> 手段选错 = 白改 + 白测。先看判据,再动手。

## 决策顺序(按风险从低到高试)

1. **selection 先行**:不动代码,换已有 kernel 就可能拿到两位数收益(见下),零风险。
2. **slc**:bit-for-bit 无损,不碰权重,改一个 bit;条件满足就上。
3. **Layout A + vmcnt**:收益最大但必须权重重排配对,放最后做。

## 核心手段表(本项目验证过)

| 手段 | 改什么 | 动权重? | 生效条件 | 典型收益 |
|------|--------|--------|---------|---------|
| slc (evict-first) | 权重 buffer_load 置 word0 bit17 (0x20000), 两种编码: lds型 E05D1000→E05F1000 / **idxen型 E05C3000→E05E3000** | 否 | ①weight > L2(8MB) ②改的是**流式读权重**的那组 load (先反汇编数清: prologue lds 型 vs K-loop idxen 型, 大头在 K-loop; 只改 prologue 收益趋零); ③M=128 的 B-operand kernel 用不了 (见死路) | bit-for-bit无损, +几%~10% |
| Layout A (blocked权重布局) | kernel改3个地址常量 (s44=256, s44=0x6400, s50: 0x100→0x8000) | **是** (重排) | M=64/128 大K | -4%~-22% (需与权重重排配对) |
| vmcnt放宽 | s_waitcnt vmcnt(0)→vmcnt(4) (低word 0x0f70→0x0f74) | 否 | 配合Layout A | 提升在途load, 降latency暴露 |

实测收益案例:
- fp8 (FusedMatmul, lds型): m128-down -22.3%, m128-gate_up -7.5%, m64-gate_up -4.55%, m64-down -10.7% (slc+W2), gate_up 262MB slc -7.2%
- **bf16 (Tensile Cijk, idxen型) gate_up (524MB)**:
  - **M=32 slc -6.2%** (422.4→396.3µs, 1241→1323 GB/s, 5轮交替A/B, 顺序混杂已排除, verify norm_error=5.57e-5 无损)
  - **M=64 slc -6.14%** (410.4→385.2µs, 1280→1355 GB/s, 6轮交替A/B, patched先跑轮次仍 -6.31% 坐实, verify norm_error=5.57e-5)
  - 两 M 是**两个不同** winner kernel: M=32 → `MT128x32x128` (VA 0x1abdb400); M=64 → `MT128x128x64` (VA 0x78886750)。但两者权重都是流式 A-side load (idxen CLEAR / saddr-only), 各自在自己 kernel 的 K-loop idxen 组上打 slc (M=32 是 20 条 E05C3000; M=64 按反汇编数清), 收益持平 → **换 M、只要 operand 角色同为流式 A-side, slc 收益一致 (~ -6%); 但编码位置/条数按各自 kernel 反汇编, 不是同一 patch 直接搬**。

策略状态:
- **bf16 slc 已确认有效 (M=32 -6.2% / M=64 -6.14%), 待上生产** (生产主路径其实是 fp8, 见 GLM5.3 记忆; bf16 用于验证方法论)。一键复现: `scripts/slc_ab.sh`。

## 关键选择判据

- **weight 字节数 vs L2 8MB**:weight ≫ 8MB(如 gate_up 262MB)→ slc 有效;weight < 8MB(qkv 52MB / o 42MB 这类按 shape 实际算)→ slc 无效。
- **operand 专属,不跨 shape 迁移**:M=64 的 weight 是 A-operand(saddr-only, idxen CLEAR);M=128 的 weight 是 B-operand→VGPR(idxen SET)。slc-on-B 在 M=128 实测慢 2.5%——**不要迁移**。
- **Layout A 铁律**:kernel patch + `reorder_weights.py` 权重重排必须配对,缺一不可(缺一个 = 算错)。
- **selection(换 kernel 不改代码)**:某些 shape 换到已存在的另一 kernel 更快(如 qkv M=128 → MT192x128x64, -14%)。通过 aiter `tuned_gemm.csv` 按名强制,不走 .co patch。零风险,优先于 patch。

## 死路清单(本项目已确认无收益/回退,勿重走)

| 死路 | 结论 |
|------|------|
| slc 用在 **M=128**（weight 是 B-operand, idxen SET）| 回退 / 无增益（qkv M=128 / o M=128 都试过） |
| M=64 结构强塞 M=128 | 慢 29% |
| vmcnt(0) drain 单独用 (不配 Layout A) | no-op |
| split-K | LDS 64KB/CU → 1 WG/CU, 无空间 |
| 手写汇编重写 B-load | 回退 5% |

## 开工前自检(30 秒,别跳过)

- weight 字节数 = N × K × 1(FP8),和 8MB L2 比大小 → 决定 slc 要不要上。
- 反汇编确认 weight 是 A 还是 B-operand(idxen CLEAR vs SET)→ 决定 slc 能不能上;
  M=64 / M=128 的 operand 角色不同,结论不互通。
- 查 aiter `tuned_gemm.csv` 是否已有更快的同名/近名 kernel → 有就走 selection。
- Layout A 候选:确认 `reorder_weights.py` 能对该 shape 跑通(配对铁律,先验证再改 kernel)。

## 与 patch_co.sh 的衔接

`kernels.txt` 的 `reorder` 列直接编码手段:

```
kernel_name|N|K|M|src_path|target_us|reorder(1=LayoutA需重排, 0=slc/vmcnt不重排)
```

- 一次 patch run 里同一 kernel 可以叠 slc + Layout A + vmcnt(收益可加),但**归因要靠分轮测**。
- reorder=1 时脚本只验正确性和计时;权重重排是 serving 侧动作,patch 保留后必须重排再重启,否则算错(脚本会打印提醒)。
- 无收益时脚本自动回滚 .co,但**不会**撤销权重重排文件——回滚轮次记得检查。

## 迭代纪律

1. 每轮只改**一类手段**,便于归因;混改 = 说不清谁贡献的。
2. 有收益 → 保留;无收益 → 回滚并**记下原因**(进死路清单,别下轮重踩)。
3. 全 kernel 测,不许挑(铁律 6):报告里出现"只测了变好的那个"即失败。
