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
| slc (evict-first) | weight→LDS 的 buffer_load 置 word0 bit17 (0x20000) | 否 | ①weight > L2(8MB) ②weight 是 A-operand(idxen CLEAR, 即 M=64 型); M=128 是 B-operand 用不了 | bit-for-bit无损, +几%~10% |
| Layout A (blocked权重布局) | kernel改3个地址常量 (s44=256, s44=0x6400, s50: 0x100→0x8000) | **是** (重排) | M=64/128 大K | -4%~-22% (需与权重重排配对) |
| vmcnt放宽 | s_waitcnt vmcnt(0)→vmcnt(4) (低word 0x0f70→0x0f74) | 否 | 配合Layout A | 提升在途load, 降latency暴露 |

实测收益案例:m128-down -22.3%, m128-gate_up -7.5%, m64-gate_up -4.55%, m64-down -10.7% (slc+W2), gate_up 262MB slc -7.2%

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
