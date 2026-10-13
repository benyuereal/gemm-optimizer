# 01 · Profiling：建基线 / 定位 kernel / 算利用率 / 判断瓶颈

## 硬件天花板（ gfx938 = CDNA3）—— 双峰值 roofline

| 参数 | 值 | 含义 |
|------|-----|------|
| HBM 峰值 | **1.8 TB/s** | 带宽墙；memory-bound 的天花板 |
| **FP8 算力峰值** | **560 TFLOPS** | 算力墙；compute-bound 的天花板 |
| L2 | 8 MB | 权重一次全流过，L2 装不下，别指望 |
| LDS | 64 KB/CU | 只够 1 WG/CU，split-K 无空间 |
| CU | 64 | tile 数 ≪ 64 时并行度不足 |

## 双利用率公式（两个都算，看哪个先到墙）

- `weight_bytes = N × K`（fp8 = 1B/元素，decode 场景权重是唯一大流量）
- `有效带宽 = weight_bytes / kernel_time`  →  **带宽利用率 = 有效带宽 / 1.8 TB/s**
- `FLOPS = 2 × M × N × K / kernel_time`    →  **算力利用率 = FLOPS / 560 TFLOPS**

## 判据：decode GEMM 几乎总是 memory-bound

decode GEMM 算术强度 ≈ **2·M FLOP/byte**（权重 N×K 占绝对大头）。屋顶线交点：

```
AI_ridge = 560 TFLOPS / 1.8 TB/s ≈ 311 FLOP/B  →  M_ridge ≈ 155
```

- **M < 155（decode 的 M=32/64/128 全在这）→ memory-bound**，FLOPS 天花板 = 1.8TB/s × 2M：
  | M | 带宽墙对应 FLOPS 天花板 | 离 560 TFLOPS 算力墙 |
  |---|---|---|
  | 32 | ~115 TFLOPS | 远没到 |
  | 64 | ~230 TFLOPS | 远没到 |
  | 128 | ~460 TFLOPS | 接近但仍在带宽墙 |
- **M > 155 → compute-bound**，天花板 = 560 TFLOPS。

**"满足"的判据 = 对 shape 各自的现实上限，不是对 1.8TB/s**。
M 越小算术强度越低、越贴 memory-bound 墙，能打到越高利用率；M 越大离墙越远，天花板反而降。
所以按 **shape × M** 判"够好"，下表是常见场景的现实达标线（gate_up 为已确认经验值，其余需在本机实测补）：

| shape | M≤16 | M≤64 | 备注 |
|-------|------|------|------|
| **gate_up** (N=51200 K=5120) | **~80%** | **~70%** | 权重 262MB ≫ L2，最典型可 patch 对象 |
| down (N=5120 K=25600) | 待实测 | 待实测 | 权重 131MB |
| qkv / o | 待实测 | 待实测 | 权重 <8MB，slc 通常无效（见 02-tuning） |

- 利用率**低于**对应 shape/M 的达标线 → 还有带宽空间可挖 = 值得 patch 的理由。
- 利用率**达到/逼近**达标线 → 该 shape 基本到头，再优化 GEMM 本身无意义，转攻别的 kernel 或 selection。
- 新 shape 先在本机跑基线定自己的达标线，别直接套 gate_up 的 70/80。

## 建基线流程

1. 同一张卡 / 同一会话，未 patch 的 kernel 先跑一遍（`scripts/bench.py <name> <N> <K> <M> ref` 或 hipblaslt_layout_bench）。
2. **同机同口径是铁律**：隔离 bench 的数字只能和同一台机器的隔离 bench 比。禁止拿别的机器/生产口径对比（本项目踩过坑：把机器差算成 patch 收益）。
3. 记录环境：`.co` md5、`HIP_VISIBLE_DEVICES`、DTK 版本、bench 参数（rounds/iters）。缺环境 = 无法复现 = 数字作废。
4. warmup ≥ 3 轮，取中位数（bench.py 已做）。

### 基线实测参考（同一台机器，layout_bench 隔离测量）

| kernel tile | shape (N K M) | 时间 | 权重 | 有效带宽 | 利用率 |
|-------------|---------------|------|------|----------|--------|
| gate_up MT128x64x256 | 51200×5120×64 | ~222µs | 262MB | ~1017 GB/s | ~56% |
| down MT96x64x256 | 5120×25600×64 | ~126µs | 131MB | ~831 GB/s | ~46% |
| gate_up MT128x128x256 | 51200×5120×128 | ~308µs | 262MB | — | — |
| down MT96x128x256 | 5120×25600×128 | ~173µs | 131MB | — | — |

用途：新机器上跑基线若明显偏离（如 gate_up M64 远快于 222µs），先怀疑机器/口径，再怀疑代码。

## 按名定位 kernel（VA 换机器会变）

命名规则：

```
Cijk_Alik_Bljk_F8BS_MT{tileM}x{tileN}x{tileK}_TT*_WG*_MFWGS*_WGM*_bias_channelwise
```

解析方法：

1. `llvm-readelf -s <.co>` 找 FUNC symbol → 得到 kernel 名 + VA。
2. kernel body 区间 = `[VA, 下一个更大 VA 的 FUNC)`。
3. `file_off = VA - 0x1000`（偏移随 DTK 变，换 DTK 先实测）。

## 判断瓶颈

- decode 场景（M 小）几乎总是 **memory-bound**：权重流式读取，算术强度极低。
- 利用率 < 60% → 搬得慢，不是算得慢；优化方向是访问模式/缓存提示/调度，不是加指令。
- 确认手段 **MMA-NOP 实验**：把 kernel 里的 MMAC 指令换成 no-op，若 kernel 时间不变 = 计算零开销，确认 100% memory-bound。
- 顺带查 tile 数是否打满 64 CU：`(N/tileN) × (K/tileK)`（无 split-K 时）远小于 64 → 并行度不足是独立瓶颈。

## 常见坑

- **基线口径混用**：生产口径 vs 隔离 bench、A 机器 vs B 机器，数字不可比。
- **只看单个 shape**：M=64 和 M=128 行为不同（基线参考里 M 翻倍时间 +39~37%），达标判断看全部相关 kernel/shape。
- **不记录环境**：.co md5 / GPU 号 / DTK 版本 / rounds 缺一样，事后无法复现，数字作废。
- **warmup 不够**：首轮含冷启动，必须 ≥ 3 轮取中位数。

## bf16 GEMM 实测（qwen3-32b shape，干净独占卡 L2-flush 冷缓存）

**bf16 走 Tensile 的 Cijk kernel（`TensileLibrary_gfx938.co`, ~774MB），不是 FusedMatmul。**
工具：`tests/bw_util_bench.py`（torch.matmul + 空闲卡检查 + L2 flush），或 `hipblaslt-bench --print_kernel_info`。

**生产口径**：bf16 qwen3-32b 类 MoE，**生产主场景 = M=32（decode 并发）**（M=1/MTP 数据有水分，已弃用，不拿它做决策）。
带宽口径与上文一致：有效带宽 GB/s = 权重字节 / µs（bf16 = 2B/元素），利用率 = GB/s ÷ 1.8TB/s。

| shape（权重） | M=16 | M=32 | M=64 | M=128 |
|---|---|---|---|---|
| gate_up (524MB) | 76.4% | 68.3% | 69.6% | 49.2% |
| down (262MB) | 82.5% | 73.6% | 54.3% | 30.1% |
| qkv (104MB) | 75.4% | 67.2% | 52.8% | 42.7% |
| o (83MB) | 55.3% | 50.4% | 43.1% | 31.6% |

结论：
1. **生产 M=32 基本达标**（gate_up 68.3 / down 73.6 / qkv 67.2，贴 M≤64 的 70% 线）。o 50% 是小 N ramp-limited，正常。
2. **曾在共享卡看到 "M=64 断崖"（39%/12%）是争抢污染，不是 kernel 选错 tile**——干净卡 M=64 与 M=32 持平。**计时必须独占空闲卡**（hy-smi 确认目标卡 HCU 占用 = 0，脚本已内置检查，不闲拒跑），否则数字作废。
3. **slc 对 bf16 有效，A/B 实测 -6.2%（M=32）**。gate_up winner kernel = `Custom_Cijk_..._MT128x32x128_SE_K1_thread384_ccyBlockSize384_gfx938`，在 `library_gfx938/TensileLibrary_gfx938.co`，VA=0x1abdb400（file_off=VA-0x1000）。
   - 该 kernel 权重流式加载是 K-loop 里 **20 条 `buffer_load_dwordx4 ... idxen`，编码 E05C3000**（不是 lds 型！prologue 只有 8 条 lds 型 E05D1000，影响小）。
   - slc 位对 idxen 型同样适用：**E05C3000 → E05E3000（+0x20000）**，同 fp8 的 bit17。
   - A/B（卡4，5 轮交替顺序排除先后混杂，L2 flush）：clean 422.4µs → patched 396.3µs，**-26µs / -6.2%**（1241→1323 GB/s，69.0%→73.5%）。patched 先跑的轮次同样快，非顺序假象。
4. **重定向 A/B 的坑（踩过的）**：`HIPBLASLT_TENSILE_LIBPATH` 必须指向**完整目录**（全部 .co + .dat），只放 TensileLibrary 会丢 Custom kernel 族 → heuristic 降级选非 Custom 慢 kernel（MT256x128x16，642µs），把"selection 变了"误判成 patch 问题。判据：重定向后先确认 winner kernel 名与默认路径一致。
5. M=128 全 shape 掉到 30-49%（tile 换到 ×128），是下一档可优化点，但生产 M=32 不受影响。
