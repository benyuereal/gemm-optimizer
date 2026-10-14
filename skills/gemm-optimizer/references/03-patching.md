# 03 · Patching：改汇编 / 拼进 .co / 权重重排 / 验证 / 诊断"没提速"

> 手段选型看 02-tuning；这里只讲**怎么落位**和**落了没生效怎么办**。
> 全流程优先 `scripts/patch_co.sh`（备份→汇编→按名拼→byte-exact→正确性→计时→回滚，7 步自动）。

## 工具链（原样用，别换版本）

| 用途 | 工具 |
|------|------|
| 生产库 (FP8 FusedMatmul) | `/opt/dtk/lib/hipblaslt/library_bw_nmz/FusedMatmulKernel_gfx938.co`（目录随 DTK 变，`find /opt/dtk -name 'FusedMatmulKernel*.co'` 定位） |
| 生产库 (bf16/fp16 Tensile Cijk) | `/opt/dtk/lib/hipblaslt/library_gfx938/TensileLibrary_gfx938.co`（bf16 slc 主战场，一键走 `scripts/slc_ab.sh`） |
| 汇编 | `/opt/dtk/aillvm/bin/clang -x assembler --target=amdgcn-amd-amdhsa -mcode-object-version=4 -mcpu=gfx938 -mwavefrontsize64 -c x.s -o x.o` |
| 读符 | `/opt/dtk/llvm/bin/llvm-readelf -s`（符号表） |
| 反汇编 | `/opt/dtk/aillvm/bin/llvm-objdump -d`（大文件先 `> /tmp/dump.txt` 再 grep，别直接管道刷屏） |
| 提 .text | `/opt/dtk/aillvm/bin/llvm-objcopy -O binary --only-section=.text x.o x.text` |

**汇编器版本决定指令编码**，与 DTK 绑定：跨 DTK 拿旧 .o/旧编码 = 字节错位。

## 按名定位 + 拼接（机械步骤）

1. `llvm-readelf -s <.co>` 解析目标 kernel 的 **VA**（FUNC symbol，按名字找；VA 换机器会变）。
2. body 区间 = `[VA, 下一个更大 VA)`；`region_size = 下一个 VA - VA`。
3. `file_off = VA - 0x1000`（偏移随 DTK 变，换 DTK 先实测）。
4. splice：

```python
b = bytearray(open(co, 'rb').read())
text = open(tag_text, 'rb').read()
assert len(text) <= region_size, "超了 = kernel 名选错"
off = VA - 0x1000
b[off:off+len(text)] = text
open(co, 'wb').write(bytes(b))
```

5. **byte-exact 校验**：`dd` 从 `file_off` 读 `len(text)` 字节，`cmp` 与 `.text` 必须 0 diff。
6. 表驱动批量：`kernels.txt` 每行 `kernel_name|N|K|M|src|target_us|reorder`（reorder=1 即 Layout A 需重排），`bash patch_co.sh <src_dir> [GPU_ID]`。

## 三条手段的位级改法（核心知识）

### slc（evict-first）

- 权重 `buffer_load_dwordx4` **word0 置 bit17（0x20000）**。**两种 lds type 都要会认**:
  - **lds 型**（prologue, 直接写 LDS）：`E05D1000 → E05F1000`（fp8 FusedMatmul 主战场）。
  - **idxen 型**（K-loop 流式, 向量地址）：`E05C3000 → E05E3000`（bf16 Tensile Cijk 主战场）。
- **先反汇编数清权重 load 分几组、大头在哪**:prologue lds 型通常只 8 条,K-loop idxen 型才是流式读大权重的主体(如 bf16 gate_up 20 条)。**改错组 = 只动了 prologue,收益趋零**。
- 只改**权重**流式 load;**activation** load 一律不动。
- M=128 的 weight 是 B-operand/idxen SET,slc 实测回退,别动。

### vmcnt

- node3 clang18 上 `vmcnt(N)` 低 word = `0x0f70 | N`：`vmcnt(0)`=0x0f70，`vmcnt(4)`=0x0f74。
- **编码随 DTK/汇编器版本变**：换 DTK 必须先汇编一个 4 指令最小片段、objdump 实测确认编码，再批量改。

### Layout A（blocked 权重布局，须配对重排）

- 权重 flat 地址 = `tile*(kt*NT*KT) + kt*(NT*KT) + n*KT + k`（NT=128, KT=256），用 `reorder_weights.py` 重排权重。
- kernel 侧改 3 个地址常量：`s44=256`、`s44=0x6400`、`s50: 0x100→0x8000`，外加 `vmcnt(0)→vmcnt(4)` 约 10 处。
- **配对铁律**：kernel patch 与权重重排必须同时生效，缺一个 = 算错；serving 侧重排后重启才完整生效。

## 换机器 / 换 DTK 前的字节比对（防搬错）

1. 目标机器上反汇编目标 kernel 区。
2. 把交付的 `*_original.s` 在目标机器上汇编、提 .text，与目标机器原 .co 对应区**逐字节比对**。
3. 0 diff = 同 DTK、编码一致，可放心沿用旧 patch；**有 diff = 必须重新反汇编、重新 patch**（编码/VA 都可能变）。

## 诊断"patch 了但不提速"（按此顺序，别跳步）

1. **基线是否同机同口径？** 最高频错误：拿隔离 bench 数字减别机器的数字 → 假造收益/假造无收益。
2. **patch 是否真在位？** 跑 `scripts/diagnose.py <clean.co> <patched.co> <kernel_name>` 数 slc/vmcnt(4) 条数和逐指令 diff；别只信"拼接成功"。patched 区 slc==0 而源码 .s 有 slc → 拼接错位或 kernel 名选错。
3. **权重是否配对？** Layout A 没跑 `reorder_weights.py` → 算错或带宽打不满（正确性过 ≠ 布局对）。
4. **该机器硬件是否响应 cache hint？** 对照组实验：12 条 slc vs 201 条 slc 计时完全一样 → 该 DCU/固件/DTK 对 evict-first 不响应 → **slc 路线该机判死**，别再堆变体。
5. **DTK/驱动版本差**：kernel 二进制与 DTK 版本绑定，.co、汇编器、驱动三者版本必须一致。
6. **重定向库丢了 kernel 族**：`HIPBLASLT_TENSILE_LIBPATH` 必须指向**完整目录**（全部 .co + .dat，可用 symlink 目录省空间：clean=全软链，patched=只 1 个真文件）。只拷 TensileLibrary 会丢 Custom kernel 族 → heuristic 降级选非 Custom 慢 kernel（实测 gate_up M=32：MT128x32x128 422µs 掉到 MT256x128x16 642µs）。**重定向后第一步：`--print_kernel_info` 核对 winner kernel 名与默认路径一致**，不一致 = A/B 无效。
7. **A/B 顺序混杂**：clean 恒在先/patched 恒在后 → 冷热混杂，收益被高估。必须**交替顺序**（奇数轮 A 先、偶数轮 B 先）跑 ≥5 轮取中位；B 先跑的轮次也照样快才是真收益（bf16 slc -6.2% 即以此法确认）。
