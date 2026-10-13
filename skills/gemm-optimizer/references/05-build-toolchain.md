# 05 · 编译工具链与 bench 流程：库加载 → 定位 kernel → patch → A/B 验证

> 覆盖完整链路：**hipblaslt 加载哪套库 → hipblaslt-bench 怎么用 → .co 怎么读/怎么改 → 重定向 A/B 怎么做**。
> 03-patching 讲位级改法（slc/vmcnt/Layout A 的编码），本篇讲**工具和流程**。
> 全部结论来自本机（DTK clang 18.0.0，卡 4 独占）实跑验证。

## 结论先行

1. **改已有 kernel 不需要 Tensile、不需要重编译任何库**：反汇编 .co 里 kernel → 改字节/改 .s → 拼回 .co 即生效。
2. **默认加载哪套库别假设**：`/opt/dtk/lib/hipblaslt/` 下多套 `library_*`，hipblaslt 内部按数据类型/heuristic 选（fp8 生产 GEMM 走 `library_bw_nmz`，bf16/fp16 走 `library_gfx938` 的 Tensile 库）。**用 `hipblaslt-bench --print_kernel_info` 看实际 kernel 名确认**，kernel 名带哪个族（`Custom_Cijk` vs `Cijk` vs `FusedMatmul`）就说明走了哪套。
3. **A/B 必须重定向完整目录 + 交替顺序 ≥5 轮取中位**，重定向后第一步核对 winner kernel 名一致，否则 A/B 无效。
4. **计时前 `hy-smi` 确认独占空闲卡**，共享卡数据全部作废（本项目踩过）。

## 库文件全景与默认加载规则（重要修正）

`/opt/dtk/lib/hipblaslt/` 下多套库并存（本机实测）：

| 目录 | 关键文件 | 谁走这套 |
|------|---------|---------|
| `library_gfx938/` | `TensileLibrary_gfx938.co` **774MB** + `TensileLibrary_gfx938.dat`（含 **Custom_Cijk** 系列）；另有 56MB `FusedMatmulKernel_gfx938.co` | **bf16/fp16 生产 GEMM**（Tensile Cijk 族，winner 如 `Custom_Cijk_..._MT128x32x128`） |
| `library_bw_nmz/` | `TensileLibrary_gfx938.co` 128MB + **`FusedMatmulKernel_gfx938.co` 24MB** | **fp8 生产 GEMM**（FusedMatmul 族） |
| `library_bw/`、`library_gfx936/` 等 | 同构文件，gfx936 老卡/其他变体 | 别拿错 gfx 目录 |
| `library/`、`library_gfx926/`、`library_gfx928/`、`library_km/` | — | 其他 arch/用途，本项目不碰 |

- **默认加载哪套由 hipblaslt 内部按数据类型/heuristic 决定，不要假设**。同一 shape 换个 dtype 可能换库。唯一可靠判据：
  `hipblaslt-bench --print_kernel_info` 打出的 kernel 名 + 用 `llvm-readelf -s` 确认该 kernel 在哪个 .co 里。
- 环境变量 **`HIPBLASLT_TENSILE_LIBPATH=<dir>`** 可重定向加载目录（A/B 的核心机制，见下节）。
- 运行时 `libhipblaslt.so` 按 `library_*` 目录加载 .co，kernel 按名字经 `*Name.txt`/`Index*.dat` 索引调用 → **直接替换 .co 文件即生效，无需重编译任何库**。

## hipblaslt-bench：编译/验证流程的核心入口

路径：`/opt/dtk/lib/hipblaslt/benchmark_tool/hipblaslt-bench`（`hipblaslt_common.yaml`/`gentest.py` 可批量生成用例）。

完整示例（bf16 gate_up，N=51200 K=5120 M=32，独占卡 4）：

```bash
B=/opt/dtk/lib/hipblaslt/benchmark_tool/hipblaslt-bench
HIP_VISIBLE_DEVICES=4 $B -m 51200 -n 32 -k 5120 \
  --lda 5120 --ldb 5120 --ldc 51200 --ldd 51200 \
  --transA T --transB N \
  -r bf16_r --a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r \
  --compute_type f32_r --activation_type None \
  --iters 100 --cold_iters 20
```

关键用法：

| 项 | 说明 |
|----|------|
| 输出 | CSV，**最后一列 = 时间（µs）** |
| `--print_kernel_info` | 打 solution/kernel 名/index——**定位 kernel 与 A/B 一致性核对的唯一入口** |
| `--verify`（`-v`） | GPU 结果对 CPU 校验，CSV 增加 norm_error 列（atol/rtol 判据） |
| `--requested_solution -1` | 枚举该 shape 全部 solution（注意：枚举列表 index 在重定向环境下同样有效，但**枚举里含不含 Custom kernel 取决于当前加载的库**——库不完整时枚举里就没有 Custom） |
| 计时口径 | bench **自带 L2 flush**（`--flush` 另加 icache flush）；`--iters`/`--cold_iters` 控制计时轮次/冷启动轮数 |

## 汇编工具链（本机实测路径）

| 工具 | 路径 | 作用 |
|------|------|------|
| 汇编器（权威） | `/opt/dtk/aillvm/bin/clang`（**18.0.0**，dcu-aicc fork） | .s → .o，gfx938 目标 |
| 旧汇编器（勿混用） | `/opt/dtk/llvm/bin/clang`（17.0.0） | 本机实测同 .s 输出字节相同，但**编码由汇编器版本决定，一律用 aillvm 的 18**，别混 |
| 读符号表 | `/opt/dtk/llvm/bin/llvm-readelf -s` | 按 kernel 名取 VA（FUNC 符号） |
| 反汇编 | `/opt/dtk/aillvm/bin/llvm-objdump -d` | **只在 aillvm 里有**（`/opt/dtk/llvm/bin/llvm-objdump` 不存在，别找错目录） |
| 提 .text | `/opt/dtk/aillvm/bin/llvm-objcopy -O binary --only-section=.text` | .o → 纯机器码字节 |
| 卡占用 | `hy-smi`（`/opt/hyhal/bin/hy-smi`） | **计时前必须确认独占空闲卡**——共享卡争抢会让数字全部作废（本项目" M=64 断崖"即共享卡假象） |
| `dccobjdump` | `/opt/dtk/llvm/bin/dccobjdump` | 是 "dcc fat binary tool"，**不是** llvm-objdump，反汇编 .co 用不上 |

**.co 文件本体**（实测）：普通 ELF64（magic `7f454c46`），单一大 `.text` 段 + dynsym 符号表，llvm 工具全家桶直接可用。符号表里 kernel 的 VA 是合成地址：**`file_off = VA − 0x1000`**（本机 DTK 的 .text 段基址差，换 DTK 先 `llvm-readelf -S` 实测重算）。kernel body 区间 = `[VA, 下一个更大 VA 的 FUNC)`。

## A/B 重定向方法论（本项目实测踩坑后定，务必照做）

机制：`HIPBLASLT_TENSILE_LIBPATH=<dir>` 让 bench 从 `<dir>` 加载库。A = clean 目录，B = patched 目录。

1. **重定向目录必须是完整目录（全部 .co + .dat）**。
   - **坑（实测）**：只放 `TensileLibrary_gfx938.co` 会丢 Custom kernel 族 → heuristic 降级选非 Custom 慢 kernel——gate_up M=32 从 `Custom ... MT128x32x128` **422µs 掉到标准 `MT256x128x16` 642µs**——把"selection 变了"误判成 patch 问题。
   - **省空间做法**：
     - clean 目录 = 对原库目录所有文件 `ln -sf`（不占空间、天然未改）；
     - patched 目录 = 只放 patch 后的那 **1 个 .co 真文件**，其余全软链。
2. **重定向后第一步**：`--print_kernel_info` 核对 winner kernel 名与默认路径一致。**不一致 = A/B 无效，停下来查目录**。
3. **A/B 必须交替顺序**：奇数轮 A 先、偶数轮 B 先，**≥5 轮取中位**，排除冷热/先后混杂。判据：**B 先跑的轮次 B 也照样快才是真收益**（bf16 slc -6.2% 即以此法确认：clean 422.4µs → patched 396.3µs，patched 先跑的轮次同样快）。

```bash
# 目录准备（以 library_gfx938 为例）
mkdir -p /tmp/ab_clean /tmp/ab_patched
for f in /opt/dtk/lib/hipblaslt/library_gfx938/*; do
  ln -sf "$f" /tmp/ab_clean/; ln -sf "$f" /tmp/ab_patched/
done
cp patched_TensileLibrary_gfx938.co /tmp/ab_patched/   # 只这 1 个是真文件

# 一轮 A/B（交替）
HIPBLASLT_TENSILE_LIBPATH=/tmp/ab_clean   HIP_VISIBLE_DEVICES=4 $B ...   # 轮1: A 先
HIPBLASLT_TENSILE_LIBPATH=/tmp/ab_patched HIP_VISIBLE_DEVICES=4 $B ...
HIPBLASLT_TENSILE_LIBPATH=/tmp/ab_patched HIP_VISIBLE_DEVICES=4 $B ...   # 轮2: B 先
HIPBLASLT_TENSILE_LIBPATH=/tmp/ab_clean   HIP_VISIBLE_DEVICES=4 $B ...
# ... 共 ≥5 轮，各取中位
```

## 端到端流程（完整顺序，别跳步）

```bash
B=/opt/dtk/lib/hipblaslt/benchmark_tool/hipblaslt-bench
A=/opt/dtk/aillvm/bin/clang; D=/opt/dtk/aillvm/bin/llvm-objdump
C=/opt/dtk/aillvm/bin/llvm-objcopy; R=/opt/dtk/llvm/bin/llvm-readelf
```

| 步 | 动作 | 命令/判据 |
|----|------|----------|
| a | **定位 kernel**：确认走哪套库、winner 是谁 | `HIPBLASLT_TENSILE_LIBPATH=（不设）$B ... --print_kernel_info` 拿 kernel 名 |
| b | **取 VA / 定区间** | `$R -s <.co> \| grep -F <kernel_name>` → FUNC 行第二列 = VA；区间 = `[VA, 下一符号 VA)`；`file_off = VA - 0x1000` |
| c | **反汇编** | `$D -d <.co> --start-address 0x<VA> ... > /tmp/dump.txt`（大文件必须重定向到文件再 grep，别直接管道刷屏） |
| d | **改字节** | slc 位 patch（E05D1000→E05F1000 / E05C3000→E05E3000）、vmcnt（低 word `0x0f70\|N`）、Layout A 常量等——位级细节见 03-patching；走汇编路线则 `$A -x assembler --target=amdgcn-amd-amdhsa -mcode-object-version=4 -mcpu=gfx938 -mwavefrontsize64 -c x.s -o x.o && $C -O binary --only-section=.text x.o x.text` |
| e | **建 clean + patched 目录** | 软链规则见上节；patched 前 `md5sum` 备份原 .co |
| f | **核对 winner 一致** | 两目录各跑 `--print_kernel_info`，kernel 名必须与默认路径一致 |
| g | **验正确性** | `$B ... --verify`，看 norm_error 列（atol/rtol 判据）；正确性过 ≠ 布局对（Layout A 还要权重重排配对） |
| h | **交替 A/B ≥5 轮** | 取中位；B 先的轮次也要快才算数 |
| i | **保留/回滚** | 有收益保留；无收益回滚，并用 `md5sum` 校验 .co 已恢复到备份值 |

## Tensile 的角色（生成 vs 修改，别混）

- **生成路径**（我们用不到）：Tensile 源码 `/models/yuebin/tensile`（ROCm Tensile fork），产物 `TensileLibrary_*.co/.dat` + `FusedMatmulKernel*.co` 在各 `library_*` 目录。换 DTK 时这些 kernel 由 Tensile 用该 DTK 的编译器重新生成——这解释了为什么二进制编码随 DTK 变。
- **修改路径**（我们）：反汇编已有 .co 里的 kernel → 改字节/.s → 拼回 .co。全程不碰 Tensile。

## 版本绑定坑（跨机器/跨 DTK 复现）

kernel 二进制编码（slc 位位置、`vmcnt(N)` 低 word = **`0x0f70|N`**：`vmcnt(0)`=0x0f70、`vmcnt(4)`=0x0f74，本机 DTK clang **18.0.0** objdump 实证）由**汇编该 kernel 的 DTK/clang 版本**决定，.co、汇编器、驱动三者必须同版本。同机 clang 17 汇编同 .s 恰好字节相同——**巧合不是保证**，跨 DTK 一律不假设。

**换机器/换 DTK 前必做**：把交付的 `*_original.s` 在目标机汇编、提 .text，与目标机原 .co 对应区逐字节 cmp。**0 diff = 同 build**，旧 patch 可沿用；**有 diff = 重新反汇编、重新确认编码、重新 patch**（别搬旧 .o/.text）。

## 最小可复现汇编验证（换机第一件事，~1 分钟）

```bash
cat > /tmp/v.s <<'EOF'
	.globl	kern
kern:
	s_waitcnt	vmcnt(4)
	s_waitcnt	vmcnt(0)
	s_nop	0
	s_endpgm
EOF
/opt/dtk/aillvm/bin/clang -x assembler --target=amdgcn-amd-amdhsa -mcode-object-version=4 \
  -mcpu=gfx938 -mwavefrontsize64 -c /tmp/v.s -o /tmp/v.o
/opt/dtk/aillvm/bin/llvm-objdump -d /tmp/v.o
# 期望（本机 DTK）：vmcnt(4)→0x...0f74, vmcnt(0)→0x...0f70, s_endpgm→0xbf810000
# 与锚点不一致 → 该机编码不同，按上节字节比对流程重做
```

注意：.s 里写 `.amdtarget gpu` 会报 `unknown directive`——真实反汇编产物没有文件头，直接是 `label:` + 指令，别自己加头。

## 常见坑汇总

- **假设库加载规则**：不同 dtype 走不同 `library_*`，永远以 `--print_kernel_info` 的 kernel 名为准。
- **重定向目录不完整**：丢 Custom 族 → heuristic 降级，selection 变化伪装成 patch 效果（422→642µs 事故）。
- **A/B 不交替**：clean 恒在先 → 冷热混杂，收益被高估。
- **共享卡计时**：`hy-smi` 没查就测 → 数据作废。
- **拿错 llvm 工具目录**：`llvm-objdump` 只在 `/opt/dtk/aillvm/bin`；`dccobjdump` 不是 objdump。
- **换 DTK 搬旧 .o**：编码随汇编器版本变，必须字节比对后才可沿用。
