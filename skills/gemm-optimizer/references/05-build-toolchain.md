# 05 · 编译工具链：从 .s 到可放进 .co 的机器码

> 03-patching 讲"怎么拼"，这里讲**用什么工具、版本绑定在哪、换机器怎么验证**。
> 全部结论来自本机（DTK clang 18.0.0）实跑验证。

## 结论一句话

**改已有 kernel 只需要 DTK 的 clang 汇编 + llvm-objcopy/objdump/readelf 拼接，
不需要 Tensile、不需要 hipBLASLt 库本身、不需要重新跑 Tensile 生成。**
Tensile 是"生成"路径的上游工具；我们是"修改"路径，两条路径互不相干。

## 工具链表（本机实测路径）

| 工具 | 路径 | 作用 |
|------|------|------|
| 汇编器（权威） | `/opt/dtk/aillvm/bin/clang`（**18.0.0**，dcu-aicc fork） | .s → .o。gfx938 目标 |
| 旧汇编器（勿混用） | `/opt/dtk/llvm/bin/clang`（**17.0.0**） | 也能汇编同一 .s，本机实测输出字节相同，但**编码由汇编器版本决定，一律用 aillvm 的 18**，别混 |
| 读符号表 | `/opt/dtk/llvm/bin/llvm-readelf -s` | 按 kernel 名取 VA（FUNC 符号，size 0） |
| 反汇编 | `/opt/dtk/aillvm/bin/llvm-objdump -d` | 验证编码（vmcnt/slc 位） |
| 提 .text | `/opt/dtk/aillvm/bin/llvm-objcopy -O binary --only-section=.text` | .o → 纯机器码字节 |
| `dccobjdump` | `/opt/dtk/llvm/bin/dccobjdump` | 是 "dcc fat binary tool"，**不是** llvm-objdump，反汇编 .co 用不上它 |

## .co 文件本体（实测）

- 生产库：`/opt/dtk/lib/hipblaslt/library_bw_nmz/FusedMatmulKernel_gfx938.co`
  （`find /opt/dtk -name 'FusedMatmulKernel*.co'` 定位；另有 gfx936 目录，别拿错）。
- 是**普通 ELF64**（magic `7f454c46`，etype=SharedObject），单一大 `.text` 段
  （vaddr 0x1cf300 / file_off 0x1ce300，~18MB）+ dynsym 符号表——所以 llvm 工具全家桶直接可用。
- 符号表里 kernel 的 "VA" 是合成地址：**file_off = VA − 0x1000**（本机 DTK 的 .text 段基址差，
  换 DTK 先 `llvm-readelf -S` 实测重算）。
- 运行时 `libhipblaslt.so`（`/opt/dtk/hipblaslt/lib/`，符号链接）按 `library_*` 目录加载 .co，
  kernel 按名字经 `FusedMatmulKernelName.txt`/`Index*.dat` 索引调用 → **直接替换 .co 文件即生效，
  无需重编译任何库**。可选环境变量 `HIPBLASLT_TENSILE_LIBPATH` 可重定向库目录。

## 机械流程（完整命令行，本机已跑通）

```bash
A=/opt/dtk/aillvm/bin/clang          # 汇编
D=/opt/dtk/aillvm/bin/llvm-objdump   # 验证
C=/opt/dtk/aillvm/bin/llvm-objcopy   # 提 .text
R=/opt/dtk/llvm/bin/llvm-readelf     # 符号

# 1. 反汇编出原 kernel 区 → x_original.s（llvm-objdump -d 大文件先重定向到文件再 grep）
# 2. 改 x_original.s → x_patched.s
# 3. 汇编（.s 无文件头指令，直接是 label: + 指令，见"最小验证"）
$A -x assembler --target=amdgcn-amd-amdhsa -mcode-object-version=4 \
   -mcpu=gfx938 -mwavefrontsize64 -c x_patched.s -o x.o
# 4. 提 .text 字节
$C -O binary --only-section=.text x.o x.text
# 5. 按名取 VA：$R -s <.co> | grep -F <kernel_name>  → FUNC 行第二列
# 6. splice（off = VA-0x1000，长度 ≤ 下一符号 VA − VA），见 03-patching
# 7. 字节校验：dd 读 .co[off:off+len] cmp x.text → 必须 0 diff
# 8. 正确性 ≤1 ULP、计时、失败回滚（md5 备份）
```

## Tensile 的角色（生成 vs 修改，别混）

- **生成路径**（我们用不到）：Tensile 源码 `/models/yuebin/tensile`（ROCm Tensile fork），
  产物 `TensileLibrary_*.co/.dat` + `FusedMatmulKernel*.co` 在 `library_*` 目录。
  换 DTK 时这些 kernel 是 Tensile 用该 DTK 的编译器**重新生成**的——这解释了为什么
  二进制编码随 DTK 变。
- **修改路径**（我们）：反汇编已有 .co 里的 kernel → 改 .s → DTK clang 汇编 → 拼回 .co。
  全程不碰 Tensile；Tensile 只在"要换 DTK/重建 kernel 库"时才相关。

## 版本绑定坑（跨机器/跨 DTK 复现）

kernel 二进制编码（slc 位位置、`vmcnt(N)` 低 word = `0x0f70|N` 等）由**汇编该 kernel 的
DTK/clang 版本**决定，.co、汇编器、驱动三者必须同版本。本机实测锚点：

- clang 18.0.0：`vmcnt(4)` → 低 word `0x0f74`，`vmcnt(0)` → `0x0f70`（objdump 实证）。
- 同机 clang 17 汇编同一 .s 恰好字节相同（本机实证）——但这是**巧合不是保证**，
  跨 DTK 一律不假设。

**换机器/换 DTK 前的字节比对（必做）**：

1. 目标机上把交付的 `*_original.s` 用目标机 clang 汇编、`llvm-objcopy` 提 .text；
2. 与目标机原 .co 对应 kernel 区（按名取 VA，`file_off = VA − 基址差`，基址差先用
   `llvm-readelf -S` 实测）**逐字节 cmp**；
3. **0 diff = 同 build**，旧 patch 可放心沿用；**有 diff = 编码/VA 都可能变，
   必须重新反汇编、重新确认 vmcnt/slc 编码、重新 patch**（别搬旧 .o、别搬旧 .text）。

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
# 与上表锚点不一致 → 该机编码不同，按上节字节比对流程重做
```

注意：.s 里写 `.amdtarget gpu` 会报 `unknown directive`——真实反汇编产物没有文件头，
直接是 `label:` + 指令，别自己加头。
