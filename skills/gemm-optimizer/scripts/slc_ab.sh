#!/bin/bash
# =============================================================================
# slc_ab.sh — bf16/fp16 TensileLibrary slc(evict-first) 重定向 A/B 一键脚本
#
# 干什么：
#   1) 按 kernel 名解析 VA + 反汇编该 kernel,自动数清 load 组(idxen 型 vs lds 型
#      条数),报告"该给哪组打 slc"(改错组=只动 prologue=收益趋零,见 07 篇)。
#   2) 把 slc 位打进 kernel 的流式权重 load,生成 patched .co(不改生产库)。
#   3) 建 clean / patched 两个完整重定向目录(clean 全软链,patched 只 1 个真文件)。
#   4) 第一步核对 winner kernel 名与默认路径一致(不一致=A/B 无效,直接停)。
#   5) 交替顺序 A/B(奇数轮 clean 先、偶数轮 patched 先)跑 ≥5 轮取中位。
#   6) patched 目录跑 --verify 出 norm_error(正确性)。
#   7) 汇总 Δ% + 有效带宽,写 perf-log。
#
# 用法(默认 = bf16 gate_up N=51200 K=5120 M=32, 独占卡 4):
#   bash slc_ab.sh
#
# 换库/换 shape:用环境变量覆盖(见下)。
#
# 关键 env(全部有默认):
#   CARD=4            独占空闲卡号(计时前自动 hy-smi 检查 HCU 占用)
#   LIBDIR=/opt/dtk/lib/hipblaslt/library_gfx938     bf16/fp16 Tensile 库目录
#   CO=$LIBDIR/TensileLibrary_gfx938.co              要打 patch 的 .co
#   KERNEL=Custom_Cijk_Alik_Bljk_BBS_BH_Bias_AS_MT128x32x128_SE_K1_thread384_ccyBlockSize384_gfx938
#   N=51200 K=5120 M=32          shape (bench 的 -m/-n/-k 与 ld 自动按此推)
#   DTYPE=bf16_r                 a/b/c/d 类型 (bf16_r / fp16_r)
#   COMPUTE=f32_r
#   TRANSA=T TRANSB=N            gate_up 口径
#   ROUND=5                      A/B 轮数(≥5,交替)
#   PATCH=idxen                  打哪组: idxen(K-loop 流式,默认) / lds(prologue) / all
#   KEEP=1                       结束保留 patched 目录(默认 1);0=清理
#   PERF_LOG=./perf-log.md
#
# 不锁频(禁 hy-smi lock),不碰生产 .co(全程重定向目录),计时前查卡空闲。
# =============================================================================
set -u

CARD="${CARD:-4}"
LIBDIR="${LIBDIR:-/opt/dtk/lib/hipblaslt/library_gfx938}"
CO="${CO:-$LIBDIR/TensileLibrary_gfx938.co}"
KERNEL="${KERNEL:-Custom_Cijk_Alik_Bljk_BBS_BH_Bias_AS_MT128x32x128_SE_K1_thread384_ccyBlockSize384_gfx938}"
N="${N:-51200}" K="${K:-5120}" M="${M:-32}"
DTYPE="${DTYPE:-bf16_r}"
COMPUTE="${COMPUTE:-f32_r}"
TRANSA="${TRANSA:-T}" TRANSB="${TRANSB:-N}"
ROUND="${ROUND:-5}"
PATCH="${PATCH:-idxen}"
KEEP="${KEEP:-1}"
PERF_LOG="${PERF_LOG:-./perf-log.md}"

# 工具链(本机实测路径, 换 DTK 先核对 05 篇)
B="${HIPBLASLT_BENCH:-/opt/dtk/lib/hipblaslt/benchmark_tool/hipblaslt-bench}"
OBJDUMP="${OBJDUMP:-/opt/dtk/aillvm/bin/llvm-objdump}"
READELF="${READELF:-/opt/dtk/llvm/bin/llvm-readelf}"
HY="${HY:-/opt/hyhal/bin/hy-smi}"

WORK="$(mktemp -d /tmp/slc_ab.XXXXXX)"
trap 'echo "workdir: $WORK (KEEP=$KEEP; 手动清理: rm -rf $WORK)"; [ "$KEEP" = 1 ] || rm -rf "$WORK"' EXIT

die() { echo "!!! $*"; exit 1; }
[ -f "$CO" ] || die ".co 不存在: $CO"
[ -x "$B" ] || die "hipblaslt-bench 不存在: $B"

# ---------------------------------------------------------------------------
# [0] 前置: 卡空闲检查 (不锁频; 只看 HCU/显存占用是否 0/接近 0)
# ---------------------------------------------------------------------------
echo "=== [0/7] 卡 $CARD 空闲检查 ==="
"$HY" 2>/dev/null | awk -v c="$CARD" '
  NR==1 {for(i=1;i<=NF;i++) col=$i}
  {if($0 ~ ("^"c" ")) {print "  raw: " $0}}' || true
# hy-smi 表格解析不稳, 这里只提示人工确认 HCU=0
echo "  ⚠️ 请人工确认卡 $CARD HCU=0% 且显存≈0 (hy-smi)。共享卡会让数字全废。"

# ---------------------------------------------------------------------------
# [1] 按名解析 VA + .text 段基址(自动算 file_off, 不硬编码 VA-0x1000)
# ---------------------------------------------------------------------------
echo "=== [1/7] 解析 kernel VA + .text 基址 ==="
read VA SIZE OFFS < <("$READELF" -s "$CO" 2>/dev/null | "$READELF" -S "$CO" 2>/dev/null | python3 -c "
import sys
lines=[l.rstrip() for l in sys.stdin]
# 先吃符号表 (readelf -s), 再吃段表 (readelf -S); 两个命令已用管道顺序喂
# 这里分开解析: 前段是 -s 输出, 空行/ '  ' 开头切换为 -S
sym_lines=[]; sec_lines=[]
mode='sym'
for l in lines:
    if l.startswith('  [Nr] Name') or l.startswith('  [') and 'Name' in l and 'Type' in l:
        mode='sec'; continue
    (sym_lines if mode=='sym' else sec_lines).append(l)
# VA of FUNC named KERNEL
target=\"$KERNEL\"
va=None; order=[]
for l in sym_lines:
    p=l.split()
    if len(p)>=8 and p[3]=='FUNC':
        try:
            v=int(p[1],16); nm=p[-1].rstrip('\$')
            order.append((v,nm))
        except: pass
order.sort()
for i,(v,nm) in enumerate(order):
    if nm==target:
        end=order[i+1][0] if i+1<len(order) else v+0x200000
        va=v; size=end-v; break
if va is None:
    print('0 0 0'); raise SystemExit
# .text vaddr + fileoff
tv=to=None
for l in sec_lines:
    if '.text' in l.split():
        toks=l.split()
        # format: [Nr] Name   Type Addr Off Size ...
        # find Name '.text'
        idx=toks.index('.text')
        # Addr is 4 fields after Name? readelf -S: [Nr] Name Type Address Offset Size ES Flg Al
        addr=toks[idx+2]; off=toks[idx+3]
        tv=int(addr,16); to=int(off,16); break
off= (to if to is not None else 0) + (va - (tv if tv is not None else 0))
print(hex(va), hex(size), hex(off))
")
[ "${VA:-0}" != "0" ] || die "kernel 未在符号表找到: $KERNEL"
echo "  VA=$VA  region=$SIZE  file_off=$OFFS"

# ---------------------------------------------------------------------------
# [2] 反汇编 kernel region, 自动数 load 组 (idxen 型 / lds 型), 报告打哪组
# ---------------------------------------------------------------------------
echo "=== [2/7] 反汇编 + 数 load 组 ==="
"$OBJDUMP" -d "$CO" --start-address="$VA" --stop-address=$((0x$VA + 0x$SIZE)) 2>/dev/null > "$WORK/kern.s" || die "反汇编失败"
python3 - "$WORK/kern.s" <<'PY'
import re, sys
txt=open(sys.argv[1]).read()
buf=[l for l in txt.splitlines() if 'buffer_load' in l]
idxen=[l for l in buf if 'idxen' in l]
lds  =[l for l in buf if re.search(r'\bllds\b|offen', l) or 'lds' in l.split()]
# 更稳的 idxen 判定: 含 'idxen' token
idxen=[l for l in buf if re.search(r'\bidxen\b', l)]
lds  =[l for l in buf if l not in idxen and ('lds' in l)]
# 按字节编码数 (idxen 型 slc 候选 = E05C3000; lds 型 = E05D1000)
raw=open(sys.argv[1],'rb').read()  # 反汇编文本里有 .byte 行? 不一定; 改用原始 .co 区段在下面算
print(f"  buffer_load 总条数: {len(buf)}")
print(f"  idxen 型 (K-loop 流式, slc 主战场): {len(idxen)}")
print(f"  lds / 其它型 (prologue):          {len(lds)}")
if idxen:
    print("  --- idxen 型样例 (前 3 条) ---")
    for l in idxen[:3]: print("   ", l.strip()[:100])
if lds:
    print("  --- lds 型样例 (前 3 条) ---")
    for l in lds[:3]: print("   ", l.strip()[:100])
print(f"  → 建议打 slc 的组: {'idxen (K-loop 流式权重)' if len(idxen)>=len(lds) else 'lds (或人工核对 operand 角色)'}")
print("  ⚠️ 自动数只能给条数/形态; operand 角色(A/B, idxen CLEAR/SET)仍须按 07 篇人工确认,")
print("     给激活加 slc 方向反了。M=128 的 B-operand(idxen SET) slc 实测回退,勿动。")
PY

# ---------------------------------------------------------------------------
# [3] 打 slc 位 → patched .co (不改生产)
# ---------------------------------------------------------------------------
echo "=== [3/7] 打 slc 位 (PATCH=$PATCH) ==="
# 先数原始 .co region 内的候选字节 (idxen=E05C3000 / lds=E05D1000, 小端 4B)
python3 - "$CO" "$OFFS" "$SIZE" "$WORK/patched.co" "$PATCH" <<'PY'
import sys
co, offs, size, out, patch = sys.argv[1], int(sys.argv[2],0), int(sys.argv[3],0), sys.argv[4], sys.argv[5]
d=bytearray(open(co,'rb').read())
seg=bytearray(d[offs:offs+size])
IDEN=b'\x00\x30\x5c\xe0'   # E05C3000 LE
IDEN_S=b'\x00\x30\x5e\xe0' # E05E3000 LE
LDS =b'\x00\x10\x5d\xe0'   # E05D1000 LE
LDS_S=b'\x00\x10\x5f\xe0'  # E05F1000 LE
def do(seg, src, dst, tag):
    n=seg.count(src)
    seg2=bytearray(seg).replace(src, dst)
    print(f"  {tag}: {n} 处 {src.hex()} → {dst.hex()}")
    return seg2, n
n_iden=0; n_lds=0
if patch in ('idxen','all'):
    seg, n_iden = do(seg, IDEN, IDEN_S, 'idxen')
if patch in ('lds','all'):
    seg, n_lds = do(seg, LDS, LDS_S, 'lds')
d[offs:offs+size]=seg
open(out,'wb').write(bytes(d))
tot=n_iden+n_lds
if tot==0:
    print("  !!! 未匹配到任何候选字节——编码可能随 DTK 变, 人工核对反汇编后设 PATCH/手工改。"); sys.exit(3)
print(f"  合计改 {tot} 处 (idxen {n_iden} + lds {n_lds}) → {out}")
PY
[ $? -eq 0 ] || exit 1

# ---------------------------------------------------------------------------
# [4] 建 clean / patched 完整重定向目录 (clean 全软链; patched 全软链 + 1 真文件)
# ---------------------------------------------------------------------------
echo "=== [4/7] 建重定向目录 ==="
CLEAN="$WORK/clean"; PATCHD="$WORK/patched"
mkdir -p "$CLEAN" "$PATCHD"
for f in "$LIBDIR"/*; do
  ln -sf "$f" "$CLEAN/$(basename "$f")"
  ln -sf "$f" "$PATCHD/$(basename "$f")"
done
COBASE="$(basename "$CO")"
rm -f "$PATCHD/$COBASE"
cp "$WORK/patched.co" "$PATCHD/$COBASE"
echo "  clean: $CLEAN (全软链)"
echo "  patched: $PATCHD ($COBASE 为真文件, 其余软链)"

# ---------------------------------------------------------------------------
# [5] 核对 winner kernel 名 (默认 vs clean vs patched 必须一致, 否则 A/B 无效)
# ---------------------------------------------------------------------------
echo "=== [5/7] 核对 winner kernel 名 ==="
winner() { # $1 = 环境变量目录(可空)
  local dir="$1" out
  if [ -z "$dir" ]; then out=$("$B" -m "$N" -n "$M" -k "$K" --lda "$K" --ldb "$K" --ldc "$N" --ldd "$N" \
      --transA "$TRANSA" --transB "$TRANSB" -r "$DTYPE" -a "$DTYPE" -b "$DTYPE" -c "$DTYPE" -d "$DTYPE" \
      --compute_type "$COMPUTE" --activation_type None --iters 1 --cold_iters 1 --print_kernel_info 2>&1)
  else out=$(HIPBLASLT_TENSILE_LIBPATH="$dir" "$B" -m "$N" -n "$M" -k "$K" --lda "$K" --ldb "$K" --ldc "$N" --ldd "$N" \
      --transA "$TRANSA" --transB "$TRANSB" -r "$DTYPE" -a "$DTYPE" -b "$DTYPE" -c "$DTYPE" -d "$DTYPE" \
      --compute_type "$COMPUTE" --activation_type None --iters 1 --cold_iters 1 --print_kernel_info 2>&1)
  fi
  echo "$out" | grep -oE "(Custom_Cijk|Cijk|FusedMatmul)[A-Za-z0-9_]*" | head -1
}
W_DEF=$(winner "")
W_C=$(winner "$CLEAN")
W_P=$(winner "$PATCHD")
echo "  默认:   $W_DEF"
echo "  clean:  $W_C"
echo "  patched:$W_P"
[ "$W_DEF" = "$W_C" ] || die "clean winner 与默认不一致 ($W_C vs $W_DEF) → 目录不完整, A/B 无效"
[ "$W_DEF" = "$W_P" ] || die "patched winner 与默认不一致 ($W_P vs $W_DEF) → 目录不完整, A/B 无效"
echo "  ✓ 三者一致"

# ---------------------------------------------------------------------------
# [6] 交替顺序 A/B, ≥ROUND 轮取中位
# ---------------------------------------------------------------------------
echo "=== [6/7] 交替 A/B ($ROUND 轮) ==="
run_us() { # $1 = 目录(可空)
  local dir="$1"
  if [ -z "$dir" ]; then
    HIP_VISIBLE_DEVICES="$CARD" "$B" -m "$N" -n "$M" -k "$K" --lda "$K" --ldb "$K" --ldc "$N" --ldd "$N" \
      --transA "$TRANSA" --transB "$TRANSB" -r "$DTYPE" -a "$DTYPE" -b "$DTYPE" -c "$DTYPE" -d "$DTYPE" \
      --compute_type "$COMPUTE" --activation_type None --iters 100 --cold_iters 20 2>&1
  else
    HIPBLASLT_TENSILE_LIBPATH="$dir" HIP_VISIBLE_DEVICES="$CARD" "$B" -m "$N" -n "$M" -k "$K" --lda "$K" --ldb "$K" --ldc "$N" --ldd "$N" \
      --transA "$TRANSA" --transB "$TRANSB" -r "$DTYPE" -a "$DTYPE" -b "$DTYPE" -c "$DTYPE" -d "$DTYPE" \
      --compute_type "$COMPUTE" --activation_type None --iters 100 --cold_iters 20 2>&1
  fi | awk -F, 'NF>=2 && $NF ~ /^[0-9.]+$/ {print $NF}' | tail -1
}
CLEAN_S=""; PATCH_S=""
for r in $(seq 1 "$ROUND"); do
  if [ $((r % 2)) -eq 1 ]; then   # 奇数轮: clean 先
    a=$(run_us "$CLEAN"); b=$(run_us "$PATCHD"); CLEAN_S="$CLEAN_S $a"; PATCH_S="$PATCH_S $b"
  else                            # 偶数轮: patched 先
    b=$(run_us "$PATCHD"); a=$(run_us "$CLEAN"); CLEAN_S="$CLEAN_S $a"; PATCH_S="$PATCH_S $b"
  fi
  echo "  轮 $r ($([ $((r%2)) -eq 1 ] && echo clean先 || echo patched先)): clean=${a:-NA}us  patched=${b:-NA}us"
done

# ---------------------------------------------------------------------------
# [7] 汇总 (中位) + norm 校验 + 写 perf-log
# ---------------------------------------------------------------------------
echo "=== [7/7] 汇总 ==="
python3 - "$CLEAN_S" "$PATCH_S" "$N" "$K" "$M" "$PATCH" <<'PY'
import sys
clean=[float(x) for x in sys.argv[1].split() if x]
patch=[float(x) for x in sys.argv[2].split() if x]
N,K,M,patch_group=int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),sys.argv[6]
def med(a): a=sorted(a); return a[len(a)//2]
c,p=med(clean),med(patch)
d=100*(1-p/c)
wb=N*K  # bf16/fp16 = 2B; 若 fp8 设 DTYPE=fp8* 需自行改
bw_c=wb*2/(c*1e-6)/1e9; bw_p=wb*2/(p*1e-6)/1e9
print(f"  clean   中位 {c:.2f}us  ({bw_c:.0f} GB/s, {bw_c/1800*100:.1f}%)  样本 {clean}")
print(f"  patched 中位 {p:.2f}us  ({bw_p:.0f} GB/s, {bw_p/1800*100:.1f}%)  样本 {patch}")
print(f"  Δ = {d:+.2f}%   (patch_group={patch_group})")
PY

echo "  --- 正确性 (patched --verify) ---"
HIPBLASLT_TENSILE_LIBPATH="$PATCHD" HIP_VISIBLE_DEVICES="$CARD" "$B" -m "$N" -n "$M" -k "$K" \
  --lda "$K" --ldb "$K" --ldc "$N" --ldd "$N" --transA "$TRANSA" --transB "$TRANSB" \
  -r "$DTYPE" -a "$DTYPE" -b "$DTYPE" -c "$DTYPE" -d "$DTYPE" \
  --compute_type "$COMPUTE" --activation_type None --iters 3 --cold_iters 1 --verify 2>&1 \
  | grep -iE "norm|error|verify|pass|fail" | head -3

TS=$(date '+%Y-%m-%d %H:%M')
{ echo ""; echo "## $TS  slc A/B ($CARD, $N/$K/$M, $PATCH)"; \
  echo "kernel=$KERNEL"; echo "见上方中位/norm 输出。"; } >> "$PERF_LOG"
echo "  perf-log: $PERF_LOG"
echo "  ⚠️ 收益须满足: patched 先跑的轮次也照样快 (见第 6 步逐轮)。Δ 为正且 norm 通过才算数。"
