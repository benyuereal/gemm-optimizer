#!/bin/bash
# =============================================================================
# patch_co.sh — hipBLASLt .co GEMM kernel patch 安装 + 验证 + 自动回滚
#
# 流程(7 步, 任一步失败自动回滚到干净 .co 并校验 md5):
#   1. 备份生产 .co
#   2. 汇编 src/*.s -> .o (DTK clang, 按名找 kernel)
#   3. 按 kernel 名字解析 VA, 把各 .o 的 .text 拼进同一个 .co
#   4. 字节校验: 每个拼进的区域 == 汇编出的 .o
#   5. 正确性: perm_check orig+ref vs patched+new (<=1 ULP)
#   6. 计时: layout_bench orig+ref (基线) vs patched+new (目标), 同场交替
#   7. 报告: 写 perf-log.md; 有收益保留 patched, 无收益回滚
#
# 用法:
#   bash patch_co.sh <patched_src_dir> [GPU_ID]
#     patched_src_dir 下需有 kernels.txt (表) 和 src/*.s
#
# kernels.txt 表格式 (每行, | 分隔):
#   kernel_name|N|K|M|src_path|target_us|reorder(1=LayoutA需重排, 0=slc不重排)
#
# Prereqs (DTK + 项目已有 bench 二进制):
#   layout_bench / perm_check 二进制路径可用环境变量覆盖
# =============================================================================
set -u

SRC_DIR="${1:?用法: patch_co.sh <patched_src_dir> [GPU_ID]}"
GPU_ID="${2:-5}"
HERE="$(cd "$(dirname "$0")" && pwd)"
CO="${CO:-/opt/dtk/lib/hipblaslt/library_bw_nmz/FusedMatmulKernel_gfx938.co}"
AS="${AS:-/opt/dtk/aillvm/bin/clang}"
OBJCOPY="${OBJCOPY:-/opt/dtk/aillvm/bin/llvm-objcopy}"
READELF="${READELF:-/opt/dtk/llvm/bin/llvm-readelf}"
LAYOUT_BENCH="${LAYOUT_BENCH:-$HERE/../../../../../_bench/gemm/hipblaslt_layout_bench}"
PERM_CHECK="${PERM_CHECK:-$HERE/../../../../../_bench/gemm/hipblaslt_perm_check}"
WORK="$(mktemp -d /tmp/gemmopt.XXXXXX)"
PERF_LOG="${PERF_LOG:-$(pwd)/perf-log.md}"

[ -f "$SRC_DIR/kernels.txt" ] || { echo "!!! 缺 $SRC_DIR/kernels.txt"; exit 1; }
echo "=== [1/7] backup $CO ==="
[ -f "$CO" ] || { echo "!!! 生产 .co 不存在: $CO"; exit 1; }
cp "$CO" "$WORK/backup.co"
CLEAN_MD5=$(md5sum "$WORK/backup.co" | cut -d' ' -f1)
echo "    基线 md5: $CLEAN_MD5"

fail() {
  echo "!!! $*"; cp "$WORK/backup.co" "$CO"
  local m; m=$(md5sum "$CO" | cut -d' ' -f1)
  [ "$m" = "$CLEAN_MD5" ] && echo "    (已回滚 $CO, md5 校验 OK)" || echo "    (回滚后 md5 不一致: $m)"
  exit 1
}

# 按 kernel 名解析 VA + region size
kernel_va_size() {
  "$READELF" -s "$CO" 2>/dev/null | python3 -c "
import sys
seen={}
for line in sys.stdin:
    p=line.split()
    if len(p)>=8 and p[3]=='FUNC':
        try: seen.setdefault(int(p[1],16), p[-1].rstrip('\$'))
        except: pass
syms=sorted(seen.items())
for i,(va,n) in enumerate(syms):
    if n=='$1':
        end=syms[i+1][0] if i+1<len(syms) else va+0x200000
        print(hex(va), hex(end-va)); break
"
}

echo "=== [2/7] 汇编 src/*.s ==="
cp "$WORK/backup.co" "$WORK/patched.co"
while IFS='|' read -r K N KLEN M SRC TGT REORD; do
  [ -z "${K:-}" ] && continue
  tag=$(echo "$SRC" | tr '/' '_')
  "$AS" -x assembler --target=amdgcn-amd-amdhsa -mcode-object-version=4 \
    -mcpu=gfx938 -mwavefrontsize64 -c "$SRC_DIR/$SRC" -o "$WORK/$tag.o" || fail "汇编失败: $SRC"
  "$OBJCOPY" -O binary --only-section=.text "$WORK/$tag.o" "$WORK/$tag.text" 2>/dev/null
  echo "    $tag .text: $(stat -c%s "$WORK/$tag.text") bytes"
done < "$SRC_DIR/kernels.txt"

echo "=== [3/7] 按名拼进 .co ==="
while IFS='|' read -r K N KLEN M SRC TGT REORD; do
  [ -z "${K:-}" ] && continue
  tag=$(echo "$SRC" | tr '/' '_')
  read VA SIZE < <(kernel_va_size "$K")
  [ -n "${VA:-}" ] || fail "kernel 未找到: $K"
  python3 - "$WORK/patched.co" "$WORK/$tag.text" "$VA" "$SIZE" <<'PY'
import sys
co, text, va, size = sys.argv[1], open(sys.argv[2],'rb').read(), int(sys.argv[3],0), int(sys.argv[4],0)
assert len(text) <= size, f".o {len(text)} > kernel region {size} (kernel 名选错?)"
b = bytearray(open(co,'rb').read()); off = va - 0x1000
b[off:off+len(text)] = text
open(co,'wb').write(bytes(b))
print(f"    spliced {len(text)}B @ VA {hex(va)} (region {size})")
PY
done < "$SRC_DIR/kernels.txt"

echo "=== [4/7] 字节校验 (区域 == 汇编 .o) ==="
while IFS='|' read -r K N KLEN M SRC TGT REORD; do
  [ -z "${K:-}" ] && continue
  tag=$(echo "$SRC" | tr '/' '_')
  read VA SIZE < <(kernel_va_size "$K")
  osize=$(stat -c%s "$WORK/$tag.text")
  dd if="$WORK/patched.co" bs=1 skip=$((VA-0x1000)) count=$((osize)) 2>/dev/null > "$WORK/$tag.verify"
  cmp -s "$WORK/$tag.text" "$WORK/$tag.verify" || fail "$tag: 拼进区域 != 汇编 .o (diff $(cmp -l "$WORK/$tag.text" "$WORK/$tag.verify" | wc -l)B)"
  echo "    $tag: BYTE-EXACT OK"
done < "$SRC_DIR/kernels.txt"

echo "=== [5/7] 安装 + 正确性 (perm_check <=1 ULP) ==="
cp "$WORK/patched.co" "$CO"
ALL_OK=1
while IFS='|' read -r K N KLEN M SRC TGT REORD; do
  [ -z "${K:-}" ] && continue
  tag=$(echo "$SRC" | tr '/' '_')
  cp "$WORK/backup.co" "$CO"
  HIP_VISIBLE_DEVICES=$GPU_ID "$PERM_CHECK" $N $KLEN $M "$K" ref "$WORK/${tag}_ref.bin" 50 >/dev/null 2>&1 || { echo "    $tag: ref 失败"; ALL_OK=0; }
  cp "$WORK/patched.co" "$CO"
  HIP_VISIBLE_DEVICES=$GPU_ID "$PERM_CHECK" $N $KLEN $M "$K" new "$WORK/${tag}_new.bin" 50 >/dev/null 2>&1 || { echo "    $tag: new 失败"; ALL_OK=0; }
  cp "$WORK/backup.co" "$CO"
  if [ "$ALL_OK" = 1 ]; then
    python3 - "$WORK/${tag}_ref.bin" "$WORK/${tag}_new.bin" <<PY || { echo "    $tag: 正确性 >1 ULP"; ALL_OK=0; }
import struct, sys
a=open(sys.argv[1],'rb').read(); b=open(sys.argv[2],'rb').read()
n=min(len(a),len(b))//2
A=struct.unpack("<%dh"%n, a[:n*2]); B=struct.unpack("<%dh"%n, b[:n*2])
worst=max((abs((A[i]&0xFFFF)-(B[i]&0xFFFF)) for i in range(n)), default=999)
print(f"    $tag: worst {worst} ULP")
sys.exit(0 if worst<=1 else 1)
PY
  fi
done < "$SRC_DIR/kernels.txt"
cp "$WORK/patched.co" "$CO"
[ "$ALL_OK" = 1 ] || fail "正确性未全过"

echo "=== [6/7] 计时 (同场交替, 写 perf-log) ==="
t() { # N K M K mode
  HIP_VISIBLE_DEVICES=$GPU_ID "$LAYOUT_BENCH" $1 $2 $3 "$4" "$5" 3 100 2>&1 \
    | grep -E 'round' | tail -1 | grep -oE '[0-9.]+ us' | grep -oE '[0-9.]+'
}
TS=$(date '+%Y-%m-%d %H:%M')
{ echo ""; echo "## $TS  patch run (card $GPU_ID, base $CLEAN_MD5)"; \
  echo "| kernel | reorder | 基线µs | patchedµs | Δ% | 正确性 |"; echo "|---|---|---|---|---|---|"; } >> "$PERF_LOG"
ANY_WIN=0
while IFS='|' read -r K N KLEN M SRC TGT REORD; do
  [ -z "${K:-}" ] && continue
  tag=$(echo "$SRC" | tr '/' '_')
  BASE=$(t $N $KLEN $M $K ref)
  NEW=$(t $N $KLEN $M $K new)
  cp "$WORK/patched.co" "$CO"
  PCT=$(python3 -c "print(f'{100*(1-$NEW/$BASE):+.1f}')" 2>/dev/null || echo NA)
  echo "| $tag | $REORD | $BASE | $NEW | $PCT% | <=1ULP |" >> "$PERF_LOG"
  python3 -c "exit(0 if $NEW < $BASE else 1)" && ANY_WIN=1
  echo "    $tag: $BASE -> $NEW ($PCT%)"
done < "$SRC_DIR/kernels.txt"

echo "=== [7/7] 结论 ==="
if [ "$ANY_WIN" = 1 ]; then
  echo "  有收益, patched .co 保留在位 (md5 $(md5sum $CO | cut -d' ' -f1))"
  echo "  ⚠️ Layout A kernel 必须配 reorder_weights.py 权重重排后再重启 serving, 否则算错。"
else
  cp "$WORK/backup.co" "$CO"
  echo "  无收益, 已回滚到干净 .co (md5 $(md5sum $CO | cut -d' ' -f1))"
fi
echo "  perf-log: $PERF_LOG"
exit 0
