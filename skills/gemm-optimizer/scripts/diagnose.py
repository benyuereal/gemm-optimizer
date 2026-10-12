#!/usr/bin/env python3
"""
diagnose.py — 诊断 patch 是否真在位（"patch 了但不提速" 第一排查项）

用法:
  diagnose.py <clean.co> <patched.co> <kernel_name> [READELF]

对目标 kernel 区做:
  1. 反汇编两版, 数 buffer_load 中 slc 置位条数、vmcnt(4) 条数
  2. 逐指令 diff (8B 对齐), 分类 slc+/slc-/vmcnt/other
  3. 报告 patched 相对 clean 到底改了什么

若 patched 区 slc==0 但源码 .s 有 slc → 拼接错位或 kernel 名选错。
"""
import struct, subprocess, sys, tempfile, os

SLC = 0x20000          # word0 bit17
BL_FAM = 0xE0000000    # gfx9 buffer_load SCODE 高字节
VMC4 = 0x0F74          # vmcnt(4) 低 word (node3 clang18)

def funcs(co):
    fd, tmp = tempfile.mkstemp(suffix=".co"); os.write(fd, co); os.close(fd)
    try:
        out = subprocess.run(["/opt/dtk/llvm/bin/llvm-readelf", "-s", tmp],
                             capture_output=True, text=True).stdout
    finally:
        os.unlink(tmp)
    m = {}
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 8 and p[3] == "FUNC":
            try: m.setdefault(int(p[1], 16), p[-1].rstrip("$"))
            except: pass
    return sorted(m.items())

def region(co, name):
    fs = funcs(co)
    for i, (va, n) in enumerate(fs):
        if n == name:
            end = fs[i+1][0] if i+1 < len(fs) else va + 0x200000
            return va, end
    return None, None

def census(buf):
    slc = vmc = pl = 0
    for i in range(len(buf)//8):
        hi = struct.unpack_from("<I", buf, i*8+4)[0]
        if (hi & 0xFE000000) == BL_FAM:
            pl += 1
            if hi & SLC: slc += 1
        if hi == VMC4: vmc += 1
    return pl, slc, vmc

def main():
    if len(sys.argv) < 4:
        print(__doc__); sys.exit(2)
    cpath, ppath, name = sys.argv[1], sys.argv[2], sys.argv[3]
    clean = open(cpath, "rb").read()
    patch = open(ppath, "rb").read()
    cva, cend = region(clean, name)
    pva, pend = region(patch, name)
    if cva is None or pva is None:
        print("!! kernel 未找到:", name); sys.exit(1)
    print(f"kernel: {name}")
    print(f"  clean   VA=0x{cva:x}  region={cend-cva}B")
    print(f"  patched VA=0x{pva:x}  region={pend-pva}B")

    cb = clean[cva-0x1000:cend-0x1000]
    pb = patch[pva-0x1000:pend-0x1000]
    n = min(len(cb), len(pb))
    pl_c, slc_c, vmc_c = census(cb[:n])
    pl_p, slc_p, vmc_p = census(pb[:n])
    print(f"\n  {'':12} {'buffer_load':>12} {'slc置位':>10} {'vmcnt(4)':>10}")
    print(f"  {'clean':12} {pl_c:>12} {slc_c:>10} {vmc_c:>10}")
    print(f"  {'patched':12} {pl_p:>12} {slc_p:>10} {vmc_p:>10}")
    print(f"  {'Δ':12} {pl_p-pl_c:>+12} {slc_p-slc_c:>+10} {vmc_p-vmc_c:>+10}")

    # 逐指令 diff
    slc_on = slc_off = vmc = other = 0
    for i in range(n//8):
        o = i*8
        a0,a1 = struct.unpack_from("<II", cb, o)
        b0,b1 = struct.unpack_from("<II", pb, o)
        if (a0,a1)==(b0,b1): continue
        if (b1 & SLC) and not (a1 & SLC): slc_on += 1
        elif (a1 & SLC) and not (b1 & SLC): slc_off += 1
        elif a1 != b1 and (a1 & ~0xF) == (b1 & ~0xF) and (a1 ^ b1) & 0xF in (0,4): vmc += 1
        else: other += 1
    print(f"\n  逐指令 diff (patched vs clean): slc+ {slc_on}, slc- {slc_off}, vmcnt {vmc}, other {other}")
    if slc_p == 0 and other == 0 and slc_on == 0:
        print("\n  ⚠️  patched 区无任何 slc/vmcnt 变化 —— 若源码 .s 本应有 slc, 说明拼接错位或 kernel 名选错。")

if __name__ == "__main__":
    main()
