#!/usr/bin/env python3
"""
bench.py — 单 kernel 计时 + 有效带宽（DCU gfx938）

用法:
  bench.py <kernel_name> <N> <K> <M> <mode:ref|new> [GPU_ID] [rounds] [iters]

调 layout_bench 计时, 输出 µs + 有效带宽 + 相对 1.8TB/s 的利用率。
mode: ref=row-major(配原始 kernel) / new=blocked Layout A(配 patched kernel)
注意: slc 类不重排的 kernel 两种 mode 数据一样, 用 ref 即可。
"""
import subprocess, sys, os

def main():
    if len(sys.argv) < 6:
        print(__doc__); sys.exit(2)
    name, N, K, M, mode = sys.argv[1:6]
    gpu = sys.argv[6] if len(sys.argv) > 6 else "5"
    rounds = sys.argv[7] if len(sys.argv) > 7 else "3"
    iters = sys.argv[8] if len(sys.argv) > 8 else "100"
    bench = os.environ.get("LAYOUT_BENCH",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                     "..", "_bench", "gemm", "hipblaslt_layout_bench"))
    env = dict(os.environ, HIP_VISIBLE_DEVICES=gpu)
    out = subprocess.run([bench, str(N), str(K), str(M), name, mode,
                          str(rounds), str(iters)],
                         capture_output=True, text=True, env=env).stdout
    # 抓所有 round 行的 µs
    vals = []
    for line in out.splitlines():
        if "round" in line:
            for tok in line.split():
                if tok.endswith("us"):
                    try: vals.append(float(tok[:-2]))
                    except: pass
    if not vals:
        print("!! 解析不到 timing, 原始输出:\n" + out); sys.exit(1)
    vals.sort()
    med = vals[len(vals)//2]
    weight_bytes = int(N) * int(K)  # fp8 = 1B
    bw = weight_bytes / (med * 1e-6) / 1e9  # GB/s
    print(f"{name}")
    print(f"  N={N} K={K} M={M} mode={mode} gpu={gpu}  ({rounds}x{iters})")
    print(f"  min={vals[0]:.1f}us  med={med:.1f}us  max={vals[-1]:.1f}us  ({len(vals)} 轮)")
    print(f"  权重 {weight_bytes/1e6:.1f}MB  有效带宽 {bw:.0f} GB/s  利用率 {bw/1800*100:.1f}% (of 1.8TB/s)")
    return med

if __name__ == "__main__":
    main()
