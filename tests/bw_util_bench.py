#!/usr/bin/env python3
"""
bw_util_bench.py — GEMM 带宽利用率速测 (DCU gfx938, bf16, torch.matmul)

在【空闲】机器上跑: 检查目标卡 HCU 占用, 不闲则拒绝启动 (防止污染计时)。
L2 flush 冷缓存计时 (production-like), 输出 µs / 有效带宽 / 对 1.8TB/s 的利用率。

用法:
  HIP_VISIBLE_DEVICES=<卡> python3 bw_util_bench.py                 # 全部 shape, M=16/32/64/128
  HIP_VISIBLE_DEVICES=4 python3 bw_util_bench.py --shapes gate_up down
  HIP_VISIBLE_DEVICES=4 python3 bw_util_bench.py --ms 32,64 --iters 200
  HIP_VISIBLE_DEVICES=4 python3 bw_util_bench.py --kernel          # 顺带报 kernel 名+tile (需 hipblaslt-bench)

注意:
  - 本脚本只测, 不锁频 (环境禁止), 不加 lossy 操作。
  - 利用率口径 = 权重点数 / 时间, peak = 1.8 TB/s。
  - kernel 名来自 hipBLASLt heuristic 的 top-1 (与 torch 生产选择一致);
    若 hipblaslt-bench 无执行权限, 脚本会提示 chmod +x。
"""
import argparse, os, subprocess, sys, time
import torch

PEAK_TBPS = 1.8
FLUSH_BYTES = 64 * 1024 * 1024  # > L2 8MB, 冷缓存

# qwen3-32b shapes: W[N,K] bf16, D = a[M,K] @ W.t() [M,N]
SHAPES = {
    "gate_up": (51200, 5120),   # 2*intermediate x hidden, 权重 524MB
    "down":    (5120, 25600),   # hidden x intermediate,        256MB
    "qkv":     (10240, 5120),   # 102MB
    "o":       (5120, 8192),    # 84MB
}

def card_busy_pct():
    """HIP_VISIBLE_DEVICES 指定卡的 HCU 占用 (%); 未设则查全部可见卡; 取不到返回 None。"""
    vis = os.environ.get("HIP_VISIBLE_DEVICES", "").strip()
    want = set(int(x) for x in vis.split(",") if x.strip().isdigit()) if vis else None
    try:
        out = subprocess.run(["hy-smi"], capture_output=True, text=True, timeout=15).stdout
        lines = [l for l in out.splitlines() if l.strip() and l[0].isdigit()]
        pcts = []
        for l in lines:
            f = l.split()
            idx = int(f[0])
            if want is not None and idx not in want:
                continue
            pcts.append(float(f[6].rstrip("%")))  # HCU% 列
        if not pcts: return None
        return int(max(pcts))
    except Exception:
        return None

def bench_shape(N, K, M, iters):
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.float32, device="cuda")
    for _ in range(10):
        c = a @ w.t()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        flush.zero_()
        c = a @ w.t()
    e.record()
    torch.cuda.synchronize()
    us = s.elapsed_time(e) / iters * 1e3
    gb = N * K * 2 / (us * 1e-6) / 1e9
    return us, gb

BENCH = "/opt/dtk/lib/hipblaslt/benchmark_tool/hipblaslt-bench"

def kernel_name(N, K, M):
    """hipblaslt-bench heuristic top-1 的 kernel 名; 拿不到返回 None。"""
    if not os.path.exists(BENCH):
        return "(hipblaslt-bench 不存在: %s)" % BENCH
    if not os.access(BENCH, os.X_OK):
        return "(hipblaslt-bench 无执行权限, 先: chmod +x %s)" % BENCH
    try:
        out = subprocess.run(
            [BENCH, "-m", str(N), "-n", str(M), "-k", str(K),
             "--lda", str(K), "--ldb", str(K), "--ldc", str(N), "--ldd", str(N),
             "--transA", "T", "--transB", "N",
             "-r", "bf16_r", "--a_type", "bf16_r", "--b_type", "bf16_r",
             "--c_type", "bf16_r", "--d_type", "bf16_r", "--compute_type", "f32_r",
             "--activation_type", "None", "--print_kernel_info",
             "--iters", "50", "--cold_iters", "10"],
            capture_output=True, text=True, timeout=180).stdout
        for ln in out.splitlines():
            if "kernel name" in ln:
                return ln.split("kernel name", 1)[1].strip().lstrip(":").strip()
        return "(未解析到 kernel name)"
    except Exception as ex:
        return "(kernel 查询失败: %s)" % ex

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=list(SHAPES), choices=list(SHAPES))
    ap.add_argument("--ms", default="16,32,64,128")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--max-busy", type=int, default=20, help="卡 HCU%% 超过此值拒跑")
    ap.add_argument("--kernel", action="store_true", help="顺带查询 kernel 名/tile")
    a = ap.parse_args()
    Ms = [int(x) for x in a.ms.split(",")]

    busy = card_busy_pct()
    if busy is None:
        print("!! 读不到 hy-smi, 无法确认卡空闲 — 请人工确认后再跑 (加 --max-busy 100 跳过检查)")
    elif busy > a.max_busy:
        print("!! 卡 HCU 占用 %d%% > %d%%, 不闲, 拒绝启动 (计时会被污染)" % (busy, a.max_busy)); sys.exit(1)
    print("卡 HCU 占用: %s%%  (peak=%.1f TB/s, L2 flush 冷缓存, iters=%d)" %
          (busy if busy is not None else "?", PEAK_TBPS, a.iters))

    for sh in a.shapes:
        N, K = SHAPES[sh]
        print("\n=== %s  N=%d K=%d  权重 %.0f MB ===" % (sh, N, K, N*K*2/1e6))
        print("  M     延迟(us)   有效带宽(GB/s)   利用率   kernel")
        for M in Ms:
            us, gb = bench_shape(N, K, M, a.iters)
            kn = kernel_name(N, K, M) if a.kernel else ""
            print("  %-5d %10.1f  %14.1f  %7.1f%%   %s" % (M, us, gb, gb/1000/PEAK_TBPS*100, kn or "-"))

if __name__ == "__main__":
    main()
