#!/usr/bin/env python3
"""
ceiling.py — 快速定位某 shape 的带宽利用率天花板（DCU gfx938）

对给定 GEMM shape (M, N, K)，算三层天花板，帮你判断"测到的利用率算好还是还有空间":

  1. 屋顶线硬顶 (roofline)  —— 理论极限, 恒正确
     ridge = FP8_PEAK / HBM_PEAK ≈ 311 FLOP/B → M_ridge ≈ 155
     M < 155 → memory-bound, FLOPS 天花板 = 2·M·1.8TB/s
  2. 流量天花板 (dilution)  —— 若完美吃满总带宽, 权重带宽利用率最多到 weight/total
     (activation+output 流量随 M 涨, 稀释权重利用率; 解释了 M 越大利用率越低)
  3. 实际天花板 (practical) —— 经验值表 (含 qkv/down 这类"形状天然低、正常"的类)
     gate_up: M≤16→80%, M≤64→70%;  其余 shape 需本机实测补

用法:
  ceiling.py --M 64 --N 51200 --K 5120 [--name gate_up]
  ceiling.py --M 64 --N 51200 --K 5120 --name gate_up --measured 56   # 带实测给判定
  ceiling.py --add qkv 64:50            # 把实测天花板记进经验表 (M:pct)
  ceiling.py --list                     # 看经验表

经验表落在同目录 ceiling-table.json, 用 --add 累积本机实测值。
"""
import argparse, json, math, os, sys

# ---- 硬件常量 (gfx938 / CDNA3) ----
HBM_PEAK = 1.8e12      # 1.8 TB/s
FP8_PEAK = 560e12      # 560 TFLOPS
N_CU = 64
L2_BYTES = 8 * 1024 * 1024
RIDGE = FP8_PEAK / HBM_PEAK          # ≈311 FLOP/B
M_RIDGE = RIDGE / 2                   # ≈155

DEFAULT_TILEN = 64   # 输出 tile 宽度, 用于估 tile 数 (诊断用, 随 selection 变)

TABLE_PATH = os.path.join(os.path.dirname(__file__), "ceiling-table.json")

# 种子经验表 (本机实测确认值)。gate_up 是用户确认的; 其余为"形状天然低类"占位, 需本机实测补。
SEED = {
    "gate_up": {"M16": 80, "M64": 70, "note": "权重262MB≪L2+tile多, 最理想可patch对象"},
    "down":    {"note": "N=5120 tile少kernel短, 利用率天然低, 45~50% 属正常, 待本机确认上限"},
    "qkv":     {"note": "N=10240 tile少, 利用率天然低, 45~50% 属正常, 待本机确认上限"},
    "o":       {"note": "K=8192 权重42MB, 待本机确认上限"},
}


def load_table():
    if os.path.exists(TABLE_PATH):
        return json.load(open(TABLE_PATH))
    return {k: dict(v) for k, v in SEED.items()}


def save_table(t):
    json.dump(t, open(TABLE_PATH, "w"), ensure_ascii=False, indent=2)


def practical_ceiling(name, M, table):
    """从经验表估实际带宽利用率天花板。返回 (pct 或 None, 说明)。"""
    row = table.get(name)
    if not row:
        return None, "无经验值; 先用 gate_up 曲线粗估(见下), 再本机实测 --add 补"
    # 支持 M16/M64 分档 + 线性插值(按 log M)
    keys = {k: v for k, v in row.items() if k.startswith("M") and isinstance(v, (int, float))}
    if keys:
        pts = sorted((int(k[1:]), float(v)) for k, v in keys.items())
        if M <= pts[0][0]:
            return pts[0][1], f"{name} M≤{pts[0][0]} 档"
        if M >= pts[-1][0]:
            # 外推: 每翻倍 -5pt (gate_up 16→64 的经验斜率)
            hi_m, hi_v = pts[-1]
            return max(20, hi_v - 5 * math.log2(M / hi_m)), f"{name} M≥{hi_m} 外推(每翻倍-5pt)"
        # 插值
        for (m0, v0), (m1, v1) in zip(pts, pts[1:]):
            if m0 <= M <= m1:
                f = (math.log2(M) - math.log2(m0)) / (math.log2(m1) - math.log2(m0))
                return v0 + (v1 - v0) * f, f"{name} M={M} 插值"
    return None, f"{name} 只有备注: {row.get('note')}"


def ramp_limited(weight_b, n_tiles, T_fixed_us=55):
    """kernel 短 / tile 少时, 固定开销(ramp-up + tile 尾部不并行)摊不薄,
    平均带宽利用率被压。返回 (估算实际顶%, 是否受ramp限制)。
    估算: T_weight = 权重/峰值; 实际顶 ≈ T_weight / (T_weight + T_fixed) × 100
    (T_fixed 经验值 ~55µs, 用 gate_up 反推量级; 只是保守下界, 非精确)"""
    t_weight = weight_b / HBM_PEAK * 1e6  # µs, 权重纯传输
    est = t_weight / (t_weight + T_fixed_us) * 100
    limited = n_tiles < 256 or t_weight < 60
    return est, limited


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int)
    ap.add_argument("--N", type=int)
    ap.add_argument("--K", type=int)
    ap.add_argument("--name", default="?")
    ap.add_argument("--measured", type=float, help="实测带宽利用率%%, 给了就给判定")
    ap.add_argument("--add", metavar="NAME M:PCT", help="记录实测天花板")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    table = load_table()

    if a.list:
        print("经验天花板表 (ceiling-table.json):")
        for k, v in table.items():
            print(f"  {k}: {v}")
        return
    if a.add:
        name, mv = a.add.split(None, 1)
        m, pct = mv.split(":")
        row = table.setdefault(name, {})
        row[f"M{int(m)}"] = float(pct)
        save_table(table)
        print(f"已记录 {name} M={m} 天花板 {pct}%; 表已存 {TABLE_PATH}")
        return
    if not (a.M and a.N and a.K):
        ap.print_help(); return

    M, N, K = a.M, a.N, a.K
    weight_b = N * K * 1            # fp8 权重
    act_b = M * K * 1               # fp8 激活
    out_b = M * N * 2               # bf16 输出
    total_b = weight_b + act_b + out_b
    flops = 2 * M * N * K

    print(f"== 天花板分析: {a.name}  M={M} N={N} K={K} (gfx938) ==")
    print(f"  权重 {weight_b/1e6:.0f}MB | 激活 {act_b/1e3:.0f}KB | 输出 {out_b/1e6:.1f}MB | 总流量 {total_b/1e6:.1f}MB")
    print(f"  L2={L2_BYTES/1e6:.0f}MB → 权重{'≫L2(slc可考虑)' if weight_b>L2_BYTES else '<L2(slc通常无效)'}")
    n_tiles = math.ceil(M / 128) * math.ceil(N / DEFAULT_TILEN)
    print(f"  输出 tile 数(估) ≈ {n_tiles}  (64 CU, {'CU 填得满' if n_tiles >= N_CU else '⚠️ tile<64 CU填不满→利用率受限'})")

    # 1. 屋顶线
    bound = "memory-bound" if M < M_RIDGE else "compute-bound"
    if bound == "memory-bound":
        max_flops = 2 * M * HBM_PEAK
        print(f"  [1 屋顶线] {bound} (M={M} < M_ridge≈{M_RIDGE:.0f}) → FLOPS 天花板 ≈ {max_flops/1e12:.0f} TFLOPS")
    else:
        max_flops = FP8_PEAK
        print(f"  [1 屋顶线] {bound} (M={M} ≥ M_ridge≈{M_RIDGE:.0f}) → FLOPS 天花板 = 560 TFLOPS")

    # 2. 流量天花板 (权重带宽利用率理论上限)
    dilution = weight_b / total_b * 100
    print(f"  [2 流量顶] 若完美吃满总带宽, 权重带宽利用率最多 ≈ {dilution:.1f}% (被激活/输出稀释 {100-dilution:.1f}pt)")

    # 3. 实际天花板: 取 (经验值, 流量顶, ramp限制) 里最严的
    exp_pct, why = practical_ceiling(a.name, M, table)
    ramp_pct, ramp_lim = ramp_limited(weight_b, n_tiles)
    cands = []
    if exp_pct is not None: cands.append((exp_pct, f"经验值({why})"))
    cands.append((dilution, "流量顶"))
    if ramp_lim: cands.append((ramp_pct, f"ramp限制(tile={n_tiles},kernel短)"))
    pct = min(cands)
    print(f"  [3 实际顶] ≈ {pct[0]:.0f}%  ← {pct[1]}")
    for p, w in cands:
        print(f"        · {w}: {p:.0f}%")
    if ramp_lim and exp_pct is None:
        print(f"  ⚠️  该 shape 无本机实测经验值, 实际顶受 kernel短/tile少 压制, 低利用率属正常,\n"
              f"      不代表有大空间可挖。想确认上限: 本机实测后 `--add {a.name} {M}:%` 记录。")

    # 判定
    if a.measured is not None:
        headroom = pct[0] - a.measured
        if exp_pct is None and ramp_lim:
            # 无本机实测经验值 + 受形状压制: 不给自信结论, 提示先实测
            verdict = ("利用率受形状压制(kernel短/tile少), 低属正常; "
                       "估算顶仅供参考, 先本机实测确认上限再决定要不要 patch")
        elif headroom <= 5:
            verdict = "已贴近天花板, 该shape基本到头, 转攻别的kernel/selection"
        else:
            verdict = f"离天花板还有 ~{headroom:.0f}pt, 值得patch"
        print(f"\n  判定: 实测 {a.measured:.0f}% vs 实际顶 {pct[0]:.0f}% → {verdict}")


if __name__ == "__main__":
    main()
