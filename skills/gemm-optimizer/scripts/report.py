#!/usr/bin/env python3
"""
report.py — 从 perf-log.md 生成 report.md 汇总

用法: report.py [perf-log.md] [report.md]

汇总: 每轮 kernel × 手段的收益, 标注有/无收益, 列环境。
"""
import re, sys, os, datetime

def main():
    perf = sys.argv[1] if len(sys.argv) > 1 else "perf-log.md"
    out = sys.argv[2] if len(sys.argv) > 2 else "report.md"
    if not os.path.exists(perf):
        print("!! 找不到", perf, "(先跑 patch_co.sh 生成 perf-log)"); sys.exit(1)
    lines = open(perf).read().splitlines()

    # 抓所有 run 块 (## 时间戳) 及其表格行
    runs = []
    cur = None
    for ln in lines:
        if ln.startswith("## "):
            cur = {"ts": ln[3:].strip(), "rows": []}
            runs.append(cur)
        elif cur is not None and ln.startswith("| ") and "µs" not in ln and "kernel" not in ln and "reorder" not in ln:
            cells = [c.strip() for c in ln.strip("|").split("|")]
            if len(cells) >= 6 and cells[4] != "Δ%":
                cur["rows"].append(cells)

    with open(out, "w") as f:
        f.write("# GEMM 优化报告 (DCU gfx938 / hipBLASLt .co)\n\n")
        f.write(f"生成时间: {datetime.datetime.now():%Y-%m-%d %H:%M}\n\n")
        total_win = 0; total_n = 0
        for r in runs:
            f.write(f"## {r['ts']}\n\n")
            f.write("| kernel | reorder | 基线µs | patchedµs | Δ% | 正确性 | 结论 |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for c in r["rows"]:
                # c = [kernel, reorder, base, new, pct, corr]
                try:
                    base, new = float(c[2]), float(c[3])
                    win = new < base
                except:
                    win = False
                total_n += 1; total_win += int(win)
                concl = "✅ 收益" if win else "❌ 无收益(回滚)"
                f.write(f"| {c[0]} | {c[1]} | {c[2]} | {c[3]} | {c[4]} | {c[5]} | {concl} |\n")
            f.write("\n")
        f.write(f"## 汇总\n\n{total_n} 个 kernel 测次, 其中 {total_win} 个有正收益。\n")
        f.write("\n**下一步** (若仍有 ❌):\n")
        f.write("- 跑 `diagnose.py clean.co patched.co <kernel>` 确认 patch 是否在位\n")
        f.write("- 核对基线是否同机同口径 (SKILL.md §5 诊断)\n")
        f.write("- 该机器是否响应 cache hint (slc 12 vs 201 条计时是否相同)\n")
    print(f"报告已生成: {out}")

if __name__ == "__main__":
    main()
