# 04 报告 phase: 记 perf-log、出报告、判定达标

## perf-log.md 格式（与 patch_co.sh 写入一致，手工记录也用这个）

每个 run 一个块，追加到 `perf-log.md`：

```
## <时间戳>  patch run (card <GPU>, base <clean .co md5前缀>)
| kernel | reorder | 基线µs | patchedµs | Δ% | 正确性 |
|---|---|---|---|---|---|
| <kernel短名> | <0/1> | <base> | <new> | <+x.x%> | <=1ULP |
```

- `reorder=1` = Layout A（需权重重排），`0` = slc（不重排）。
- 短名须能还原全名（tag 或 `kernels.txt` 可查）。

## 每条记录必带信息（短清单）

- kernel 全名（或可还原的全名）
- `.co` md5：patched 和 clean 各一个
- 卡号 `HIP_VISIBLE_DEVICES`、DTK 版本
- bench 参数（rounds/iters，mode ref/new）
- 有效带宽（GB/s）和利用率（%），**不只是 µs**
- 正确性 worst 值（≤1 ULP 的实际最大偏差）

## 达标判定规则

- **达标**：全部相关 kernel 有正收益 **且** 正确性 ≤1 ULP → 保留 patched .co。
- **收敛**：连续 3 轮无提升 → 停手，报告当前最优。
- **❌ 无收益**：记录原因（按 03-patching 的 5 步诊断走一遍并引用结果），回滚。
- **禁止只报变好的 kernel**（铁律 6）：达标判断必须看所有相关 kernel 的表。

## report.py

```
python3 report.py [perf-log.md] [report.md]
```

解析 perf-log 的 run 块表格，生成：每轮 kernel×手段收益表 + ✅收益/❌无收益(回滚) 标注
+ 汇总（N 测次 / M 正收益）+ 下一步建议（diagnose.py 验 patch 在位、核基线口径、
验机器是否响应 cache hint）。报告以 report.py 输出为骨架，人工补齐上面"必带信息"清单。

## 收尾（写进报告）

- Layout A kernel：注明必须配 `reorder_weights.py` 做权重重排后再重启 serving，**缺重排 = 算错**。
- 确认 `.co` 处于期望终态（patched 或干净），记录最终 md5。
- 全程不碰生产服务/生产卡（铁律 7）。

## 反面案例（同口径铁律，本项目踩过）

曾把 **生产口径**基线（234.8µs，真实权重/真实 serving）对比 **隔离 bench 口径**的
patched（218.5µs，合成数据），得出 -6.9%；但同一隔离 bench 的未 patch 基线只有
222.1µs —— 两种测量方法（生产 vs 隔离 bench）本身就有差距，混着减会把方法差算进 patch 收益。
**要干净归因：基线和 patched 必须同一 bench、同一张卡、同场交替测**（铁律 1）。
要坐实生产口径的收益，就在生产路径里再跑一次 patched 的 A/B，而不是拿隔离 bench 的数。
