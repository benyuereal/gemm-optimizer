# 07 · 优化原理：为什么 slc 有效（硬件 + 指令）

> 一句话结论：decode GEMM 是带宽 bound，HBM 流量 = 一次性流过的权重 + 反复复用的激活 + 写回输出。
> 默认缓存策略下，大权重流把 L2 填满、把小激活逐出 → 激活每次复用都 miss 回 HBM，凭空多出一轮
> "激活流量"。slc（evict-first，`buffer_load` word0 bit17）告诉硬件"这批一次性数据缓存但优先
> 逐出"，把位置留给复用数据，bit-for-bit 无损。实测 bf16 gate_up 524MB M=32：-6.2%
> （422.4→396.3µs，1241→1323 GB/s，以 01/02 篇为准）。
>
> 本文讲这条因果链的两层（硬件 → 指令）；
> 数字全部以 01/02/03 篇已记录内容为准，不自造数字。

## 1. 硬件层：缓存层级与"污染"问题

### 1.0 优化背景：M=32/64 带宽利用率不足（slc 的动机）

**我们的背景**：decode 场景 GEMM 是带宽 bound，但 **M=32 / M=64 的实测带宽利用率只有
~60%**，距离 **~80% 的理想值还有明显差距**——说明 HBM 流量里有可以省掉的冗余字节，
存在优化空间。这正是 slc（evict-first）针对的问题：大权重流冲刷 L2、把小激活逐出，
造成激活复用时多余的 HBM 流量（详见 1.3）。

**学界背景**：GEMM 的带宽低效是已知问题——attention / MoE 里的投影 GEMM 因**算术强度低**
（QKV 投影的算术强度比 down 投影低）而**带宽受限**，是 kernel 融合、GEMM 优化的主要对象。
下面这篇参考论文（arXiv:2510.27656）专门总结了 attention 与 MoE GEMM 的带宽低效：

> 内联 SVG（Markdown 渲染器支持即显示）。红色高亮 = QKV 投影因算术强度低而带宽受限的段落，
> 直接点出"投影 GEMM 的带宽低效"这一背景。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 720" font-family="Georgia,'Times New Roman',serif">
  <rect width="1000" height="720" fill="#ffffff"/>
  <text x="40" y="30" font-size="11" fill="#666" font-family="-apple-system,'PingFang SC',sans-serif">arXiv:2510.27656v2 [cs.CL]</text>
  <text x="960" y="30" text-anchor="end" font-size="11" fill="#666" font-family="-apple-system,'PingFang SC',sans-serif">3 Oct 2025</text>
  <line x1="40" y1="40" x2="960" y2="40" stroke="#ddd" stroke-width="1"/>
  <text x="40" y="66" font-size="12" fill="#1a1a1a">
    <tspan x="40" dy="0">with a single fused kernel. For example, the prefill Q and KV</tspan>
    <tspan x="40" dy="16">projections (Section 3.1), the post-attention linear layer (Sec-</tspan>
    <tspan x="40" dy="16">tion 3.1), and the QKV projection in MLA (Section 2.3) can be</tspan>
    <tspan x="40" dy="16">fused. These fused kernels are shown in Figure 11. We measure</tspan>
    <tspan x="40" dy="16">the kernel fusion&#8217;s effect on GPU utilization. For the MLA QKV</tspan>
    <tspan x="40" dy="16">projection, fusion leads to a 1.25&#215; reduction in GPU time. For</tspan>
    <tspan x="40" dy="16">the prefill QKV projection in GPT-OSS, fusion leads to a 1.3&#215;</tspan>
    <tspan x="40" dy="16">speedup. For the post-attention linear layer, fusion leads to a</tspan>
    <tspan x="40" dy="16">1.54&#215; speedup. Fusion can also enable a single kernel launch,</tspan>
    <tspan x="40" dy="16">reducing launch overhead (Chen et al., 2024; Shao et al., 2024;</tspan>
    <tspan x="40" dy="16">Yang et al., 2024b).</tspan>
  </text>
  <rect x="34" y="240" width="422" height="176" fill="#fdecea" stroke="#d33" stroke-width="1.6"/>
  <text x="48" y="262" font-size="13" font-weight="700" fill="#111">5.4.2 Attention Kernel Fusions</text>
  <text x="48" y="284" font-size="12" fill="#1a1a1a">
    <tspan x="48" dy="0">We further find that the performance of the QKV projection</tspan>
    <tspan x="48" dy="16">is significantly lower than the down projection in the GQA-</tspan>
    <tspan x="48" dy="16">based models. This is due to the lower arithmetic intensity of</tspan>
    <tspan x="48" dy="16">the QKV projection. We use Q4, K2, and V2 in the following</tspan>
    <tspan x="48" dy="16">text to refer to the Q projection with the GQA ratio of 2, the</tspan>
    <tspan x="48" dy="16">K projection, and the V projection, respectively.</tspan>
  </text>
  <text x="48" y="452" font-size="13" font-weight="700" fill="#111">5.5 Inefficiencies in MoE Operations</text>
  <rect x="44" y="466" width="3" height="16" fill="#d33"/>
  <text x="48" y="479" font-size="13" font-weight="700" fill="#111">5.5.1 Routing</text>
  <text x="48" y="502" font-size="12" fill="#1a1a1a">
    <tspan x="48" dy="0">We investigate the routing computation in MoE models and</tspan>
    <tspan x="48" dy="16">find several inefficiencies: the top-k expert selection is</tspan>
    <tspan x="48" dy="16">performed sequentially (limiting GPU utilization), the softmax</tspan>
    <tspan x="48" dy="16">is computed redundantly per expert, and routing is not</tspan>
    <tspan x="48" dy="16">overlapped with the following MoE GEMM. Our fused routing</tspan>
    <tspan x="48" dy="16">kernel computes the top-k experts in parallel and overlaps</tspan>
    <tspan x="48" dy="16">routing with the downstream GEMM.</tspan>
  </text>
  <rect x="44" y="620" width="3" height="16" fill="#d33"/>
  <text x="48" y="633" font-size="13" font-weight="700" fill="#111">5.5.2 GEMM</text>
  <text x="48" y="656" font-size="12" fill="#1a1a1a">
    <tspan x="48" dy="0">We develop a GEMM with per-token-block scaling. Across all</tspan>
    <tspan x="48" dy="16">MoE layers, our optimized MoE operations speed up 1.17&#215; to</tspan>
    <tspan x="48" dy="16">1.89&#215; over the reference implementations.</tspan>
  </text>
  <text x="520" y="66" font-size="12" fill="#1a1a1a">
    <tspan x="520" dy="0">... 512 experts in the second and third MoE layers. Across all</tspan>
    <tspan x="520" dy="16">models and all MoE layers, our fused routing leads to 1.05&#215;</tspan>
    <tspan x="520" dy="16">to 1.90&#215; speedup in the routing stage.</tspan>
  </text>
  <text x="520" y="150" font-size="13" font-weight="700" fill="#111">5.5 Inefficiencies in MoE Operations</text>
  <text x="520" y="172" font-size="12" fill="#1a1a1a">
    <tspan x="520" dy="0">We identify several inefficiencies in MoE operations, including</tspan>
    <tspan x="520" dy="16">the inefficiency in routing and MoE GEMM, and provide our</tspan>
    <tspan x="520" dy="16">optimizations.</tspan>
  </text>
  <rect x="516" y="220" width="3" height="16" fill="#d33"/>
  <text x="520" y="233" font-size="13" font-weight="700" fill="#111">5.5.1 Routing</text>
  <text x="520" y="256" font-size="12" fill="#1a1a1a">
    <tspan x="520" dy="0">The current routing computation contains several inefficien-</tspan>
    <tspan x="520" dy="16">cies. First, the computation of the top-k experts is not</tspan>
    <tspan x="520" dy="16">parallel: the top-k experts are selected sequentially, which</tspan>
    <tspan x="520" dy="16">limits GPU utilization. Our fused routing kernel computes the</tspan>
    <tspan x="520" dy="16">top-k experts in a parallel manner. Second, the softmax is</tspan>
    <tspan x="520" dy="16">computed redundantly: the softmax over the selected experts&#8217;</tspan>
    <tspan x="520" dy="16">scores can be computed once rather than per expert. Third, the</tspan>
    <tspan x="520" dy="16">routing computation and the following MoE GEMM are not</tspan>
    <tspan x="520" dy="16">overlapped; we overlap them to improve utilization.</tspan>
  </text>
  <rect x="516" y="420" width="3" height="16" fill="#d33"/>
  <text x="520" y="433" font-size="13" font-weight="700" fill="#111">5.5.2 GEMM</text>
  <text x="520" y="456" font-size="12" fill="#1a1a1a">
    <tspan x="520" dy="0">The MoE GEMM operates on a gathered, permuted activation</tspan>
    <tspan x="520" dy="16">matrix, and the down / gate-up projections differ in shape and</tspan>
    <tspan x="520" dy="16">arithmetic intensity. We compare our optimized MoE operations</tspan>
    <tspan x="520" dy="16">with the reference implementations across all layers, reporting</tspan>
    <tspan x="520" dy="16">the average kernel time per token where the top-8 experts are</tspan>
    <tspan x="520" dy="16">selected from the 256 routed experts. Across the layers, our</tspan>
    <tspan x="520" dy="16">MoE operations speed up 1.17&#215; to 1.89&#215;.</tspan>
  </text>
  <text x="520" y="590" font-size="13" font-weight="700" fill="#111">5.6 Optimizations for Speculative Decoding</text>
  <text x="520" y="612" font-size="12" fill="#1a1a1a">
    <tspan x="520" dy="0">We can skip the routing computations during verification: only</tspan>
    <tspan x="520" dy="16">the first token is sampled and the remaining tokens are</tspan>
    <tspan x="520" dy="16">discarded. In practice the routing computations take ~38% time</tspan>
    <tspan x="520" dy="16">on the sampled tokens. We can also skip the expert computation</tspan>
    <tspan x="520" dy="16">for tokens in a sequence known to be discarded, saving ~17% on</tspan>
    <tspan x="520" dy="16">the first token.</tspan>
  </text>
  <text x="40" y="708" font-size="11" fill="#888" font-family="-apple-system,'PingFang SC',sans-serif">摘录：attention / MoE GEMM 的带宽低效（QKV 投影算术强度低、bandwidth-bound）—— 本文 slc 优化的背景参考</text>
</svg>

### 1.1 gfx938（CDNA3）缓存链

| 层级 | 容量 | 管理方式 |
|------|------|----------|
| HBM | 峰值 **1.8 TB/s** | 带宽墙，memory-bound 的天花板（01 篇） |
| L2 | **8 MB**（全 CU 共享） | 硬件自动缓存，受 cache hint（slc/glc）影响 |
| L1 / LDS | LDS **64 KB/CU**（显式管理）；L1 是 `buffer_load` 走的硬件缓存 | LDS 只够 1 WG/CU，split-K 无空间（01/02 篇） |

64 个 CU。decode 算术强度 ≈ 2·M FLOP/byte，roofline 交点 M_ridge ≈ 155（560TFLOPS÷1.8TB/s），
**M < 155 全部 memory-bound**（M=32 带宽墙天花板仅 ~115 TFLOPS）：优化方向是**搬得更快**，
不是算得更快（01 篇判据）。

> 📊 **数据通路总图**见下方 1.5 节内联 SVG：权重(蓝)过 L2 但不进 LDS、标 evict-first；
> 激活(绿)经 LDS 中转、L2 常驻。

### 1.2 小 M decode GEMM 的流量结构（以 bf16 gate_up M=32 为例）

- 权重 N×K = 51200×5120×2B = **524MB**：**streaming**，每个输出 tile 读一遍自己那块，读完不再读（一次性）。
- 激活 M×K = 32×5120×2B = **327KB**：**复用**，被全部 N-tile 反复读，工作集小。
- 输出 M×N 写一次，占比小。

**"权重恰好全局读一遍"的推导**（winner kernel `MT128x32x128`，tile-M=128 / tile-N=32）：
M=32 ≤ tile-M=128 → **只有 1 个 M-tile**；N=51200 ÷ tile-N=32 = **1600 个 N-tile**，
每个 N-tile 只读自己那片 32×5120 的权重 → **权重总量恰好被读 1 次（纯流式）**；
而激活 327KB 被 1600 个 N-tile **各读 1 遍（复用 1600 次）**。

### 1.3 关键不等式与污染机制

权重 524MB **≫** L2 8MB（必然流过、不可能常驻）；激活 327KB **≪** 8MB（理可常驻）。理想：激活常驻 L2，权重流过不留痕，HBM 总字节 ≈ 权重 + 激活(一次) + 输出。

实际（默认策略）：权重的流式 load **填充 L2 cache line**，524MB 流过 = 反复冲刷 8MB，把激活的
line **逐出** → 激活下次被别的 N-tile 读就 miss 回 HBM。HBM 总字节多出"激活 miss × 复用次数"
项。**量化上限**：激活若每次读都 miss → 额外 HBM 流量 = 1600 × 327KB ≈ **524MB ≈ 权重流量翻倍**
（最坏 HBM 总流量 2×权重）；slc 后激活大部分 hit → 回到 ≈1× 权重（理想）。
**实测只拿到 -6.2%（01/02 篇），小于理论上限**：clean 下 LRU 本来就留下部分激活命中，且 HBM
读取与计算时间重叠。**-6.2% 不是天花板，只是当前 LRU 行为下的实测值**，别拿它当上限。

### 1.4 为什么这是"无损"优化

缓存策略只决定数据**从哪一级来**（L2 命中还是回 HBM），不改变数据内容本身：改后 kernel 输出
与原始 **bit-exact**（bf16 slc verify norm_error=5.57e-5，属 bf16 正常误差，02 篇）。这是 slc
相对 Layout A（重排权重、须 serving 侧配对）的根本优势：**只改二进制里一个 bit，零正确性风险**。

### 1.5 数据通路图：谁过路、谁常驻（D = A × W）

> 内联 SVG（Markdown 渲染器支持即显示）。权重(蓝)=过路旅客，激活(绿)=常驻居民，
> 琥珀虚框=slc 作用点。复用次数推导见 1.2。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 600" font-family="-apple-system,'PingFang SC','Microsoft YaHei',sans-serif">
  <defs>
    <marker id="aw" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="#2f6fdb"/></marker>
    <marker id="aa" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="#2f9e5b"/></marker>
  </defs>
  <rect width="1200" height="600" fill="#ffffff"/>
  <text x="600" y="34" text-anchor="middle" font-size="21" font-weight="700" fill="#1f2937">gfx938 GEMM 数据通路　HBM → L2 → 共享内存(LDS) → 寄存器(VGPR)</text>
  <text x="600" y="58" text-anchor="middle" font-size="13.5" fill="#5b6472">权重 = 过路旅客（流式，slc evict-first）　·　激活 = 常驻居民（高复用，留 L2）</text>
  <g fill="#4b5563">
    <rect x="40"  y="88" width="230" height="34" rx="6"/>
    <rect x="330" y="88" width="210" height="34" rx="6"/>
    <rect x="600" y="88" width="200" height="34" rx="6"/>
    <rect x="860" y="88" width="300" height="34" rx="6"/>
  </g>
  <g fill="#ffffff" text-anchor="middle" font-size="15.5" font-weight="700">
    <text x="155" y="111">HBM</text>
    <text x="435" y="111">L2  8MB</text>
    <text x="700" y="111">LDS  64KB/CU</text>
    <text x="1010" y="111">VGPR  /  mmac</text>
  </g>
  <g fill="#f6f7f9" stroke="#c7cdd6" stroke-width="1.4">
    <rect x="40"  y="130" width="230" height="344" rx="6"/>
    <rect x="330" y="130" width="210" height="344" rx="6"/>
    <rect x="600" y="130" width="200" height="344" rx="6"/>
    <rect x="860" y="130" width="300" height="344" rx="6"/>
  </g>
  <text x="55" y="150" font-size="12.5" font-weight="700" fill="#2f6fdb">权重 W（流式）</text>
  <g>
    <rect x="55" y="158" width="200" height="24" fill="#2f6fdb"/><text x="155" y="175" text-anchor="middle" fill="#fff" font-size="12" font-weight="700">W₀</text>
    <rect x="55" y="185" width="200" height="24" fill="#2f6fdb"/><text x="155" y="202" text-anchor="middle" fill="#fff" font-size="12" font-weight="700">W₁</text>
    <rect x="55" y="212" width="200" height="24" fill="#2f6fdb"/><text x="155" y="229" text-anchor="middle" fill="#fff" font-size="12" font-weight="700">W₂</text>
    <text x="155" y="256" text-anchor="middle" font-size="16" fill="#9aa2af">⋯</text>
    <rect x="55" y="264" width="200" height="24" fill="#2f6fdb"/><text x="155" y="281" text-anchor="middle" fill="#fff" font-size="11" font-weight="700">W₁₅₉₉</text>
  </g>
  <text x="155" y="308" text-anchor="middle" font-size="11.5" fill="#5b6472">524MB · 1600 片 · 各读 1 次</text>
  <rect x="55" y="345" width="200" height="70" rx="6" fill="#2f9e5b"/>
  <text x="155" y="375" text-anchor="middle" fill="#fff" font-size="14" font-weight="700">激活 A</text>
  <text x="155" y="397" text-anchor="middle" fill="#eafaf0" font-size="12">327KB · 复用 1600 次</text>
  <rect x="345" y="345" width="180" height="70" rx="6" fill="#2f9e5b"/>
  <text x="435" y="375" text-anchor="middle" fill="#fff" font-size="14" font-weight="700">A 常驻</text>
  <text x="435" y="397" text-anchor="middle" fill="#eafaf0" font-size="11.5">RESIDENT · 腾位保住</text>
  <rect x="345" y="170" width="180" height="92" rx="6" fill="#eaf1fd" stroke="#d97706" stroke-width="2.4" stroke-dasharray="7 4"/>
  <text x="435" y="200" text-anchor="middle" fill="#2f6fdb" font-size="13.5" font-weight="700">W evict-first</text>
  <text x="435" y="224" text-anchor="middle" fill="#b45309" font-size="12.5" font-weight="700">最先逐出</text>
  <text x="435" y="246" text-anchor="middle" fill="#5b6472" font-size="11">进 L2 但先牺牲</text>
  <text x="435" y="262" text-anchor="middle" fill="#b45309" font-size="11" font-weight="700">slc · bit17</text>
  <rect x="615" y="345" width="170" height="70" rx="6" fill="#2f9e5b"/>
  <text x="700" y="375" text-anchor="middle" fill="#fff" font-size="14" font-weight="700">A tiles</text>
  <text x="700" y="397" text-anchor="middle" fill="#eafaf0" font-size="11">ds_write → ds_read</text>
  <rect x="615" y="190" width="170" height="60" rx="6" fill="#ffffff" stroke="#c7cdd6" stroke-width="1.4" stroke-dasharray="5 4"/>
  <text x="700" y="220" text-anchor="middle" fill="#9aa2af" font-size="13" font-weight="700">权重不经过 ✕</text>
  <text x="700" y="240" text-anchor="middle" fill="#b6bdc7" font-size="11">W 绕过共享内存</text>
  <rect x="875" y="150" width="270" height="44" rx="6" fill="#2f6fdb"/>
  <text x="1010" y="178" text-anchor="middle" fill="#fff" font-size="13.5" font-weight="700">W　buffer_load 直入</text>
  <rect x="875" y="204" width="270" height="44" rx="6" fill="#2f9e5b"/>
  <text x="1010" y="232" text-anchor="middle" fill="#fff" font-size="13.5" font-weight="700">A　ds_read 回读</text>
  <rect x="875" y="288" width="270" height="92" rx="8" fill="#4b5563"/>
  <text x="1010" y="330" text-anchor="middle" fill="#fff" font-size="16" font-weight="700">mmac 计算</text>
  <text x="1010" y="356" text-anchor="middle" fill="#e5e7eb" font-size="13.5" font-weight="700">D = A × W</text>
  <path d="M255,205 L345,205" fill="none" stroke="#2f6fdb" stroke-width="3.2" marker-end="url(#aw)"/>
  <path d="M540,205 Q 700,110 875,172" fill="none" stroke="#2f6fdb" stroke-width="3.2" marker-end="url(#aw)"/>
  <path d="M1010,194 L1010,288" fill="none" stroke="#2f6fdb" stroke-width="3.2" marker-end="url(#aw)"/>
  <text x="700" y="104" text-anchor="middle" font-size="12" fill="#2f6fdb" font-weight="700">权重绕过 LDS → 直入寄存器</text>
  <path d="M255,380 L345,380" fill="none" stroke="#2f9e5b" stroke-width="3.2" marker-end="url(#aa)"/>
  <path d="M525,380 L615,380" fill="none" stroke="#2f9e5b" stroke-width="3.2" marker-end="url(#aa)"/>
  <path d="M785,375 Q 830,330 875,226" fill="none" stroke="#2f9e5b" stroke-width="3.2" marker-end="url(#aa)"/>
  <path d="M1010,248 L1010,288" fill="none" stroke="#2f9e5b" stroke-width="3.2" marker-end="url(#aa)"/>
  <text x="600" y="410" text-anchor="middle" font-size="12" fill="#2f9e5b" font-weight="700">激活经 LDS 中转</text>
  <g transform="translate(40,495)">
    <rect x="0" y="0" width="1120" height="92" rx="8" fill="#f6f7f9" stroke="#c7cdd6" stroke-width="1.4"/>
    <text x="20" y="28" font-size="13.5" font-weight="700" fill="#1f2937">读法</text>
    <rect x="20" y="42" width="16" height="16" fill="#2f6fdb"/>
    <text x="44" y="56" font-size="12.5" fill="#374151">权重 W：HBM → L2 → 寄存器（<tspan font-weight="700">不经 LDS</tspan>）；流式读 1 次 → 打 slc evict-first</text>
    <rect x="20" y="68" width="16" height="16" fill="#2f9e5b"/>
    <text x="44" y="82" font-size="12.5" fill="#374151">激活 A：HBM → L2 → LDS → 寄存器；复用 1600 次 → <tspan font-weight="700">留 L2 常驻</tspan></text>
    <rect x="700" y="42" width="26" height="16" fill="#eaf1fd" stroke="#d97706" stroke-width="2" stroke-dasharray="5 3"/>
    <text x="736" y="56" font-size="12.5" fill="#b45309" font-weight="700">slc 作用点（bit17）</text>
    <text x="700" y="82" font-size="12" fill="#5b6472">slc ≠ bypass：加位的权重仍进 L2，只是最先被逐</text>
  </g>
</svg>

### 1.6 流程图：无 slc vs 有 slc（左右对照）

> **权重 = 过路旅客，激活 = 常驻居民；slc 是给旅客的"优先让座"标记，保证居民不被挤走。**

**精确点 1：slc 不是"不缓存"，是"缓存但最先被逐"（evict-first 优先级标记）。**
权重加了 slc 后**仍会进 L1/L2**（mmac 马上就要读，短期复用还是要的），只是 L2 腾位置时
权重行**优先被牺牲**。真正"不进 L1"的是 **GLC 位（bit16）**，本项目**没用它**——
不要把 slc 理解成 cache bypass。

**精确点 2：权重"全局恰好读一遍"是推导出来的**（见 1.2）：M=32 ≤ tile-M=128 → 1 个 M-tile；
1600 个 N-tile 各读自己那片权重 → 权重纯流式 1 次；激活被 1600 个 N-tile 各读 1 遍（复用 1600 次）。

```
 数据角色        流量特征                无 slc (clean)                     slc 加在权重 load 上
 ─────────────  ──────────────────      ──────────────────────────────     ──────────────────────────────
 权重 524MB     全局读 1 次              HBM ──> L2/L1 ──> VGPR ──> mmac     同左，但缓存行标记 EVICT-FIRST：
 (B-operand)    (1600 N-tile 各读一片)   L2 被权重行持续填充，                        权重行照常进 L2/L1（短期复用），
                    冲刷 8MB L2           激活行被权重行逐出 (LRU)                  但腾位置时权重行先被牺牲
 激活 327KB     被 1600 N-tile 各读 1 遍  N-tile #1:   HBM 读入 (首次 miss)      N-tile #1:   HBM 读入 (首次 miss)
 (A-operand)    (复用 1600 次)            N-tile #2:   L2 MISS (被权重逐出)       N-tile #2:   L2 HIT ← 短链路
                                       ...  大量回 HBM                            N-tile #3..1600: 基本 HIT
                                       最坏额外 1600×327KB≈524MB                激活常驻 L2，链路变短
                                       HBM 总流量 ≈ 2×权重                      HBM 总流量 ≈ 1×权重（回到理想）
```

**理论上限 vs 实测**：最坏 +524MB 冗余（2×权重）→ 理想 1×权重，理论空间是"把流量砍半"；
实测 bf16 M=32 只有 **-6.2%**（01/02 篇），差在：clean 下 LRU 已留下部分激活命中 + HBM 与
计算重叠。理论上限是分析工具，-6.2% 是这台硬件当前 LRU 行为下的实测值，**两者都别当"天花板"**。

## 2. 指令层：slc 位是什么

### 2.0 两个 operand 的路径不一样（slc 只落在权重这条上）

反汇编核实（`buffer_load` 直接进 VGPR，`ds_read_b128` 从 LDS 回读）。两条路只有"是否过 LDS"不同：

```
  HBM                L2/L1 (缓存)           LDS                VGPR           mmac
  ───                ────────────          ───                ──────         ─────
  权重 W ──buffer_load─┐                                              ┌──v_mmac
  (B-operand)          │◄ 过 L2/L1，但【不进 LDS】──────────────► v[96:127]
                        │
  激活 A ─buffer_load─► VGPR ─ds_write─► LDS ─ds_read_b128─► VGPR ─┘
  (A-operand)          (先落寄存器)   (显式中转)   (读回寄存器)
```

| | 路径 | 关键指令 | 过 L2? | 过 LDS? |
|---|---|---|:---:|:---:|
| **权重 W（B-operand）** | HBM → L2/L1 → VGPR → mmac | `buffer_load`(idxen) 直接进 VGPR | ✓ | ✗ |
| **激活 A（A-operand）** | HBM → VGPR → LDS → VGPR → mmac | `buffer_load` + `ds_write` + `ds_read_b128` | ✓ | ✓ |

三个要点（易错点）：
- **"权重直达寄存器" ≠ 绕过缓存**。权重路中间**仍过 L2/L1**（`buffer_load` 走的硬件缓存），只是不碰 LDS。
- **slc 只打在权重这条的 `buffer_load` 上**（它过 L2 那一级的 cache line），标 evict-first。
- 激活虽走 LDS，它的 L2 命中与否仍由 **LRU + 权重是否占位** 决定——slc 把 L2 让给激活。**给激活加 slc = 方向反**。

### 2.1 两个 cache hint 位：word0 bit 图

gfx900 系 `buffer_load` 指令 **word0** 的缓存策略位（本例只动 bit17，其余位原样不动）：

```
  word0  (32 bit)  —— 只画相关位，其余位（寻址/宽度/LDS 偏移）不碰
  ┌───────────────────────────┬────────────┬────────────────────┐
  │   ... 寻址 / 宽度 / 偏移 ... │  bit17     │   bit16  ...        │
  │                            │   SLC ★    │   GLC   ...         │
  └───────────────────────────┴────────────┴────────────────────┘
                                  │
                                  ▼
                      翻这一位 = word0 |= 0x20000

  SLC (bit17)  ★ 本项目用  evict-first：进缓存但【优先逐出】
  GLC (bit16)  本项目不用   L1 无效，只走 L2
```

| bit | 名字 | 语义 | 用没用 |
|-----|------|------|:---:|
| **17** | **SLC** | evict-first：缓存但优先逐出 | ★ 用 |
| 16 | GLC | L1 无效，只走 L2 | 不用 |

通用改法 `word0 |= 0x20000`，**只翻 bit17，不碰其他任何位**。策略位与寻址/宽度/LDS 偏移字段独立
→ 不影响数值、不影响指令流 → **bit-exact 无损**。

> ⚠ 反复强调：slc **不是** "不缓存/bypass"。加 slc 的权重**仍会进 L2/L1**（mmac 马上要读，短期
> 复用要保），只是腾位时权重行**先被牺牲**。真正 bypass L1 的是 GLC(bit16)，本项目不用。

### 2.2 两种指令形态都要会认（位同为 bit17，编码随指令变）

同一个 bit17，落在两种**不同编码形态**的 `buffer_load` 上，改法（哪个字节翻哪一位）不同：

| 形态 | 位置 | 特征 | 编码变化（word0） | 主战场 |
|------|------|------|----------|--------|
| **lds 型** | prologue | 直接写 LDS（`buffer_load_dwordx4 ... offen lds`） | `E05D1000 → E05F1000` | fp8 FusedMatmul |
| **idxen 型** | K-loop | 向量索引加载（`... idxen`） | `E05C3000 → E05E3000` | bf16 Tensile Cijk K-loop |

**"哪组 load 是流式读权重的大头"必须反汇编数清**——bf16 gate_up M=32 winner kernel
（`Custom_Cijk_..._MT128x32x128_SE_K1_thread384`）里：

```
  winner kernel 的 load 组（流式读 524MB 大权重）
  ┌────────────────────────────────────────────────────────────┐
  │  prologue:  8 条 lds 型  ──►  一次性、量小 ──► 影响小        │
  │                                                            │
  │  K-loop:    20 条 idxen 型 ──►  每轮读一片权重 ──► 主体 ★    │
  │                          slc 打在这里                       │
  └────────────────────────────────────────────────────────────┘
  → 只改 prologue（8 条 lds）= 改错组 = 收益趋零
```

区分权重 load vs 激活 load（只给**一次性大流数据（权重）**加 evict-first，**复用数据（激活）**
保持默认）：激活 load 有独立特征（0x8003xxxx / idxen **SET** / 不同 s-operand）。
给激活加 slc = 亲手把它逐出，**方向反了**。

### 2.3 与 C++ intrinsic 的对应（patch bit ≡ 改源码常量重编译）

`__builtin_hcu_buffer_load_dwordx4(Para1, ..., Para5)` 的 **Para4/Para5 = glc / slc 常量**，
都是**编译期常量**，直接折进 word0 的 bit16/bit17：

```
  源码路径                        二进制路径
  ────────                        ────────
  Para5 (slc常量) = true   ──►  编译器折进 word0 bit17
        │                               ▲
        │                               │  产物字节一致
        └───────────────────────────────┘
              patch word0 |= 0x20000
```

**patch 二进制 word0 bit17 ≡ 把源码 Para5 翻成 true 再重编译**，两者生成的指令字节完全一致。
"改 bit 无损"的另一角度：策略位不含数据依赖，只由常量决定"从哪一级取数"。改 bit16 即改 Para4(GLC)，
本项目不用。
