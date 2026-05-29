# PoC T7 Report — Final Gate Decision (2026-05-30)

**实验**: T6 完成 — 1000 sample × 2 conditions × Claude Sonnet 4.6
**Cost**: $4.50, 27.4 min wallclock, 100% JSON parse success
**Conclusion**: Paper 原主线 "V12.31-augmented LLM > deployed LGBM" **数据反驳**。Pivot 到 Hybrid Pipeline.

---

## Full 3-Way Comparison (n=1000, stratified)

| Method | RankIC | Top10% ret | Top10 WR | Top20% ret | Top20 WR |
|---|---|---|---|---|---|
| **LGBM (E1.1, deployed-style)** | **+0.0677** | **+1.25%** | **56%** | **+1.47%** | **56%** |
| LLM raw (p_up signal) | +0.0138 | +0.91% | 51% | +1.33% | 53.5% |
| LLM expert (p_up signal) | -0.0161 | +1.16% | 54% | +0.93% | 53.5% |
| LLM raw (p_up - p_down) | +0.0128 | +0.82% | 52% | +1.33% | 53.5% |
| LLM expert (p_up - p_down) | -0.0077 | +0.39% | 51% | +0.44% | 54% |
| LLM raw (pump_ratio) | +0.0125 | +0.90% | 52% | +1.16% | 53% |
| LLM expert (pump_ratio) | -0.0070 | +0.62% | 51% | +0.64% | 54% |
| Expert rule (V12.31 onset_score) | -0.0337 | -0.14% | 52% | +0.90% | 56% |

## Stratum 拆解（最 instructive insight）

| Stratum | n | mean fwd_r5 | LGBM Top10% | LLM raw Top10% | LLM expert Top10% |
|---|---|---|---|---|---|
| high (expert onset 触发) | 250 | +0.50% | +0.11% | +0.25% | **-0.28%** |
| edge (3/4 信号触发) | 250 | -0.25% | +0.29% | **+3.09%** | +1.94% |
| low (≤1 信号触发) | 500 | +0.72% | **+2.21%** | +1.11% | +0.78% |

**核心发现**: 三种方法的 **task-fit 不同**：
- LGBM 强在 noisy strata (low stratum +2.21%)
- LLM 强在 edge cases (edge stratum +3.09%)
- Expert prompt 让 LLM 在 onset-triggered 样本上**过度自信** (high -0.28%)

## 三个 brutal honest findings

### Finding 1: 🔴 Paper 原 main claim FALSIFIED
"V12.31-augmented LLM > deployed LGBM" 不成立：
- LGBM 在所有 4 项指标上都 best
- 即使 LLM + V12.31 expert prompt，仍 -10% Top10% vs LGBM

### Finding 2: 🟡 Expert effect 真实但弱
- 100 sample: expert > raw +130% Top10% return (噪声)
- 1000 sample: expert > raw +27% Top10% return (真实但 marginal)
- Top10% winrate +3pp (54% vs 51%)

### Finding 3: 🔴 Expert 在 onset 类反而 worse
- high stratum (onset triggered): LLM expert -0.28% vs raw +0.25%
- 解读: expert prompt 让 LLM 过度信任 rule-detected onsets，不再做 critical 判断
- Implication: expert knowledge 应当作为 **assist**，不是 **dictator**

## 100 → 1000 Sample 的关键 lesson

| Metric | 100 sample (noisy) | 1000 sample (truer) | Δ |
|---|---|---|---|
| expert Top10% return | +2.88% | +1.16% | -60% (噪声褪去) |
| stratum edge mean fwd_r5 | -3.56% | -0.25% | 噪声 dominated |
| stratum high mean fwd_r5 | +1.23% | +0.50% | 噪声 dominated |

**Lesson**: paper-grade evidence 必须 ≥ 1000 sample，且需要 walk-forward + multi-seed。

## Paper Repositioning: Hybrid Pipeline (新主线)

### 现有 evidence 已经暗示 hybrid 潜力

| Stratum | LGBM | LLM raw | LLM expert | Winner |
|---|---|---|---|---|
| high (onset triggered) | +0.11% | +0.25% | -0.28% | LLM raw |
| edge (ambiguous) | +0.29% | **+3.09%** | +1.94% | **LLM raw (+3.09%)** |
| low (noise) | **+2.21%** | +1.11% | +0.78% | **LGBM (+2.21%)** |

**Hybrid hypothesis**: 用 confidence-based router，让每个 stratum 都用 best method:
- LGBM 高 confidence → 用 LGBM
- LGBM 中等 confidence (edge case) → 用 LLM raw
- LGBM 低 confidence + expert pattern triggered → 用 LLM expert

**预期效果**: 联合 LGBM 在 noisy strata + LLM 在 edge case 上的优势，应能超 LGBM alone。

### Paper title candidates

> **"Confidence-Routed Hybrid: When to Use Statistical Learning vs LLM Agents
> for Stock Movement Onset Detection"**

或：

> **"Architecture × Stratum Fit: A Hybrid LGBM-LLM Pipeline for Stock
> Onset Prediction"**

### Contributions (新方案)

| # | Contribution | Evidence |
|---|---|---|
| C1 | 实证发现：LGBM/LLM 各擅长不同 strata (architecture × task fit) | T6 stratum 数据 ✓ |
| C2 | 提出 confidence-based router 设计 | 待 hybrid 实验 |
| C3 | Hybrid 在 onset detection 上超 LGBM alone | 待 hybrid 实验 |
| C4 | 与 V12.31 deployed system (α +2.2pp/月) 对比 | 已有 baseline ✓ |

目标 venue：**KDD ADS / WSDM Industry / AAAI Industry / CIKM**
CCF A 概率: 20-30%

### 风险

| 风险 | 概率 | 对策 |
|---|---|---|
| Hybrid 不 work (1+1=1.5 而非 2) | 40% | fallback 到 "comparative study" paper (CCF B) |
| Stratum 区分需要在线计算 (实战瓶颈) | 30% | 用 LGBM confidence 作 routing signal (online cheap) |
| Hybrid 收益 < LGBM 的 noise floor | 30% | bootstrap CI 检验 robustness |

## T7 Decision: 实施 Hybrid Pipeline

### 后续工作 (1 周内)

1. **Hybrid router 实现** (`src/agent/hybrid_router.py`)
   - Input: 1000 sample + LGBM prediction + LLM raw + LLM expert
   - Routing rule: based on LGBM confidence + onset signal
   - Output: hybrid prediction signal

2. **3 种 routing 策略对比**
   - (a) LGBM-confidence-based: high conf → LGBM, low conf → LLM
   - (b) Stratum-based: high → LLM raw, edge → LLM raw, low → LGBM
   - (c) Ensemble: weighted average by stratum/conf

3. **Evaluation**:
   - Same 1000 sample, 3 hybrid strategies + 3 baselines
   - Plus 100-sample walk-forward (3 splits) on best hybrid

4. **Decision Gate**:
   - Hybrid Top10% > LGBM Top10% × 1.05 → 主线 PASS
   - Hybrid Top20% > LGBM Top20% × 1.05 → 主线 PASS
   - 任一 PASS → 继续；都失败 → fallback 到 CCF B "comparative study"

预算：未来 1 周不超过 $10 LLM cost (已经有 1000 sample 预测，只需小 routing 实验)。
