# Paper Outline — Pathway 1: Agentic Framework (复用现有 evidence)

**Working Title**:
> **V12-Agentic: A Multi-Agent Framework for Stock Movement Onset Detection under Short-Sale Constraints**

**Target Venue (P1)**: KDD ADS Track / WSDM Industry / AAAI Industry
**Fallback Venue**: CIKM Long Paper / ICDM
**Estimated CCF A probability**: 25-35%
**Timeline**: 4-6 weeks
**Code change required**: ~0% (复用全部 V12.31 + walk-forward evidence)
**Paper writing only**

---

## 1. Strategic Framing (与用户白皮书对齐)

Architecture: **4-Agent System**

| Agent | Role | Implementation status |
|---|---|---|
| **Macro Regime Monitor** | Detect market regime, disaster months, sector rotation | ✅ V12.31 `market_context.py` (1629 lines) + `disaster_filter.py` |
| **Alpha Factor Explorer** | Generate / select features dynamically | ✅ V12.31 153 factor lab + 8 concept heat factors; LLM-prompted expert knowledge |
| **Pattern Core** | Core onset classifier (Phase 1: LGBM; Phase 2: TCN+Cross-Attn; Phase 3: + Barlow Twins SSL) | ✅ LGBM (Phase 1) ; ❌ TCN (Phase 2) ; ❌ SSL (Phase 3) |
| **Backtest & Verifier** | Online evaluation + reflection loop + router selection | ✅ `hybrid_router.py` + `eval_wf_hybrid.py` walk-forward eval |

**Key claim**: This Agentic decomposition + V12.31 deployed expert knowledge + walk-forward validation = a **systematic, deployment-ready** framework for stock movement onset detection in markets with short-sale constraints.

---

## 2. Paper Structure (10-12 pages)

```
1. Introduction                                       1.5 pages
2. Related Work                                       1.5 pages
3. Movement Onset: Problem & Asymmetry Framework      1.0 page
4. V12-Agentic Architecture                           2.0 pages
   4.1 Macro Regime Monitor (V7c + disaster filter)
   4.2 Alpha Factor Explorer (153 factors + concept heat + V12.31 expert knowledge)
   4.3 Pattern Core (LGBM + Triple-Barrier label engineering)
   4.4 Backtest & Verifier (hybrid router + walk-forward)
5. Experiments                                        2.5 pages
   5.1 Datasets (D1 A-shares 2022-2026, 5,275,812 rows)
   5.2 Baselines (LGBM, raw LLM, expert LLM, expert rule)
   5.3 Walk-forward protocol (3 splits × 2000 random anchors)
   5.4 Main results (RankIC, Top10%, Top20%, bootstrap CI)
   5.5 Per-split + Pooled comparison
   5.6 Stratified vs random sampling artifact analysis
   5.7 Regime-aware routing (oracle upper bound + 5 routers)
6. Ablations & Sensitivity                            1.0 page
   6.1 Stratification effect (PoC vs walk-forward)
   6.2 Boost magnitude sweep
   6.3 LLM model size (Sonnet vs Haiku)
   6.4 Expert prompt + Stratum interaction
7. Discussion                                         0.5 page
   7.1 Why stratified sampling overstates LLM advantage
   7.2 When does LLM help — quarter heterogeneity
   7.3 Path to Pattern Core upgrade (TCN, Barlow Twins) — future work
8. Conclusion                                         0.3 page
References + Appendix                                 2-3 pages
```

---

## 3. Contributions Statement (4 C's)

**C1 — Empirical**:
We present **V12.31**, a deployed multi-agent system with documented α +2.236pp/month and Sharpe 2.20 on Chinese A-share markets, and **a knowledge encoding methodology** that converts the deployed system's tacit expert knowledge (Movement Onset detection rules, disaster-month signals, asymmetric short-sale constraints) into structured prompts for LLM agents.

**C2 — Methodological**:
We propose a **4-agent decomposition** (Macro Regime Monitor + Alpha Factor Explorer + Pattern Core + Backtest Verifier) for stock movement onset detection that systematizes industrial best practices in a paper-ready framework.

**C3 — Critical**:
We empirically demonstrate a **stratified-sampling artifact** in hybrid LLM-LGBM evaluations: PoC stratified sampling overstates LLM contribution by +54% Top10% return, whereas true random walk-forward shows near-equivalence. This is a methodological warning to the field.

**C4 — Analytical**:
On 6000 walk-forward anchors across 3 quarterly splits, we analyze when LLM contributes alpha (cross-quarter heterogeneity) and provide an **oracle upper bound** (+0.34pp Top10%) for any online router design — a useful constraint for future hybrid system designs.

---

## 4. Section-by-Section Draft Plan

### §1 Introduction (1.5 pages)

**Hook**: Industrial vs academic perspective gap in stock movement prediction. Most academic methods evaluate on synthetic / stratified subsets; deployed systems struggle with real distribution drift.

**Motivation**:
- A-share market has cap of ±10% per day, asymmetric short-sale (T+1, restricted)
- Onset events are sparse (~8% true rate in real distribution)
- LLM agents are 2024-26 hot topic but real-world evaluation lacking
- Need: framework that combines deployed system knowledge + LLM reasoning

**Approach**: V12-Agentic 4-agent framework, evaluated rigorously.

**Contributions**: 4 listed above.

**Outline**: §2 Related, §3 Problem, §4 Architecture, §5 Experiments, §6 Ablations, §7 Discussion.

### §2 Related Work (1.5 pages)

**2.1 Stock Movement Prediction**
- López de Prado (2018) Triple-Barrier Method (TBM)
- HIST (KDD'21), MASTER (AAAI'24), StockMixer (AAAI'24)
- Direction: continuous regression vs binary/multiclass classification

**2.2 LLM Agents in Finance**
- FinGPT, FinMA, FinLLaMA (finance-tuned LLMs)
- FinAgent (KDD'24), AlphaForge (KDD'24), TradingGPT
- Direction: full delegation vs hybrid with statistical models

**2.3 Multi-Agent Systems**
- LLM agent coordination literature
- Reflection / self-correction (Shinn 2023)

**2.4 Self-Supervised Representation for Time Series**
- TS2Vec, COST, Barlow Twins (Zbontar 2021)
- TCN (Bai 2018), Causal Cross-Attention

**2.5 Short-Sale Constraints**
- Atilgan et al. 2022 (Chinese market asymmetry)
- Implication on label design

**Differentiator**: Our work is the **first** to systematically encode deployed quant system knowledge into agent prompts AND validate on real-world walk-forward distribution.

### §3 Problem & Asymmetry (1 page)

Formal definitions:
- Movement Onset: P_{i,t+H}/P_{i,t} ≥ θ_up AND max drawdown ≤ θ_dd
- Backward-Context: past_r5 ≤ τ (V12.31 v3c production label)
- Asymmetric: short-sale constraint → bearish onset is avoid-signal, not entry

Formalize 4-agent state space:
- z^M (regime) ∈ {normal, disaster}
- z^O (per-stock onset) ∈ {rest, bullish_onset, bearish_onset, trend, exhaustion}
- Triple-Barrier label: {-1, 0, +1, dynamic trailing}

### §4 V12-Agentic Architecture (2 pages)

**Section 4.1 — Macro Regime Monitor**
- V12.31 indices (上证, 创业板, 中证500/1000)
- Disaster signal: vote ≥ 2/3 of {index, volume, sector}
- Implementation: `market_context.py` + `disaster_filter.py`

**Section 4.2 — Alpha Factor Explorer**
- 153 traditional factors (TA-Lib + custom)
- 8 concept heat factors (5932 stocks × 1813 concepts)
- V12.31 expert knowledge prompt (Round 1-3 captured via interview)
- LLM extracted expert rules as feature/prompt

**Section 4.3 — Pattern Core (Phase 1: LGBM)**
- Multi-class onset classifier
- Forward-label leakage handling (r5/r10/r20/r30/r40/dd5-dd40 blacklist)
- Production-style: ST filter at data layer
- Future work: TCN + Cross-Attention (Pathway 2), Barlow Twins SSL (Pathway 3)

**Section 4.4 — Backtest & Verifier**
- Hybrid router: 5 strategies (A confidence, B stratum, C soft, D avoid expert, E LGBM+LLM boost) + 4 regime-aware (F disaster, G disagreement, H ensemble, I onset-aware)
- Walk-forward evaluation
- Bootstrap CI

### §5 Experiments (2.5 pages)

**5.1 Dataset**
- D1 A-shares Tushare: 5,275,812 rows × 5434 stocks × 1062 trading days (2022-01 to 2026-05)
- 185 columns (OHLCV + 173 factors)
- ST filter source-level (excluded 266 stocks → 5434)
- Forward-label fields blacklist (10 fields)

**5.2 Baselines**
- BL_LGBM: trained on 36-month rolling window, 165 features after blacklist
- BL_LLM_raw: Sonnet 4.6, minimal system prompt
- BL_LLM_expert: Sonnet 4.6, V12.31 knowledge in user-prefix
- BL_expert_rule: Pure rule (bottoms_rising + above_5d_low + ma_pattern + vol_boost)

**5.3 Walk-Forward Protocol**
- 3 splits, each 36-month train + 3-month val + 3-month test
- Per split: random sample 2000 anchors (real distribution, ~8% onset rate)
- Total: 6000 anchors

**5.4 Main Results — Per-split and Pooled (Table 1)**
- All metrics: RankIC, Top10% / Top20% return, winrate, bootstrap CI
- 14 methods × 3 splits × pooled

**5.5 Stratified vs Random Artifact Analysis (Table 2 + Figure 1)**
- PoC stratified (n=1000, 25% high stratum): H_E_0.30 +54% over LGBM
- WF random (n=6000, 8% onset): H_E_0.30 -15% vs LGBM
- Methodological warning for the field

**5.6 Regime-Aware Routing (Table 3)**
- F/G/H/I + oracle upper bound (+0.34pp Top10%)
- G_disagreement_boost: +0.07pp marginal (not significant)

### §6 Ablations (1 page)

- Boost magnitude: 0.15, 0.30, 0.50 across H_E variants
- LLM model: Sonnet vs Haiku (Haiku quality drop +0.22% vs +1.16% Sonnet expert)
- Expert prompt: helps in stratified, hurts in random
- Sample size: stratified bias visible from 100 → 1000 → 6000

### §7 Discussion (0.5 page)

- When LLM helps: ambiguous mid-conviction cases (stratified edge stratum +3.09%)
- When LLM hurts: high-conviction onset cases (stratified high stratum -0.28%)
- Quarter heterogeneity: Split 1 LGBM dominant, Split 2 LLM raw dominant, Split 3 hybrid dominant
- Path to upgrade: TCN + Cross-Attention pattern core (Pathway 2)

### §8 Conclusion (0.3 page)

- Framework systematizes industrial best practices
- Honest negative findings: hybrid does not robustly beat LGBM in random sampling
- But: oracle headroom + stratified artifact + cross-quarter heterogeneity = valuable insights
- Future work: TCN/Cross-Attention/Barlow Twins (Pathway 2/3)

---

## 5. Tables Plan

| Table | Content | Source |
|---|---|---|
| T1 | Main results: 14 methods × {per-split, pooled} × {RankIC, Top10%, Top20%, WR, CI} | `wf_hybrid_eval/metrics.json` + `wf_regime_eval/metrics.json` |
| T2 | Stratified vs random: 5 methods × 2 sampling regimes × Top10% | PoC `poc_full` + WF `wf_*` |
| T3 | Regime routers + oracle upper bound: F/G/H/I + best-per-split | `wf_regime_eval/metrics.json` |
| T4 | Ablation: boost magnitude × {0.15, 0.30, 0.50} × {pooled, per-split mean} | `wf_*` |
| T5 | LLM model comparison: Sonnet vs Haiku × {raw, expert} × {RankIC, Top10%} | PoC results |
| T6 | Per-split agent contribution: regime monitor signals × {pre, during, post disaster} | new analysis needed |

## 6. Figures Plan

| Figure | Content |
|---|---|
| F1 | V12-Agentic 4-agent architecture diagram |
| F2 | Walk-forward split timeline with disaster overlay |
| F3 | Stratified vs random sample distribution + Top10% comparison |
| F4 | Per-method Top10% return distribution (boxplot) across 3 splits |
| F5 | Oracle vs best router gap visualization |
| F6 | LLM-LGBM disagreement signal: high vs low → outcome |

---

## 7. Timeline (Pathway 1)

| Week | Deliverable |
|---|---|
| **1** | Outline + Intro + Related Work draft (this week) |
| 2 | Problem + Architecture sections + V12.31 knowledge appendix |
| 3 | Experiments section + Table 1-3 + bootstrap analyses |
| 4 | Ablations + Discussion + Figures |
| 5 | Conclusion + Appendix + bibliography + internal review |
| 6 | Final polish + submission prep |

**Submission target**: end of week 6.

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Reviewer says "Pattern Core is just LGBM, not novel enough" | Frame as Phase 1 baseline; cite Pathway 2 (TCN) as ongoing work; emphasize systematization contribution |
| Reviewer says "negative finding (hybrid doesn't beat LGBM)" not a paper | Reframe as methodological insight (stratification artifact); +0.34pp oracle headroom is a positive contribution |
| Walk-forward only on Chinese A-shares | Acknowledge limitation; cite Pathway 3 plans (multi-market) |
| 6000 anchors may be considered small for ML venue | DM test + bootstrap CI + per-split breakdown shows real, not just lucky |

---

## 9. Immediate Next Actions

1. ✅ Outline (this file)
2. 📝 Section 1 (Intro) draft
3. 📝 Section 2 (Related Work) draft
4. 📝 Section 4 (Architecture) — leverage existing V12-Agentic mapping
