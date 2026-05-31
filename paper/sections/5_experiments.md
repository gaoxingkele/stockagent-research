# 5. Experiments

This section reports the main empirical results. We organize the evaluation around three sampling regimes (stratified PoC, walk-forward random, quarter-stratified) and twelve methods (3 baselines + 9 hybrid routers). All Top-K returns and rank ICs are computed against the realized 5-day forward simple return $r^{(5)}_{i,t}$.

## 5.1 Dataset

We use **D1**, a panel-format dataset built from Tushare Pro covering the Chinese A-share market from 2022-01-04 to 2026-05-27. After ST-stock filtering at the data layer (266 stocks excluded; cf. §4.2), the dataset contains:

- **5,275,812 stock-day anchors**
- **5,434 unique stocks**
- **1,062 trading days**
- **185 columns** (OHLCV + 173 factors from V12.31 factor library)

The bullish onset rate (V12.31 v3c label, Def. 1) is **8.0%** across the universe. After excluding the ten forward-looking columns identified in §4.3, the effective Pattern Core feature count is **165**.

Walk-forward LGBM models for the three test splits use expanding train windows:
- Split 1: train 2022-01-04 → 2024-12-31, val 2025-01 → 2025-03, test **2025-04 → 2025-06**
- Split 2: train 2022-01-04 → 2025-03-31, val 2025-04 → 2025-06, test **2025-07 → 2025-09**
- Split 3: train 2022-01-04 → 2025-06-30, val 2025-07 → 2025-09, test **2025-10 → 2025-12**

Per-split LGBM models achieve, on full test windows: Split 1 RankIC +0.069 IR 0.73; Split 2 RankIC −0.002 IR −0.03 (model fails); Split 3 RankIC +0.037 IR 0.30. The Split 2 LGBM failure is itself empirically interesting and motivates the regime-routing analysis in §5.7.

## 5.2 Baselines

We compare 12 methods, organized into three families:

**Pure baselines.**
- `BL_LGBM`: Pattern Core alone, with $\hat{P}_{\text{up}} / (\hat{P}_{\text{down}} + 0.01)$ as the ranking signal (the *pump-ratio* signal in V12.31 production).
- `BL_LLM_raw`: Claude Sonnet 4.6 prompted with a minimal system message (no V12.31 expert knowledge), returning $\hat{P}_{\text{up}}, \hat{P}_{\text{neutral}}, \hat{P}_{\text{down}}$ via strict JSON output.
- `BL_LLM_expert`: same model and output format, but with the V12.31 expert prompt (~5,500 characters) injected as the head of the user message.
- `BL_expert_rule`: the executable rule (`is_bullish_onset` from §4.2) used directly as a ranking signal.

**Five "naive hybrid" routers (A–E).** Hand-designed combinations of LGBM and LLM signals without explicit regime conditioning (Table 1, §4.4).

**Four "regime-aware" routers (F–I).** Routers conditioned on macro disaster signal (F), LGBM-LLM disagreement (G), market regime + ensemble (H), or onset-signal-aware boost (I).

LLM evaluation uses `claude-sonnet-4-6` via Cloubic's OpenAI-compatible endpoint. Each anchor incurs ~2,800 input tokens (raw) or ~3,200 input tokens (expert), with up to 200 output tokens. Total LLM cost for the walk-forward experiment (12,000 calls = 6,000 anchors × 2 conditions) is **\$27** USD across **168 minutes** of wall-clock time (8-worker concurrent execution).

## 5.3 Walk-Forward Protocol

Per the protocol fixed in §3.3 and §4.4, each split contributes 2,000 random anchors drawn uniformly from its test window (no stratification). Total: **6,000 walk-forward anchors**. Onset rates per split: 7.95%, 8.20%, 9.40% — closely tracking the universe rate of 8.0% and verifying the absence of stratification bias.

## 5.4 Main Results (Walk-Forward Random)

Table 1 reports per-split and pooled metrics for all 12 methods. We highlight three observations:

**Table 1: Walk-Forward Results (6,000 anchors, mean ± std across 3 splits)**

| Method | RankIC | Top-10% ret | Top-10% WR | Top-20% ret |
|---|---|---|---|---|
| BL_LGBM | **+0.088 ± 0.079** | **+1.63 ± 1.21%** | 60.7% | **+1.45 ± 1.02%** |
| BL_LLM_raw | −0.073 ± 0.057 | +0.36 ± 0.57% | 45.3% | +0.49 ± 0.21% |
| BL_LLM_expert | −0.103 ± 0.029 | +0.18 ± 0.20% | 44.6% | −0.05 ± 0.34% |
| BL_expert_rule | −0.089 ± 0.003 | +0.00 ± 0.74% | — | −0.07 ± 0.34% |
| A_confidence_0.5 | +0.074 ± 0.058 | +1.51 ± 0.83% | 57.7% | +1.37 ± 0.93% |
| B_stratum | +0.033 ± 0.068 | +1.06 ± 0.48% | 52.2% | +1.07 ± 0.63% |
| C_soft_w0.6 | +0.040 ± 0.048 | +0.85 ± 0.14% | 51.5% | +0.78 ± 0.23% |
| D_avoid_expert | −0.080 ± 0.040 | +0.24 ± 0.57% | 44.8% | +0.27 ± 0.26% |
| E_lgbm_floor_boost_0.15 | +0.075 ± 0.073 | +1.44 ± 0.80% | 57.5% | +1.41 ± 0.86% |
| E_lgbm_floor_boost_0.30 | +0.060 ± 0.064 | +1.24 ± 0.55% | 54.8% | +1.17 ± 0.58% |
| F_disaster_aware | +0.060 ± 0.057 | +1.23 ± 0.83% | 58.5% | +1.22 ± 0.85% |
| **G_disagreement_boost** | **+0.079 ± 0.077** | **+1.63 ± 1.21%** | **61.3%** | +1.43 ± 1.04% |
| H_market_regime_ensemble | +0.047 ± 0.052 | +1.21 ± 0.62% | 55.7% | +1.21 ± 0.72% |
| I_onset_aware_boost | +0.069 ± 0.079 | +1.35 ± 0.88% | 57.3% | +1.39 ± 0.97% |

**Observation 1: LGBM dominates LLM-only baselines.** Pure LLM baselines (raw and expert) achieve consistently negative RankIC and substantially lower Top-10% return than LGBM (e.g., +1.63% vs +0.36%). In the realistic random-sample regime, the LLM agent is **not** a competitive standalone ranker. This finding contrasts sharply with the same model's performance in the stratified PoC regime (§5.5).

**Observation 2: Hybrid routing achieves near-parity with LGBM, not exceeding it.** The best hybrid router (G_disagreement_boost) ties LGBM's mean Top-10% return at +1.63%, with marginally higher Top-10% winrate (61.3% vs 60.7%) and identical pooled mean. None of the hybrid variants achieves a statistically significant improvement: 95% bootstrap CIs heavily overlap with the BL_LGBM CI of $[+1.07\%, +2.06\%]$ pooled.

**Observation 3: Per-split sign reversals reveal cross-quarter heterogeneity.** Examining individual splits (Table 1 supplementary):
- Split 1 (2025-Q2): BL_LGBM Top-10% **+3.34%** dominates.
- Split 2 (2025-Q3): BL_LGBM **fails** (Top-10% +0.88%, RankIC −0.015); BL_LLM_raw is competitive (+1.10%).
- Split 3 (2025-Q4): E_lgbm_floor_boost_0.30 (+1.48%) and BL_LGBM (+0.68%) flip — the hybrid dominates.

This per-split heterogeneity motivates the regime-aware routing analysis in §5.7.

## 5.5 The Stratified-vs-Random "Reversal" Is Within Noise

A recurring claim in LLM-finance work is that hybrid LLM signals beat a strong tabular baseline. On our own stratified proof-of-concept (PoC) we initially observed exactly such a result, which then disappeared on the walk-forward random sample. The two point estimates are reproduced in Table 2.

**Table 2: Stratified PoC vs Walk-Forward Random (point estimates)**

| Method | Stratified PoC (n=1000, 25% onset) | Walk-Forward Random (n=6000 pooled, 8% onset) | Δ |
|---|---|---|---|
| BL_LGBM Top-10% | +1.25% | +1.56% | +0.31pp |
| BL_LLM_raw Top-10% | +0.91% | +0.32% | **−0.59pp** |
| BL_LLM_expert Top-10% | +1.16% | +0.26% | **−0.90pp** |
| E_lgbm_floor_boost_0.30 Top-10% | **+1.93%** | +1.32% | **−0.61pp** |
| BL_expert_rule RankIC | +0.108 | −0.086 | **−0.194** |

Taken at face value, the best hybrid "beats LGBM by +54%" on the stratified PoC (+1.93% vs +1.25%) and "trails LGBM by −15%" on the random sample. It is tempting to attribute this to a base-rate mechanism. We instead ran three controlled experiments — all on the *already-scored* anchors, so no new LLM cost — and find that the effect is **not a base-rate artifact and is not statistically distinguishable from zero**.

**(i) No composition of the random pool reproduces the reversal.** Holding the 6,000 walk-forward anchors and their LLM scores fixed, we sweep the evaluation set's marginal onset rate (the `is_bullish_onset` fraction) and, separately, the expert-score stratification axis used by the PoC (the `onset_score ≥ 3` fraction) across 5–50%, recomputing all signals at each composition. As shown in Figure (`c3_dose_response`), the hybrid's Top-10% return stays **below** LGBM at *every* composition under both sweeps (Δ between −0.23pp and −0.81pp, no zero-crossing). If base-rate compression drove the PoC result, raising the onset rate of the random pool to 25% should reproduce hybrid > LGBM; it does not. **Base rate is not the cause.**

**(ii) A stronger base ranker does not erase the edge.** Restricting to the 583 PoC anchors that fall inside the split-2/3 test windows (where the walk-forward LGBM is legitimately out-of-sample), we hold the anchors and LLM scores fixed and swap only the LGBM base ranker: the PoC's original single-window model (RankIC +0.028) vs the stronger walk-forward per-split model (RankIC +0.056). The hybrid-minus-LGBM gap is essentially unchanged (+0.70pp → +0.81pp). **Baseline strength is not the cause either.**

**(iii) The PoC edge is within date-clustered noise.** Anchors sharing a `trade_date` are cross-sectionally correlated; an anchor-independent bootstrap (as is common in the field) understates uncertainty. Re-evaluating the full 1,000-anchor PoC with a **date-clustered** bootstrap (1,000 resamples over dates), the headline gap is Δ(Hybrid − LGBM) Top-10% = **+0.72pp, 95% CI [−0.56pp, +2.05pp]** — comfortably spanning zero. The "+54%" is a ratio of two small-sample, date-correlated means, neither of which is significantly above the other.

**Reframed finding.** The apparent stratified-vs-random reversal is the difference between two point estimates that are *each* statistically indistinguishable from zero for the hybrid-vs-LGBM contrast. This is consistent with Observation 2 of §5.4 (hybrid achieves near-parity, never significant superiority). The methodological warning is therefore sharper than "stratification inflates effects via base rate": **at the sample sizes typical of LLM-finance studies, hybrid-over-baseline effects of this magnitude are within date-clustered noise, and benchmarks that report such effects as ratios, without cluster-robust confidence intervals, will systematically manufacture spurious "LLM helps" claims.** We urge the field to report cluster-robust intervals on the *difference* of interest, matched-deployment distributions, and absolute (not ratio) effect sizes.

## 5.6 Regime-Aware Routing and the Oracle Upper Bound

To probe whether routing could recover the per-split heterogeneity observed in §5.4 Observation 3, we evaluate four regime-aware strategies (F–I) on the walk-forward 6,000-anchor panel.

**Oracle upper bound.** We compute the highest achievable per-split Top-10% by retrospectively selecting the best method for each split:
- Split 1 Oracle: BL_LGBM +3.34%
- Split 2 Oracle: BL_LLM_raw_ratio +1.24%
- Split 3 Oracle: E_lgbm_floor_boost_0.30 +1.48%
- **Mean Oracle Top-10%: +2.02%**
- **Mean BL_LGBM Top-10%: +1.63%**
- **Oracle − LGBM: +0.39pp** (this is the *upper bound* for any online quarter-conditional router)

**Online routers achieve a fraction of this oracle headroom.** Among the 4 regime-aware routers (Table 1):
- G_disagreement_boost: +1.63% (essentially tied with LGBM mean; +0.00pp absolute gain).
- F_disaster_aware: +1.23%.
- H_market_regime_ensemble: +1.21%.
- I_onset_aware_boost: +1.35%.

G recovers ≈ 0 of the +0.39pp oracle headroom, suggesting that LGBM-LLM disagreement alone is insufficient signal for regime routing. F's disaster signal is too sparse (only 30 out of 6,000 anchors flagged as disaster). More expressive routing — incorporating rolling LGBM RankIC, sector momentum, or learned regime embeddings — is a promising direction (§7 and Pathway 2).

**Statistical significance.** Across all hybrid methods, the 95% bootstrap CI for the Top-10% difference vs BL_LGBM straddles zero. Specifically, for G_disagreement_boost vs BL_LGBM: 95% CI = $[-0.19\%, +0.31\%]$; for E_lgbm_floor_boost_0.15: $[-0.13\%, +0.30\%]$. We conclude that no online router robustly beats LGBM in this experiment, and we explicitly flag this as a negative finding for the field.

## 5.7 Cross-Quarter Heterogeneity Detail

Table 3 (Appendix) reports the full per-split breakdown. Patterns by quarter:

| Split | BL_LGBM | BL_LLM_raw | BL_LLM_expert | Best hybrid | Best method overall |
|---|---|---|---|---|---|
| 1 (2025-Q2) | **+3.34%** | −0.30% | +0.20% | E_0.30 +1.76% | **BL_LGBM** |
| 2 (2025-Q3) | +0.88% | **+1.10%** | +0.47% | E_0.15 +0.68% | **BL_LLM_raw** |
| 3 (2025-Q4) | +0.68% | +0.27% | +0.04% | **E_0.30 +1.48%** | **E_0.30 (hybrid)** |

Three different methods are optimal on the three splits — a clean illustration of regime heterogeneity. The challenge for future work is to detect *online* which method to deploy *next quarter*. We hypothesize that the recent rolling RankIC of each method, computed on a 30-day window of prior predictions and realized returns, can serve as a routing signal. We leave the implementation and evaluation of this richer regime-conditioning to Pathway 2.
