# 4. V12-Agentic Architecture

Figure 1 (to be added) depicts the V12-Agentic architecture. Four agents operate in a producer-consumer cascade: the Macro Regime Monitor and Alpha Factor Explorer prepare context, the Pattern Core produces per-stock onset probabilities, and the Backtest & Verifier combines all signals into a final prediction and emits feedback to the upstream agents. We detail each agent below.

## 4.1 Macro Regime Monitor

The Macro Regime Monitor observes market-wide indicators and outputs a binary disaster flag $z^M_t \in \{\text{normal}, \text{disaster}\}$. The flag is computed daily via a vote-2/3 composite of three signal groups:

**Signal A — Index AND**
$$
A_t = \mathbb{1}\!\bigl[r^{\text{SH}}_t < -0.02 \;\wedge\; r^{\text{GEM}}_t < -0.03\bigr]
$$
where $r^{\text{SH}}$ and $r^{\text{GEM}}$ are the Shanghai Composite (broad-market) and ChiNext (growth-stock) daily returns.

**Signal B — Volume OR.** $B_t = B^{(1)}_t \vee B^{(2)}_t \vee B^{(3)}_t$ where
- $B^{(1)}_t$: 5-day mean total amount / 20-day mean total amount $< 0.70$
- $B^{(2)}_t$: limit-down stock count $> 100$ OR limit-down/limit-up ratio $> 3$
- $B^{(3)}_t$: up-stock fraction $< 0.30$ OR down-stock fraction $> 0.70$

**Signal C — Sector inner-vote ≥ 2/3.** $C_t = \mathbb{1}\bigl[\,(C^{(1)}_t + C^{(2)}_t + C^{(3)}_t) \geq 2\bigr]$ where
- $C^{(1)}_t$: fraction of industries with negative mean return $> 0.80$
- $C^{(2)}_t$: top-5 hot concepts all negative
- $C^{(3)}_t$: top-5 hot concepts average return $< -0.01$

**Composite.** $z^M_t = \mathbb{1}\bigl[\,(A_t + B_t + C_t) \geq 2\bigr]$.

In our D1 dataset, this rule identifies 12 disaster days across 1062 trading days (1.13%), recovering the historically known disaster month of 202603 noted in V12.31 commit history. The full implementation is in `src/onset/disaster_filter.py`; we elaborate the threshold sensitivity in Section 6.

## 4.2 Alpha Factor Explorer

The Alpha Factor Explorer maintains the candidate feature pool and, critically, the encoded production expert knowledge.

**Factor pool.** We use the V12.31 153-factor library (technical indicators from TA-Lib plus custom features including pyramid-velocity, kline-pattern, money-flow strength) and 8 concept-heat factors derived from a multi-source concept database covering 5,932 stocks and 1,813 concepts. After excluding 10 forward-looking columns (`r5`, `r10`, `r20`, `r30`, `r40`, `dd5`–`dd40`) — these are V12.31 training-label assistants and are not features — the effective Pattern Core input dimension is 165.

**Expert Knowledge Prompt.** A core contribution of this paper is the **knowledge encoding methodology** for converting production system experience into a structured agent prompt. We elicited four rounds of input from the V12.31 designer through semi-structured interviews covering:

- **Round 1 — Bullish onset definition.** Three necessary conditions plus a bonus condition, formalized as:
  - **C1 Bottoms rising**: $\text{low}_{[t-4, t]} \geq 0.98 \cdot \text{low}_{[t-19, t]}$ (2% tolerance to avoid noise)
  - **C2 Above 5-day low**: $\text{close}_t \geq 1.05 \cdot \text{low}_{[t-4, t]}$ (5% confirmation gap)
  - **C3 MA pattern**: spread of $\{\text{MA}_5, \text{MA}_{10}, \text{MA}_{20}\} < 3\%$ AND $\text{MA}_5$ upward-tilting
  - **C4 (bonus) Volume boost**: 5-day mean volume $> 1.2 \times$ 20-day mean volume
- **Round 1 (cont.) — Bearish onset asymmetry.** Bearish onset is an *avoidance signal*, not a tradeable direction; it does not mirror the bullish rules.
- **Round 2 — Sub-pattern simplifications.** No W-bottom / U-bottom / circular-bottom subtype distinction; no market-cap or sector stratification; no breakout/reversal subtype split. These ablations were deliberately rejected by the designer to maintain interpretability.
- **Round 3 — Disaster month composite signal.** The vote-2/3 design formalized in §4.1.

The full prompt (~5,500 characters Chinese) is in Appendix A and `prompts/v12_31_expert_v1.md`. Validation: on the full D1 panel, the executable rule (`is_bullish_onset = C1 ∧ C2 ∧ C3`) fires at a rate of **7.76%**, closely matching the v3c production label positive rate (~8%).

**LLM-injected expert prompt.** For agent-based experiments, we inject this prompt at the head of the user message (not as a system prompt, to bypass observed API-gateway system-prompt truncation). The full prompt + per-anchor context yields a ~3,000-token input per LLM call.

## 4.3 Pattern Core (Phase 1: LightGBM)

The Pattern Core consumes the factor pool and emits per-stock onset probabilities $\hat{P}(z^O_{i,t} = k \mid \mathbf{x}_{i,t})$ for $k \in \{\text{down}, \text{neutral}, \text{up}\}$ (corresponding to label $y_{i,t} \in \{-1, 0, +1\}$).

**Model.** Phase 1 employs LightGBM (Ke et al. 2017) with the following hyperparameters: 63 leaves, 200 min-data-per-leaf, 0.05 learning rate, $\ell_1 = \ell_2 = 0.1$ regularization, 500 boosting rounds with early stopping at 30 rounds. Forward-label leakage prevention: ten columns (`r5/r10/r20/r30/r40`, `dd5/dd10/dd20/dd30/dd40`) are blacklisted; these are training-label-construction artifacts that exhibit $\text{spearman}(\text{r5}, r^{(5)}_{i,t}) = +0.8983$ — i.e., they are the forward return itself, not a past feature.

**Phase 2 and 3 plans.** Pathway 2 (this paper's follow-up) replaces the Phase 1 LGBM Pattern Core with a TCN-causal-dilated convolutional encoder followed by spatio-temporal cross-attention (Sec. 7). Pathway 3 further pretrains the encoder with the Barlow Twins redundancy-reduction objective (Zbontar et al. 2021). All three phases share the same Macro Regime Monitor, Alpha Factor Explorer, and Backtest Verifier — a controlled-comparison design that we believe is essential for honest ablation.

**Sample-and-conquer alternative: triple-barrier labeling.** As an additional baseline, we use López de Prado's triple-barrier method (E1.3 in our experiments) with $(u, d, H) \in \{(0.05, 0.03, 5), (0.08, 0.05, 20)\}$. While TBM does not condition on backward context (i.e., omits our Definition 1 condition 3), it provides a strong reference baseline for the value of the backward-context constraint.

## 4.4 Backtest & Verifier

The Backtest & Verifier combines outputs from the Pattern Core and the LLM-prompted Alpha Factor Explorer into a final hybrid signal, evaluates against held-out data, and emits feedback.

**Hybrid routing strategies.** We implement nine routing strategies, grouped by mechanism:

| Strategy | Mechanism | Signal formula |
|---|---|---|
| BL_LGBM | Pattern Core alone | $\hat{P}^{LGBM}_{\text{up}} / (\hat{P}^{LGBM}_{\text{down}} + 0.01)$ |
| BL_LLM_raw | LLM Alpha Explorer alone | $\hat{P}^{LLM}_{\text{up}}$ (no expert prompt) |
| BL_LLM_expert | LLM with V12.31 prompt | $\hat{P}^{LLM, \text{exp}}_{\text{up}}$ |
| A_confidence | Switch by LGBM confidence | LGBM if conf ≥ 0.5, else LLM |
| B_stratum | Switch by onset score | LLM if onset_score ≥ 3, else LGBM |
| C_soft_ensemble | Static weighted rank average | $w \cdot \text{rank}_{LGBM} + (1-w) \cdot \text{rank}_{LLM}$ |
| D_avoid_expert_onset | Avoid LLM expert on triggers | LLM raw if onset ≥ 3; LLM expert if 1 ≤ onset < 3; else LGBM |
| E_lgbm_floor_llm_boost | LGBM main + LLM boost | $\text{rank}_{LGBM} + b \cdot \mathbb{1}[\text{rank}_{LLM} \geq 0.75]$ |
| F_disaster_aware | Switch by macro flag | LLM if $z^M = \text{disaster}$, else LGBM |
| G_disagreement_boost | LGBM main + LLM-disagreement boost | LGBM rank, boosted where LLM disagrees AND is bullish |
| H_market_regime_ensemble | Regime + soft ensemble | Disaster: LLM; normal: $E$ with $b=0.15$ |
| I_onset_aware_boost | Onset-conditioned boost | LGBM rank + boost when expert onset AND LLM bullish |

**Walk-forward evaluation.** We adopt a strict expanding-window walk-forward protocol: 3 splits, each with 36-month training, 3-month validation, and 3-month test. Per split, we draw 2000 random test anchors (uniform within the test window), yielding 6000 total walk-forward anchors. For each anchor, we compute the LLM prediction (using `claude-sonnet-4-6` via OpenAI-compatible API) under both raw and expert conditions.

**Bootstrap CI.** We report 1000-resample 95% bootstrap confidence intervals for Top-10% and Top-20% returns and for per-split RankIC, both per-split and pooled.

**Reflection loop (planned).** In Pathway 2, the Backtest Verifier will emit periodic performance signals back to the upstream agents — e.g., a sustained RankIC decay signal triggers re-training of the Pattern Core, and persistent LLM-LGBM disagreement triggers re-elicitation of the expert prompt. The current paper does not include a closed-loop reflection experiment; we report the open-loop baseline as the foundation for future work.
