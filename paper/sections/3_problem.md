# 3. Movement Onset: Problem and Asymmetry Framework

## 3.1 Definitions

Let $\mathcal{S}$ denote the universe of stocks and $\mathcal{T}$ the set of trading days. For each $(i, t) \in \mathcal{S} \times \mathcal{T}$, let $P_{i,t}$ denote the (split- and dividend-adjusted) closing price and let
$$
r^{(h)}_{i,t} = \frac{P_{i,t+h}}{P_{i,t}} - 1
$$
denote the forward $h$-period simple return.

**Definition 1 (Movement Onset — Bullish).**
A *bullish onset* occurs at $(i, t)$ if the following three conditions hold simultaneously:

1. **Forward magnitude**: $r^{(H)}_{i,t} \geq \Delta^+$
2. **Bounded drawdown**: $\min_{0 \leq h \leq H} P_{i,t+h} / P_{i,t} \geq 1 - \Delta^-$
3. **Backward context**: $\sum_{s=t-W}^{t-1} \log\!\bigl(P_{i,s+1}/P_{i,s}\bigr) \leq \tau$

In the V12.31 production system, $(H, \Delta^+, \Delta^-, W, \tau) = (5, 0.10, 0.05, 5, 0.08)$ — five-day forward window, requires ≥10% appreciation, ≤5% maximum drawdown, conditional on past 5-day cumulative return not exceeding 8%. The backward-context condition prevents the model from classifying already-trending stocks as "onsets" (Section 4.2).

**Definition 2 (Movement Onset — Bearish).**
A *bearish onset* is defined symmetrically by replacing the forward inequalities with $r^{(H)}_{i,t} \leq -\Delta^+$ (cumulative decline) and $\max_{0 \leq h \leq H} P_{i,t+h} / P_{i,t} \leq 1 + \Delta^-$ (bounded rebound).

**Definition 3 (Onset Label).**
The onset label $y_{i,t} \in \{-1, 0, +1\}$ assigns +1 to bullish onsets, −1 to bearish onsets, and 0 to all other states (rest, trend continuation, exhaustion).

## 3.2 Asymmetric Short-Sale Constraint

In the Chinese A-share market, the vast majority of retail-accessible stocks cannot be shorted: only a small subset is borrowable, and the cost-to-borrow is prohibitive. Consequently, a *bearish onset* (negative-direction signal) **cannot be monetized as a short position** in standard retail accounts. It can only be used as an **avoidance criterion** — i.e., to exclude the affected stock from a long-only portfolio.

This asymmetry has two consequences for our framework:

**Proposition 1 (Bearish-as-Filter).** Under short-sale constraints, the optimal use of a bearish onset prediction is to remove the stock from the candidate pool, not to take a short position. Formally, given a long-only portfolio $\pi$ and predicted bearish probabilities $\hat{p}^-_{i,t}$, optimal $\pi$ satisfies
$$
\pi_{i,t} = 0 \quad \text{whenever} \quad \hat{p}^-_{i,t} \geq \theta_{\text{avoid}}.
$$

In our system, $\theta_{\text{avoid}}$ is set to the 80th percentile of $\hat{p}^-$ within the V7c candidate pool (see Section 4.2 V12.31 v3c production label).

**Proposition 2 (Asymmetric Information Diffusion, after Atilgan et al. 2022).** In markets where short-selling is restricted, negative information from peer firms (supply chain, industry) carries disproportionately more predictive power than positive information of equal magnitude. This empirical finding motivates our **two-track architecture**: a primary long-bullish-onset prediction track, and a separate bearish-onset filter that contracts the long-only candidate pool.

## 3.3 Sparse-Event Distribution and Sampling Considerations

Under Definition 1, the bullish onset rate on the D1 5.27M-anchor universe is approximately **8.0%** (Table to be added in §5). Three sampling regimes appear in our experiments:

1. **Stratified-PoC** (Section 5.5): 250 onset-positive + 250 partial-trigger + 500 non-onset → 25% effective positive rate.
2. **Walk-Forward-Random** (primary): uniform random sampling within each test window → 7.9–9.4% positive rate, matching the production deployment distribution.
3. **Quarter-Stratified** (Section 5.7, oracle analysis): random within each 3-month split.

We report results under all three regimes and demonstrate (Section 5.5) that Stratified-PoC overstates hybrid LLM-LGBM advantage by a factor inversely proportional to the true onset rate — a key methodological warning.

## 3.4 The Four-Agent State Space

We decompose the prediction task into four latent state variables aligned with the four agents:

| State | Description | Estimator |
|---|---|---|
| $z^M_t$ | Macro market regime: $\{\text{normal}, \text{disaster}\}$ | Macro Regime Monitor (Sec. 4.1) |
| $\mathbf{z}^F_t$ | Active alpha factor pool (subset of 153 factors) | Alpha Factor Explorer (Sec. 4.2) |
| $z^O_{i,t}$ | Per-stock onset state: $\{\text{rest}, \text{bull-onset}, \text{bear-onset}, \text{trend}, \text{exhaustion}\}$ | Pattern Core (Sec. 4.3) |
| $\hat{y}_{i,t}$ | Final hybrid prediction | Backtest & Verifier (Sec. 4.4) |

The macro state $z^M_t$ modulates the per-stock $z^O_{i,t}$ posterior (the disaster filter), and the active factor pool $\mathbf{z}^F_t$ is conditioned on both — making the framework adaptive to non-stationary market conditions. We formalize the agent-wise inference in Section 4 and report the empirical interaction effects in Section 5.

## 3.5 Evaluation Targets

Throughout the paper we report the following metrics, all computed on the test windows of three walk-forward splits:

- **RankIC**: cross-sectional Spearman rank correlation between the model's signal and the realized $r^{(5)}_{i,t}$, averaged per date.
- **Top-K return**: mean realized $r^{(5)}_{i,t}$ among the top-$K$ candidates by signal (K $\in \{$Top-10\%, Top-20\%$\}$).
- **Top-K winrate**: fraction of top-$K$ candidates with $r^{(5)}_{i,t} > 0$.
- **Bootstrap 95% CI** on the above, via 1000 re-samples within each split.

We do not report Sharpe ratio in the main tables because the K-day prediction target does not directly translate to an annualized portfolio strategy without further assumptions; we instead report Sharpe ratio in the appendix on the V12.31 production-backtest where holding period and rebalancing are specified.
