# Appendix A: V12.31 Expert Knowledge Prompt

This appendix reproduces the V12.31 expert knowledge prompt referenced in §4.2. The prompt was elicited through three rounds of semi-structured interviews with the V12.31 system designer; total length is approximately 5,500 Chinese characters (~1,800 tokens). In the agent experiments, this prompt is injected at the head of the user message (not the system message) to bypass observed API-gateway system-prompt truncation; see §4.4.

The full prompt is available in the supplementary code release at `prompts/v12_31_expert_v1.md`. Here we reproduce its structure with English glosses:

## A.1 Bullish Onset Detection Rules (Round 1)

**Three necessary conditions (all must hold):**

1. **Bottoms rising**: `low_5d ≥ low_20d × 0.98` (2% tolerance to avoid noise misclassification)
2. **Above 5-day low**: `close_t ≥ low_5d × 1.05` (5% confirmation gap above recent low)
3. **MA pattern**: `MA_5, MA_10, MA_20` spread < 3% AND MA_5 upward-tilting

**One bonus condition (additive, not required):**

4. **Volume boost**: `volume_5d_mean > volume_20d_mean × 1.2`

## A.2 Bearish Onset Asymmetry (Round 1)

Bearish onset is **not** a tradeable short signal. Detection of bearish onset triggers:
- Exclude the affected stock from the long-only candidate pool
- Do not construct any short position

This asymmetry reflects the A-share short-sale constraint and is justified by the asymmetric information-diffusion result (Atilgan et al. 2022).

## A.3 Sub-pattern Simplifications (Round 2)

The expert deliberately rejects fine-grained sub-pattern classification:
- No W-bottom / U-bottom / circular-bottom subtype distinction
- No market-cap or sector stratification (uniform thresholds)
- No breakout-vs-reversal subtype split

Rationale: complexity here is presumed to overfit historical patterns without improving forward generalization.

## A.4 Disaster Month Composite Signal (Round 3)

Disaster month detection uses a **vote-2/3** composite of three signal groups:

**Signal A — Index AND (`r_sh < -2%` AND `r_gem < -3%`)**

**Signal B — Volume OR (any of):**
- `B1`: 5-day mean total amount / 20-day mean total amount < 0.70
- `B2`: limit-down stock count > 100 OR limit-down/limit-up ratio > 3
- `B3`: up-stock fraction < 0.30 OR down-stock fraction > 0.70

**Signal C — Sector inner-vote ≥ 2/3:**
- `C1`: fraction of industries with negative mean return > 0.80
- `C2`: top-5 hot concepts all negative
- `C3`: top-5 hot concepts average return < −0.01

**Composite**: `(A + B + C) ≥ 2`

## A.5 Failure Modes Listed for the LLM

The prompt explicitly enumerates failure modes the LLM should not predict as onset:
- Disaster month active (any one of the above)
- Industry rank in worst 10% by 60-day momentum
- Trend-zombie stocks (MA_60 long horizontal)
- ST-flagged stocks (handled at data layer, not prompt)
- Already-sustained uptrend (past_r5 > +8% indicates trend continuation, not new onset)

## A.6 Decision Signal Encoding

The prompt instructs the LLM to output the decision signal as:

```
pump_ratio = P(bullish_onset) / (P(bearish_onset) + 0.01)
```

with strict JSON output format (three floats summing to ≈ 1, no rationale or markdown).

## A.7 V7c Five Iron Rules (Deployed Hard Filters)

For reference (these are applied in production V12.31 but not encoded as LLM-side conditions; they are part of the Alpha Factor Explorer post-processing):

1. r20_pred in top 5% of the day
2. pyr_velocity_20_60 below the 35th percentile (no premature momentum)
3. PCA components |f1_neg1| < 0.005 and |f2_pos1| < 0.005
4. NOT is_zombie (no horizontal MA_60)
5. industry 60d momentum rank ≥ 0.10 (exclude worst-10% industries)

These five rules apply as hard filters after LLM scoring, ensuring the Final Verifier produces deployment-realistic candidate pools.
