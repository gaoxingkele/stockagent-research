# Walk-Forward Protocol (W1.1)

**Status**: 2026-05-30 finalized.
**Purpose**: validate H_E hybrid pipeline robustness on truly out-of-sample
quarters, replacing the single-split stratified PoC.

---

## 3 Walk-Forward Splits

Each split: 36-month train + 3-month val + 3-month test. Expanding-window
training (train end advances 3 months per split).

| Split | Train window | Val window | **Test window** |
|---|---|---|---|
| 1 | 2022-01-04 → 2024-12-31 | 2025-01 → 2025-03 | **2025-04 → 2025-06** |
| 2 | 2022-01-04 → 2025-03-31 | 2025-04 → 2025-06 | **2025-07 → 2025-09** |
| 3 | 2022-01-04 → 2025-06-30 | 2025-07 → 2025-09 | **2025-10 → 2025-12** |

(Note: D1 currently covers up to ~2026-05; we choose 3 splits within 2025 so
that all use FH labels with full 5-day forward returns available.)

## Sample Design (Random, Not Stratified)

Per split: **random sample of 2000 anchors from the test window**.

Critical difference from PoC:
- PoC used 25% high / 25% edge / 50% low stratified sampling
- Walk-forward uses **uniform random** — reflects real onset rate ~8%
- This makes evaluation realistic (most anchors won't be onsets)

## LGBM Training (3 models)

Each split gets its own LGBM trained on the train window above.
- Same hyperparams as `configs/model/lgbm.yaml`
- Forward-label blacklist preserved (r5/r10/.../dd40 excluded)
- ST filter active at data layer
- Save to `results/wf_lgbm_split{1,2,3}/`

## LLM Conditions (per anchor)

For each of 6000 anchors:
1. `raw` — minimal system prompt, no V12.31 expert knowledge
2. `expert` — V12.31 expert prompt body in user-prefix

Total LLM calls: 6000 × 2 = **12,000 calls**
Estimated cost: 12000 × $0.0023 (Sonnet) = **$27.60**
Estimated time: 12000 / (8 workers × 0.74 calls/s) = **2027 sec ≈ 34 min/condition × 2 ≈ 68 min** wallclock

## Evaluation Metrics

For each (method, split) combination:
- **RankIC** (cross-sectional Spearman, per date in test window)
- **Information Ratio** (RankIC mean / RankIC std)
- **Top 10% return** (mean fwd_r5 of top-decile by signal)
- **Top 20% return**
- **Top 10% winrate** (fraction with positive fwd_r5)
- **Top K Sharpe** (annualized; per-date TopK return / per-date std × sqrt(252))

For aggregation across splits:
- Mean ± std across 3 splits
- **Bootstrap 1000-resample 95% CI** on each metric
- **DM test** (Diebold-Mariano) for H_E vs LGBM on monthly returns

## Decision Gate (W1.5)

H_E PASS criteria:
1. Top10% return: mean H_E > LGBM × 1.05 across 3 splits **AND**
   bootstrap 95% CI for (H_E - LGBM) > 0
2. RankIC: H_E ≥ LGBM (within 1 std)
3. Robustness: H_E > LGBM in at least 2 of 3 splits

If 1+2 PASS: → W2 cross-market with confidence
If only 1 PASS: → CIKM rather than KDD A
If neither PASS: → fallback to "comparative study" paper (CCF B)

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Random sample reduces evidence (n=2000 has fewer onsets) | We still have 2000 × 3 = 6000 total; bootstrap CI shows real spread |
| 3 splits too few for tight CI | Add 2 more splits if time permits; report as sensitivity |
| LLM cost runs over $30 | Cap at 1500 anchors per split if needed |
| Walk-forward shows degradation > 50% | Honest reporting; pivot to "stratum-conditioned" framing |
