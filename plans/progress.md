# Progress — onset (启动子) algorithm implementation

Cross-iteration memory. Each Ralph iteration appends what it did, learned, and what's next.

## Goal
Regime-invariant neural intensity model of stock-movement onset, with PU/weak-supervised labels and leakage-safe, cluster-robust evaluation. See `plans/prd.json`. Contribution = formalization + reference method + benchmark, not alpha.

## Status
- Backlog seeded with 8 tasks (T-001 … T-008). None implemented yet.
- Verify gate: `.venv-xpu\Scripts\python.exe -m pytest tests/algo -q` (sentinel green).
- Branch: onset/algo-impl (create on first task).

## Dependency order
T-001 (PU labels), T-002 (weak supervision), T-003 (intensity head), T-007 (objectives) are independent → do first.
T-005 (evaluator) reuses C3 bootstrap. T-004 (IRM) independent. T-006 (cutoff probe) reuses eval_e3.
T-008 (integration) depends on T-001..T-005 + T-007 → do LAST.

## Log
- (seed) Backlog + guardrails created. Repo already provides: TCN encoder (src/models/tcn_cross_attn.py), expert rules (src/onset/expert_pattern.py), clustered bootstrap (src/evaluation/c3_dose_response.py), leakage plumbing (src/eval_e3/). Reuse them.
- **T-001 DONE** (branch onset/algo-impl): `src/onset/pu_labels.py` — `build_pu_sets` (Positive/Unlabeled partition), `class_prior` (labeled-positive-rate floor), `nnpu_risk` (non-negative PU risk, Kiryo et al. 2017, sigmoid surrogate, with the max(0,·) correction). Pure numpy, no torch. 5 hermetic tests green (`tests/algo/test_pu_labels.py`). Gate `pytest tests/algo` = 6 passed.
  - Note: PU prior π is NOT the labeled-positive rate (that's a lower bound) — `class_prior` returns the floor; downstream tasks should pass a better π estimate.
  - **Next:** T-002 (weak-supervision label model over expert rules) or T-003/T-007 (independent). T-008 integration last.
- **T-002 DONE**: `src/onset/weak_supervision.py` — `majority_vote` + `label_model` (EM-like: estimate per-LF accuracy via agreement with current hard label, weight by log-odds, recombine to soft labels; abstain→0.5). Snorkel-style aggregation of expert rules as labeling functions. numpy. 4 hermetic tests green.
  - **Next:** T-003 (neural intensity head), T-007 (objectives) — both torch/CPU, independent. Then T-004, T-005, T-006; T-008 last.
- **T-003 DONE**: `src/models/onset_intensity.py` — `OnsetIntensityHead` ([B,T,D]→softplus λ[B,T]) + `discrete_time_nll` (Bernoulli hazard p=1-exp(-λ)). Decoupled from encoder for hermetic testing. 3 CPU tests green (torch import ~35s one-time).
  - **Next:** T-007, T-004, T-005, T-006; T-008 integration last.
- **T-007 DONE**: `src/training/objectives.py` — `soft_rank` (pairwise-sigmoid), `soft_rank_ic_loss` (=-soft-Spearman IC), `topk_utility_loss` (=-softmax-weighted top-k return). Differentiable, replaces CE. 3 CPU tests green.
  - **Next:** T-004 (IRM), T-005 (evaluator), T-006 (cutoff probe); T-008 integration last.
- **T-004 DONE**: `src/training/irm_onset.py` — `irm_penalty` (IRMv1 dummy-scale grad penalty) + `train_step` (mean-env ERM + λ·mean-env IRM penalty). Environments = walk-forward splits/quarters. Structural tests only (SIGN-008). 2 CPU tests green.
  - **Next:** T-005 (evaluator), T-006 (cutoff probe); T-008 integration last.
- **T-005 DONE**: `src/evaluation/onset_eval.py` — `clustered_bootstrap` + `naive_bootstrap` + `point_in_time_guard`. Unifies C3 (date-clustered CI) + C5 (leakage guard). 3 tests green.
- **T-006 DONE**: `src/eval_e3/cutoff_probe.py` — `split_by_cutoff` + `leakage_flag` (pre-high & post-chance => leakage). Upgrades C5 toward causal. 4 tests green.
  - **Next:** T-008 integration (depends on T-001..T-005, T-007) — LAST task.
- **T-008 DONE**: `src/train_onset.py` — end-to-end reference method wiring T-001/T-002 (labels) + T-003 (intensity head/NLL) + T-004 (IRM) + T-007 (ranking) + T-005 (cluster-robust eval). Tiny epoch on synthetic walk-forward data -> results/onset/<run>/stats.json. Smoke <4s.

## COMPLETE
All 8 backlog tasks pass; full gate `pytest tests/algo` = 26 passed. Branch onset/algo-impl. The reference onset (启动子) pipeline is implemented end-to-end with hermetic tests. Next (human/research): run train_onset on the REAL walk-forward data, run the cutoff-probe on real LLM scores to make C5 causal, and swap TinyEncoder -> TCN encoder.

## NEW LINE (onset/identify) — leakage-free identification of LLM contribution
Backlog reset to plans/prd.json (old onset-algo backlog archived -> plans/prd_onset_done.json, all 8 done).
Goal: first IDENTIFIED estimate of LLM reasoning contribution on leakage-free A-shares + LLM-as-weak-supervisor distillation + leakage-calibrated de-biasing of FinBen. Branch onset/identify.
Dependency order: ID1->ID2->ID3 ; WS1->WS2 ; DB1->DB2 ; SYN last. CF is skip (needs paid LLM).
Key reusable $0 data: results/poc_full (A-share LLM scores+_fwd_r5), results/e3_* (FinBen+A-share no-context), src/evaluation/onset_eval.py, src/onset/*, src/train_onset_real.py.
Gate: pytest tests/algo. SIGN-R1: real experiments gate on machinery, null findings are valid.
- **ID1 DONE** (branch onset/identify): `src/identify/leakage_validity.py` — holds iff no-context CI lower bound <= chance. Verified on REAL data: A-share holds=True (acc .486, margin -.014), ACL18 holds=False (acc .733, margin +.233). 4 hermetic tests green. This is the identification precondition ID3 depends on.
  - **Next:** ID2 (contribution estimator), then ID3 (real A-share identification). WS1/DB1 independent.
- **ID2 DONE**: `src/identify/contribution.py` — partial rank corr (LLM vs target | baseline) + clustered-CI bootstrap. 3 hermetic tests green.
- **ID3 DONE + REAL**: `src/identify/run_ashare_identify.py`. FINDING (leakage-free A-shares, n=1000, 212 dates, identification HOLDS): identified LLM contribution over LGBM = raw **+0.033** clustered CI [-0.037,+0.110], expert **+0.006** CI [-0.062,+0.078] -> **both clean nulls (span 0)**. The first IDENTIFIED estimate of LLM reasoning value-add; ~0, consistent with the weak-signal thesis. stats.json committed under results/identify/ashare. smoke green.
  - **Next:** WS1->WS2 (distillation), DB1->DB2 (de-bias). SYN last.
- **WS1 DONE**: `src/identify/llm_lf.py` — LLM signals -> labeling functions feeding weak_supervision. 3 tests green.
- **DB1 DONE**: `src/identify/debias.py` — recall-corrected accuracy = us_full-(us_nocontext-chance), calibrated on clean market. 3 tests green.
- **DB2 DONE + REAL**: `src/identify/run_debias_finben.py`. FINDING: after removing the memorization excess, FinBen reasoning-only accuracy = ACL18 **0.440**, BigData22 **0.388**, CIKM18 **0.485** -> all at/below chance; clean-market reasoning ref only +0.02. The headline 60-80% LLM "skill" is essentially all memorization. results/identify/debias/finben_corrected.json committed.
  - **Next:** WS2 (distillation, real XPU), then SYN.
- **WS2 DONE + REAL**: `src/identify/run_distill.py` (LightGBM downstream, two arms). FINDING (held-out split3 clustered RankIC): ArmA true-labels **+0.109** [+0.037,+0.184]; ArmB +LLM-weak-refined **-0.040** [-0.114,+0.028]; identified improvement **-0.149** -> LLM-as-weak-supervisor HURTS (79.8% labels overwritten). Clean attribution under leakage-freeness; consistent with ID3 (~0 contribution). smoke green.
  - **Next:** SYN (synthesis: ID3+WS2+DB2 -> table+figure). LAST.
- **SYN DONE + REAL**: `src/identify/summarize_identify.py` -> results/identify/summary.md + paper/sections/figures/identify_summary.png. WS2 improvement CI [-0.298,-0.009] excludes 0 (LLM weak supervision significantly HURTS).

## COMPLETE (onset/identify)
All 8 tasks pass (CF skipped, needs paid LLM). Full gate pytest tests/algo = 43 passed.
HEADLINE FINDINGS (all on leakage-free A-shares, identification HOLDS):
  - ID3: identified LLM contribution over LGBM ~ 0 (raw +0.033 [-0.037,+0.110], expert +0.006 [-0.062,+0.078]).
  - WS2: LLM-as-weak-supervisor HURTS (-0.149 [-0.298,-0.009]).
  - DB2: de-biased FinBen collapses to ~chance (ACL 0.440 / BigData 0.388 / CIKM 0.485); ~all headline US "skill" = memorization.
Net: the first IDENTIFIED measurement says LLM reasoning adds ~nothing to A-share onset prediction, and the field s positive numbers are memorization.

## NEW LINE (research/leakage-frontier) — leakage frontier + de-biasing methods
Backlog reset to plans/prd.json (identify backlog archived -> plans/prd_identify_done.json, all 8 done).
Goal: (1) leakage-freeness is a (model x market x period) property — China-trained LLMs may have memorized A-shares, flipping the validity test; (2) de-biasing estimator as standalone methods contribution. Branch research/leakage-frontier.
Tasks: LF1 multi-model probe harness -> LF2 frontier aggregator; ID-SCALE ($0 full-anchor x horizon); DBX1 de-bias sensitivity -> DBX2 ($0 benchmark table); SYN2 synthesis. LF3 (paid multi-model real) skip:true until novelty + cost approved.
PRECONDITION: novelty-check (a) model-relative LLM-finance memorization (b) cross-market de-biasing BEFORE paid work.

## SHELVED: leakage-frontier line (novelty check failed)
LF (model-relative leakage) NOT novel — covered by DeepFund (per-model cutoff eval), MemGuard-Alpha (cross-model disagreement), general model-relative contamination lit. DBX (cross-market de-biasing) thin/uncertain — decontamination-correction genre crowded (ITD, DeconIEP reference-model, cross-lingual); cross-MARKET anchor adjacent but not clearly novel. Shelved -> plans/prd_leakage_frontier_shelved.json. No active prd until next direction decided. Paper to be retuned as case-study/experience+negative-results (pending).

## NEW LINE (research/market-neutral-alpha) -- beta-timing vs alpha-selection identification
Goal: target market/sector-NEUTRAL returns (idiosyncratic = where alpha lives), decompose into beta-timing (Macro agent) vs alpha-selection (Pattern Core), identify LLM contribution to EACH on leakage-free A-shares. $0 (reuse poc_full/poc_wf scores; neutral targets from _fwd_r5+trade_date+industry). Method not novel; contribution = leakage-free timing-vs-selection identification + honest tradable-alpha measurement.
Tasks: NB1 neutral targets -> NB2 decomposition identifier -> NB3 REAL$0 (selection/timing + long-short Sharpe); NB4 contrastive encoder -> NB5 REAL XPU (contrastive vs raw, market-neutral eval); NB6 synthesis. No paid tasks.
SIGN-A1: report tradable long-short, shrinkage != failure.
- **NB1 DONE** (branch research/market-neutral-alpha): `src/identify/neutral_targets.py` -- market/sector neutral residuals (per-date / per-date-industry demean) + systematic components. 3 hermetic tests green.
- **NB2 DONE**: `src/identify/decompose_identify.py` -- selection_contribution (idiosyncratic, reuses ID2) + timing_contribution (date-level LLM aggregate vs systematic move, block bootstrap). 2 hermetic tests green.
- **NB3 DONE+REAL**: `src/identify/run_neutral_identify.py`. FINDING (A-share, identification holds, n=1000): (a) idiosyncratic SELECTION contribution of LLM over LGBM = null even market-neutral (raw -0.004 [-0.064,+0.054], expert -0.031 [-0.089,+0.023]); (b) TIMING weak-positive but NOT significant (raw +0.085 [-0.055,+0.221]) -> hint that LLM may help beta-timing > alpha-selection, unconfirmed; (c) baseline market-neutral long-short annualized Sharpe 0.59 (mean CI [-0.011,+0.028] spans 0). Net: no significant idiosyncratic alpha after neutralization. stats committed.
  - **Next:** NB4 contrastive encoder -> NB5 contrastive vs raw (XPU); NB6 synthesis.
- **NB5 DONE+REAL** (XPU 109s): `src/identify/run_contrastive.py` contrastive vs raw, two arms. FINDING: market-neutral RankIC null both (raw +0.016 [-0.016,+0.054], neutral -0.001 [-0.043,+0.049]); long-short Sharpe raw 2.18 / neutral 1.71 with mean CI excluding 0 -- but **single split3 (known-favorable quarter), RankIC null -> NOT alpha evidence**; multi-split/cost/seed validation required. NEUTRAL did not beat RAW (shrinkage, expected per SIGN-A1). stats committed.
  - **Next:** NB6 synthesis (LAST).
- **NB6 DONE+REAL**: `src/identify/summarize_neutral.py` -> results/identify/neutral_summary.md + figure. full gate 54 passed.

## COMPLETE (research/market-neutral-alpha)
All 6 NB tasks pass. HEADLINE: even market/sector-neutral, the identified LLM idiosyncratic SELECTION contribution is null; TIMING shows a weak non-significant positive hint (LLM may aid beta-timing > alpha-selection, unconfirmed); contrastive/neutral target did NOT beat raw (shrinkage as expected). A long-short Sharpe ~1.7-2.2 appears on the single split3 window but RankIC is null and it is one favorable quarter -> NOT alpha evidence. Net: no significant idiosyncratic alpha; the only suggestive thread is LLM beta-timing, which needs multi-split confirmation.
- **NB4 DONE** (commit was deferred): `src/identify/contrastive_encoder.py` -- ContrastiveEncoder (stock + reference seq -> spread -> neutral-target head). 2 CPU tests green.

## NEW LINE (research/candlestick-onset) -- alpha1 hunt via candle geometry + relative position
User idea: recent 1-3 candlesticks RELATIVE to prior 3-9 bars = dynamic 3-12 bar onset. Scale/vol/regime-invariant (fixes non-stationarity); A-shares = inefficient market where candlestick edge can survive; matches V12.31 rules. Method NOT novel; contribution = HONEST multi-split, cost-aware, market-neutral test.
Tasks: K1 candle geometry + relative-position features -> K2 dynamic 3-12 bar assembler (flat + sequence); K3 REAL$0 LGBM all-split market-neutral cost-aware; K4 REAL XPU learned sequence (multi-seed); K5 ablation vs smoothed factors; K6 synthesis + alpha verdict.
SIGN-K1: alpha ONLY if pooled NET long-short Sharpe CI excludes 0 AND net Sharpe>0 in >=2/3 splits -- fixes the NB5 single-split mirage. Prior market-neutral line merged + archived to prd_market_neutral_done.json.
- **K1 DONE** (branch research/candlestick-onset): `src/onset/candle_geometry.py` -- per-bar geometry (body/wicks/close_loc/range_over_atr/gap) + relative-position (close_pct_prior/breakout/dist_low_atr/higher_lows/vol_ratio/compression), scale-invariant + point-in-time, panel wrapper. 3 hermetic tests green (finite, scale-invariance, breakout). Full gate 57 passed.
  - **Next:** K2 (3-12 bar assembler: flat + sequence) -> K3 LGBM all-split.
- **K2 DONE**: `src/onset/candle_pattern.py` -- anchor_features (flat 3-bar vector) + anchor_sequences (W-bar geometry sequence), reuse build_anchor_sequences. Point-in-time verified (future bars do not leak). 2 hermetic tests green.
- **K3 DONE+REAL**: `src/onset/run_candle_lgbm.py`. FINDING: candle-geometry features -> pooled market-neutral RankIC **+0.039 [+0.016,+0.065] (excludes 0)** = strongest cross-sectional signal in the project. BUT cost-aware tradable: pooled NET long-short Sharpe 0.54, net mean CI [-0.003,+0.009] spans 0; per-split net Sharpe +0.34/+1.52/-0.26 (split3 reverses). SIGN-K1: net>0 in 2/3 but pooled net CI spans 0 -> **NOT alpha1** (small real signal, not cost-surviving / regime-stable). lgbm.json committed.
  - **Next:** K4 learned sequence (XPU), K5 ablation, K6 verdict.
- **K4 DONE+REAL** (XPU 328s): `src/onset/run_candle_seq.py` learned candle-geometry sequence (GRU, 3-seed ensemble). FINDING: pooled market-neutral RankIC **+0.062 [+0.036,+0.089]** (excludes 0, stronger than K3 flat +0.039); pooled NET long-short Sharpe 0.63, net mean CI [-0.0022,+0.0099] spans 0 -> not cost-surviving. Consistent with K3.
  - **Next:** K5 ablation, K6 verdict (LAST).
- **K5 DONE+REAL** (caught+fixed a _fwd_r5 leak in factor_cols -> had given absurd RankIC 0.825). FINDING: candle geometry is COMPLEMENTARY to the 165 smoothed factors -- pooled net long-short Sharpe factors 0.81 -> factors+candle 1.76 (incremental **+0.95**), RankIC 0.061 -> 0.083. candle-alone net negative. Strongest positive signal in project, but pooled-only; per-split SIGN-K1 verdict in K6.
  - **Next:** K6 synthesis + alpha1 verdict (LAST).
- **K6 DONE+REAL**: `src/onset/summarize_candle.py` -> results/candle/summary.md + figure. full gate 67 passed.

## COMPLETE (research/candlestick-onset)
alpha1 VERDICT (SIGN-K1): NO confirmed tradable alpha. candle-only (K3/K4) = null after 0.4% cost (pooled net CI spans 0). BUT candle geometry is COMPLEMENTARY to smoothed factors: factors+candle pooled net long-short CI EXCLUDES 0 (Sharpe 1.76, RankIC 0.083), incremental +0.95 net Sharpe over factors -> labelled PROMISING-UNCONFIRMED (per-split net robustness not yet measured for that combo). Candle geometry also gives the strongest pooled market-neutral RankIC in the project (seq +0.062, factors+candle +0.083). DECISIVE NEXT: per-split net long-short of factors+candle (must be >0 in >=2/3 splits incl the C4-unstable split3) to clear the full SIGN-K1 bar.

## DECISIVE per-split check (Step 2) -- factors+candle PASSES SIGN-K1
factors+candle: pooled net long-short Sharpe 1.76, net mean CI [+0.0049,+0.0179] EXCLUDES 0; per-split net Sharpe split1 +3.22 / split2 +0.85 / split3 +1.20 -> POSITIVE in 3/3 splits. SIGN-K1 verdict: REAL (alpha1). Mechanistically coherent: candle geometry rescues split2 (factors-only -0.67 -> +0.85), the C4-unstable quarter -- the relative-position normalization fixing non-stationarity as hypothesized. First signal in the project to clear the honest bar.
BUT NOT YET DEPLOYABLE ALPHA -- binding caveats: (1) A-shares largely CANNOT be shorted, so the long-short is not directly tradable; the realistic test is LONG-ONLY top portfolio minus market. (2) cost model is a simple 0.4
## NEW LINE (research/deployability) -- alpha1 deployability gauntlet
Goal: force the SIGN-K1-passing factors+candle signal through real A-share constraints. (1) LONG-ONLY top-K market-excess (no shorting); (2) realistic cost model (asymmetric stamp duty / commission / slippage / T+1 / limit-up); (3) cross-period (2023/24 windows) + liquidity + top-K robustness + capacity. HONEST: long-only likely weaker -> genuine go/no-go. $0.
Tasks: LO1 long-only metric -> COST1 A-share cost model -> LO2 REAL factors+candle long-only net-excess go/no-go; ROB1 REAL cross-period+liquidity+topK; CAP1 capacity; DSYN verdict. New SIGN-D1.
Candlestick line merged to main + archived to prd_candlestick_done.json.
- **LO1 DONE** (branch research/deployability): `src/onset/long_only.py` -- long-only top-K market-excess (top-K minus equal-weight market per date) + summarize_excess (annualized Sharpe + block CI, cost-aware). 3 hermetic tests green.
- **COST1 DONE**: `src/onset/ashare_cost.py` -- realistic A-share round-trip cost ~0.2% (stamp on sell only) + enterable (excludes limit-up entries) + net_excess. 3 hermetic tests green.
- **LO2 DONE+REAL** (decisive go/no-go): factors+candle LONG-ONLY survived removing the short leg + realistic cost. pooled net market-excess annualized Sharpe 1.10, mean +0.71%/period, CI [+0.0011,+0.0138] EXCLUDES 0; per-split net mean +1.17/+0.52/+0.45% -> positive 3/3 (only split1 individually significant, per-split CIs wide). PASS go on the long-only gate. FULL deployable verdict still pending ROB1 (cross-period 2023/24). long_only.json committed.
  - Next: ROB1 cross-period + liquidity + topK, CAP1, DSYN.
- **ROB1 DONE+REAL** (cross-period KILL SHOT): the long-only factors+candle edge that looked good on 2025 (LO2 Sharpe 1.10) is NEGATIVE in 2023 (net Sharpe -0.37) AND 2024 (-0.36); liquid-only -0.32; top-K {5,10,20%} all negative. holds_across_years=FALSE -> the 2025 result is period-specific, NOT a stable deployable edge. SIGN-D1 cross-period gate FAILED. Discipline caught what a single-year backtest would have missed.
  - Next: CAP1 (moot but completes backlog), DSYN final verdict (NOT deployable / collapsed cross-period).
- **CAP1 DONE**: src/onset/capacity.py -- ADV-based capacity estimate (participation*sum ADV). 2 hermetic tests green. (Moot given ROB1 non-deployable, but completes the backlog.)
- **DSYN DONE+REAL**: src/onset/summarize_deploy.py -> results/deploy/summary.md + figure. full gate 81 passed.

## COMPLETE (research/deployability)
FINAL DEPLOYABILITY VERDICT: STATISTICALLY-REAL-ON-2025-BUT-NOT-DEPLOYABLE. The factors+candle signal passed the in-sample 2025 SIGN-K1 long-short bar AND the 2025 long-only cost gate (LO2 net Sharpe 1.10, 3/3 splits positive), but COLLAPSES cross-period: long-only net Sharpe 2023 -0.37, 2024 -0.36 (ROB1) -> period-specific, not a stable tradable edge. SIGN-D1 not met. The discipline caught what a 2025-only backtest would have missed; deploying on the 2025 result would have lost money in 2023/24.
