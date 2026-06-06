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

## NEW LINE (research/production-edge) -- where V12.31 actually makes money
Insight (from Appendix A): V12.31's edge is NOT the broad onset signal but (1) disaster-month TIMING, (2) EXTREME FILTERING to a tiny high-conviction long-only pool (V7c iron rules), (3) ASYMMETRIC downside avoidance. We tested the wrong object all along.
Tasks: TIM1 timing overlay + Sharpe decomposition -> TIM2 REAL disaster-month timing test; FILT1 extreme-filter pool builder -> FILT2 REAL filtered-pool edge; COMBO1 REAL production-faithful (timing x filter x asymmetry, SIGN-D1); OPT1 REAL walk-forward knob optimization; PSYN verdict + decomposition. All $0 (D1 + disaster_filter + expert_pattern). New SIGN-P1.
Deployability line merged to main + archived to prd_deployability_done.json.
- **TIM1 DONE** (branch research/production-edge): `src/onset/timing_overlay.py` -- timed_series + decompose_sharpe (buy-hold vs timed vs timed+selected; incremental timing/selection Sharpe). 3 hermetic tests green.
- **TIM2 DONE+REAL**:  two regime arms. FINDING: timing edge NOT reproducible with available signals. disaster_filter reproduction is BROKEN (fires 0.5%, 0 in 2022 bear; concept signals C2/C3 are unimplemented TODOs). Standard trend regime = net wash (pooled +0.08 Sharpe, CI spans 0), regime-dependent (helps 2022/2024, hurts 2023/2025). The 'Sharpe is mostly timing' hypothesis is NOT supported here; the production timing edge needs the full disaster composite. timing.json committed. results/production un-ignored.
  - Next: FILT1 extreme-filter pool -> FILT2; COMBO1; OPT1; PSYN.
- **FILT1 DONE**: src/onset/extreme_filter.py -- extreme-filter high-conviction pool (onset + not-overheated + not-zombie + not-worst-industry + top-pct). 3 hermetic tests green.
- **FILT2 DONE+REAL**: reproduced onset+filter selection pool UNDERPERFORMS the market: long-only excess Sharpe 2023 -2.62 / 2024 -2.14 / 2025 -2.66, pooled -1.83 (CI negative, excludes 0); 0/3 positive years. Selection component as reproduced is a NEGATIVE edge. Stricter settings swept in OPT1.
  - Next: COMBO1, OPT1, PSYN.
- **COMBO1 DONE+REAL** (artifact caught): Sharpe decomposition shows timed/combo Sharpe >> buy-hold, but it is a Sharpe-INFLATION artifact (cash during volatile down periods cuts vol) + small-sample noise (40-48 pts, 2025 timed 5.78), CONTRADICTING robust TIM2 (timing wash) + FILT2 (selection negative). On the robust mean basis: no reliable edge. PSYN uses the mean criterion, not Sharpe.
  - Next: OPT1 (walk-forward knob sweep), PSYN (verdict).
- **OPT1 DONE+REAL**: walk-forward knob optimization -> NO config recovers an edge. WF-selected OOS test Sharpe -2.14 / -2.66; best train config still negative. Selection pool robustly NEGATIVE across all knobs, in-sample AND OOS.
  - Next: PSYN final verdict.
- **PSYN DONE+REAL**: production-edge verdict -> NOT REPRODUCIBLE. timing wash (+0.08), selection negative (-1.83, 0/3 yrs), WF-opt OOS negative, disaster filter broken (0.47%). full gate 94.

## COMPLETE (research/production-edge)
FINAL: the DOCUMENTED V12.31 rules (onset + V7c filters + simplified disaster timing) do NOT reproduce the production Sharpe 2.20. Timing is a wash; selection is robustly NEGATIVE (all years, all knobs, OOS); the reproduced disaster filter barely fires (concept signals C2/C3 unimplemented). The real edge lives in parts NOT in the released knowledge: the FULL disaster composite, the actual r20_pred predictive model, execution/risk/discretion, and production parameter calibration. The released expert prompt is INSUFFICIENT to reproduce the alpha.

## NEW LINE (research/onset-motif) -- information-theoretic go/no-go for the onset-motif upgrade
Core: we proved MARGINAL I(pattern; fwd_r) ~ 0. The promoter-needs-transcription-factor hypothesis = the info is CONDITIONAL: I(pattern; fwd_r | regime) > 0. This line MEASURES conditional vs marginal MI with a within-regime PERMUTATION null (MI is positively biased -> raw MI is not evidence), and tests cross-period stability. Conditional MI > null & stable -> build the graph/point-process onset-motif model; else the onset concept is information-theoretically exhausted.
Tasks: MI1 estimators+permutation null -> REG1 regime states; MI2 REAL marginal-vs-conditional MI probe (decisive); MI3 REAL cross-period MI stability; MSYN go/no-go verdict. All $0 (pure measurement, no model). New SIGN-M1.
Production-edge line merged to main + archived to prd_production_edge_done.json.

## 2026-06-05 MI1 done -- MI estimators + permutation null
src/onset/mutual_info.py: quantile_bins (equal-freq, tie-robust), mutual_info (marginal, binned, biased), conditional_mi (sum_z P(z)*MI within stratum), perm_pvalue (shuffle Y within Z-strata for conditional / globally for marginal; p=(1+#null>=obs)/(1+n)). SIGN-M1 enforced: significance only from permutation null, never raw MI.
Hermetic test: XOR-style constant-magnitude signs so NO magnitude leaks marginally -> conditional MI >> marginal (~0), conditional perm p<0.02 while marginal p>0.10; pure-noise stays non-significant. 5/5 green, full gate 99 passed. Next: REG1 regime builder.

## 2026-06-05 REG1 done -- point-in-time regime states
src/onset/regimes.py: trend_state (sign of shifted trailing market return), vol_state (tercile of shifted trailing market vol), disaster_state (disaster_filter), regime_states(df)->per-date trend/vol/disaster, map_states_to_rows broadcasts onto anchor rows. All point-in-time (shift). Bug fixed: NaN>0 yields False not NaN so fillna(True) never fired -> used .where(trailing.notna(),True). 3/3 green. Next: MI2 REAL probe (decisive).

## 2026-06-05 MI2 done -- REAL marginal-vs-conditional MI probe (DECISIVE, positive)
src/onset/run_mi_probe.py on 200k-row subsample of D1 (13 candle feats + onset_score x {_fwd_r5, market-neutral residual} x {trend,vol,disaster}).
KEY METHOD FIX (skepticism): the naive within-stratum permutation only tests I(X;Y|Z)>0, which is TRIVIALLY true when marginal>0 -- and at n=2e5 BOTH marginal and conditional p saturate at the 1/n_perm floor. Added interaction_pvalue (mutual_info.py): permutes the Z LABELS (keeps X,Y pairs) to test whether the REAL regime adds more than a random regime of equal granularity -> interaction info II = I(X;Y|Z)-I(X;Y). This is the correct motif null.
FINDING: REGIME-ADDS-INFORMATION. 52/78 (feat x target x regime) combos have interaction>0 with Z-perm p at floor. The TREND regime dominates: for the top features (close_pct_prior, dist_low_atr, onset_score, body, close_loc) the interaction is ~70-76% of the conditional MI -- i.e. ~3/4 of these candle features' (small) forward-return information appears ONLY once you condition on the trend regime. This is the first AFFIRMATIVE support for the conditional-information ('promoter needs transcription-factor') hypothesis.
CAVEATS (do not overclaim): absolute MIs are tiny (cond_mi~0.006 nats -> corr~0.11); p is floor-saturated by n so it proves >0 not 'large'; pooled across 2022-2025. MI is statistical dependence, NOT net-of-cost tradable return (does not contradict the deployability-line null). DECISIVE next test = MI3 cross-period stability: if the interaction only lives in one year it is not a modeling basis.

## 2026-06-05 MI3 done -- cross-period stability of the interaction info (STABLE)
src/onset/run_mi_stability.py: per-year (2022/23/24/25) interaction_pvalue for the top-6 (feature,regime) hits from MI2, full data (capped 150k rows/year). Optimized _mi_from_codes (np.bincount instead of np.add.at -> 24x faster).
FINDING: STABLE-CONDITIONAL-INFORMATION. ALL 6 items pos_sig=4/4 -- the interaction is positive AND Z-perm-significant in EVERY year. The TREND regime adds info about candle-feature -> fwd_r5 consistently (close_pct_prior +0.0023..+0.0103 nats across years; the VOL regime for close_pct_prior is flat ~+0.004-0.005 every year). The 'promoter needs transcription-factor' hypothesis SURVIVES the non-stationarity test that killed the return edge.
DISCIPLINE: p floors at 1/n_perm each year (n~150k) so p proves >0 not large -- effect SIZE (cond_mi~0.006 nats, corr~0.07-0.14) is the honest metric. INFORMATION != net-of-cost RETURN (deployability/production lines already showed the return edge collapses after cost/long-only); MI is sign-blind so part of this may be downside/risk info. Next: MSYN go/no-go synthesis.

## 2026-06-05 MSYN done -- onset-motif go/no-go VERDICT: BUILD-THE-MOTIF-MODEL
src/onset/summarize_motif.py -> results/motif/{summary.md,verdict.json} + paper/sections/figures/motif_mi_summary.png. motif_verdict(hits,n_stable,n_tested): BUILD only if regime adds info (MI2) AND it is cross-period stable (MI3).
VERDICT: BUILD-THE-MOTIF-MODEL (hits=52/78, stable=6/6). The trend/vol regime is a genuine 'transcription factor' that activates the candle 'promoter' -- conditional information is permutation-significant AND positive+significant every year 2022-2025.
This is the onset-motif line's payoff: the FIRST cross-period-robust positive result in the program. It UPGRADES the 'promoter' to a context-conditional 'onset motif' at the information-theoretic level -- a defensible, novel contribution (C6 candidate).
4 honesty caveats baked into the verdict (information != alpha): p is n-saturated (proves >0 not large); information != net-of-cost return (deployability/production nulls stand); MI is sign-blind (may be downside/risk info); converting stable conditional info -> monotone costable long-only signal is the unsolved binding constraint. ONSET-MOTIF BACKLOG COMPLETE.

## NEW LINE (research/motif-tradability) -- is the stable conditional information TRADABLE or information-only?
onset-motif merged to main (BUILD-THE-MOTIF-MODEL on info-theoretic grounds). But MI is SIGN-BLIND -- it can't tell directional (tradable long) from variance/risk (not). This line decomposes the information by CONDITIONAL MEAN + monotonicity (not MI) and runs the SIMPLEST regime-gated, long-only, cost-aware, cross-sectional, per-year backtest. If even the simplest gate can't make net-of-cost money cross-period, do NOT build the complex motif model.
Tasks (all $0): SGN1 tradability decomposer (directionality+monotonicity+var-vs-mean) -> RGT1 regime-gated long-only signal -> TRD1 REAL directional-vs-sign-blind diagnosis -> TRD2 REAL regime-gated net-of-cost backtest (decisive) -> TSYN TRADABLE vs INFORMATION-ONLY verdict. New SIGN-T1. onset-motif archived to prd_onset_motif_done.json.

## 2026-06-06 SGN1 done -- tradability decomposer (what MI can't see)
src/onset/tradability.py: directionality (per-regime Spearman slope of feature vs return + sign-hit-rate), monotonicity (conditional mean across feature quantile buckets + mono_coef = rank corr bucket-index vs bucket-mean), variance_vs_mean (directional_fraction = mean-dispersion / (mean+var dispersion)). All conditional-MEAN based, never MI (SIGN-T1).
Hermetic test proves the point: a SIGN-BLIND synthetic (Var[y|x] depends on x, E[y|x]=0) has mutual_info>0.01 but directionality slope ~0 and mono_coef<0.5 -- exactly the case MI cannot distinguish from tradable. Directional synthetic scores slope>0.15, mono_coef>0.7. Next: RGT1 regime-gated signal.

## 2026-06-06 RGT1 done -- simplest regime-gated long-only signal
src/onset/regime_gate.py: regime_gated_excess (per-date long-only top-K by feature via long_only_excess, ZERO on out-of-regime dates, net of round-trip A-share cost), gated_vs_ungated (isolates the regime's marginal contribution). Reuses long_only + ashare_cost (no reimplementation). Point-in-time. 3/3 green. Next: TRD1 REAL directional-vs-sign-blind diagnosis.

## 2026-06-06 TRD1 done -- directional-vs-sign-blind diagnosis (MIXED, leaning positive)
src/onset/run_tradability.py applies SGN1 to the 6 MI3 stable winners. FINDING (VERDICT: SOME-DIRECTIONAL):
- best_mono = 1.000 for ALL top features -- the CONDITIONAL MEAN of fwd_r5 is PERFECTLY MONOTONE across feature quantile buckets within the trend regime. This is the key positive: it is NOT pure sign-blind risk; there is a real, monotone directional core.
- best_state Spearman slope ~0.08-0.10 (appreciable for cross-sectional selection).
- BUT directional_fraction ~0.37-0.43: only ~40% of the feature's predictive DISPERSION is in the mean; ~60% is in the variance (sign-blind risk). close_pct_prior(0.43) and close_loc(0.40) clear the 0.4 tradable bar; dist_low_atr/onset_score/body(0.37-0.39) just miss.
READING: the stable conditional info is a MIX -- a genuine monotone directional component plus a larger risk component. MI alone could not have told these apart. DECISIVE next test = TRD2: does the monotone directional core survive as net-of-cost, long-only, cross-period return? (prior lines warn A-share cost ~0.2% + non-stationarity usually eats slopes this size).

## 2026-06-06 TRD2 done -- regime-gated net-of-cost backtest (DECISIVE: information-only)
src/onset/run_regime_gate_bt.py: simplest regime-gated (trend-up) long-only top-10% by feature, NON-OVERLAPPING 5-day periods, net of A-share round-trip ~0.2%, date-clustered CI, per-year. VERDICT: EATEN-BY-COST-OR-NONSTATIONARY (information-only). survivors=[].
- close_pct_prior gated Sharpe=-1.18 (CI all<0); close_loc=-2.35; onset_score=-1.39. pos_years=0/4 for all. regime_frac=0.55.
- KEY: gating IMPROVES every feature (gated > ungated: -1.18 vs -1.89, -2.35 vs -2.71, -1.39 vs -2.04) -- the trend regime contribution is REAL and correctly-signed, consistent with the MI + monotonicity findings. But the gross directional edge (~0.1%/5d, slope 0.09 monotone) sits BELOW the ~0.2% round-trip cost floor, so net is negative every year.
FULL CHAIN COMPLETE & CONSISTENT: MI says regime adds stable info -> TRD1 says it's a monotone directional core (~40%) + sign-blind risk (~60%) -> TRD2 says the directional core is too SMALL to beat A-share cost. The onset-motif is REAL information with a genuine directional component, but NOT net-of-cost long-tradable. Binding constraint = transaction cost, exactly as the deployability/production lines warned. Vindicates doing the cheap diagnostic BEFORE building the complex model. Next: TSYN final verdict.

## 2026-06-06 TSYN done -- onset-motif tradability VERDICT: INFORMATION-ONLY (eaten-by-cost)
src/onset/summarize_tradability.py -> results/motif/{tradability_summary.md,tradability_verdict.json} + paper/sections/figures/motif_tradability.png.
FINAL VERDICT: INFORMATION-ONLY (eaten-by-cost). build_trading_motif_model=False. directional_monotone=True, gating_helps=True, net-of-cost survivors=0/3.
The honest chain, complete: (1) MI -- trend regime adds STABLE conditional information every year; (2) TRD1 -- genuine MONOTONE directional core (rank-corr ~0.09, mono_coef ~1.0) + ~60% sign-blind risk; (3) TRD2 -- gating helps (regime real) but gross edge ~0.1%/5d < ~0.2% cost floor -> net Sharpe<0 every year. The onset-motif is REAL, cross-period-stable information with a real directional component (defensible scientific/methodological finding) but NOT net-of-cost long-tradable. Binding constraint = transaction cost, consistent with deployability + production lines. Cheap diagnostic BEFORE modeling avoided building a complex model on an untradable edge.
MOTIF-TRADABILITY BACKLOG COMPLETE.

## NEW LINE (research/regime-baseline-bench) -- do 2025-26 SOTA baselines survive our honest protocol?
motif-tradability merged+archived. After the SOTA review, benchmark the closest baselines by REIMPLEMENTING their method on D1 + our protocol (NOT matching their US numbers; SIGN-B1).
EXP-A rolling-HMM regime LightGBM (MDPI 2026) vs our regime-gating vs plain LGBM; EXP-B When-Alpha-Breaks two-level abstention (2603.13252) on our onset ranker; EXP-C NMI+permutation info-theory (2511.16339 / 2601.00395) vs our interaction info. Method tasks: BENCH1 rolling-HMM, BENCH2 Deflated Sharpe, BENCH3 abstention. All report raw Sharpe + DSR + date-clustered CI net of A-share cost, per-year. BSYN synthesis. hmmlearn 0.3.3 installed. New SIGN-B1.

## 2026-06-07 BENCH1 done -- rolling-HMM regime detector
src/bench/hmm_regime.py: rolling_hmm_states (GaussianHMM, expanding past-window refit every k, Viterbi-last-of-X[:t+1] -> point-in-time; canonical state order by mean of feature 0 to stabilize labels across refits). Reimplements the MDPI 2026 rolling-HMM component. 2/2 green: recovers a 2-regime synthetic + no-future-leakage (perturbing the tail leaves earlier states identical). Next: BENCH2 Deflated Sharpe.

## 2026-06-07 BENCH2 done -- Deflated Sharpe Ratio
src/bench/deflated_sharpe.py: probabilistic_sharpe (PSR, skew/kurtosis/n-adjusted), expected_max_sharpe (SR0 = expected max of n_trials Sharpes, Bailey-LdP), deflated_sharpe (DSR = PSR vs SR0). Annualized-in -> per-period conversion via periods_per_year. 5/5 green: more trials -> higher SR0 -> lower DSR; strong single-trial Sharpe stays significant (dsr>0.95); marginal Sharpe at 50 trials deflates (psr>0.5 but dsr<0.5). Next: BENCH3 abstention.

## 2026-06-07 BENCH3 done -- two-level abstention gate
src/bench/abstention.py: regime_instability (trailing state-switch rate), abstain_mask (TRADE when uncertainty AND instability both below expanding PAST q-quantile; warmup -> trade). Reimplements When-Alpha-Breaks abstention, point-in-time. 4/4 green. Method tasks (BENCH1/2/3) done; next REAL: EXPA regime-LGBM head-to-head.

## 2026-06-07 EXPA done -- Regime-Aware LightGBM vs ours, on our protocol (ALL SUB-COST)
src/bench/run_expa.py: one LGBM cross-sectional ranker, 3 gates (plain/HMM-favorable/trend-up), D1 walk-forward (test 2025Q2/Q3/Q4), non-overlap 5-day, A-share cost, date-clustered CI + Deflated Sharpe (n_trials=3).
RESULT (VERDICT: ALL SUB-COST / not DSR+CI significant): plain Sharpe +0.89 DSR .654 CI[-0.0016,+0.0036]; hmm +1.92 DSR .993 CI[0.0,+0.0025]; trend +1.45 DSR .821 CI[-0.0008,+0.0038].
KEY SKEPTICAL READ: the HMM arm's high Sharpe+DSR is a CASH-DURING-VOLATILITY Sharpe-INFLATION artifact (gating to cash cuts vol -> inflates Sharpe), NOT selection alpha -- its mean-excess CI lower bound is exactly 0.0, i.e. no robustly-positive mean edge. No arm clears strictly-positive net-of-cost mean CI. The MDPI Regime-Aware LightGBM method, reimplemented on A-shares under our honest protocol, is sub-cost just like our own onset signal. Strong paper point: regime gating buys Sharpe via timing/risk-reduction, not net-of-cost alpha. Next: EXPB abstention.

## 2026-06-07 EXPB done -- abstention does NOT rescue the onset edge
src/bench/run_expb.py: onset_score ranker, 3 gates (always/trend/abstain), full nonoverlap 5-day grid, A-share cost, date-clustered CI + DSR, per-year. Abstain = trade only when model-uncertainty (neg cross-sectional score dispersion) AND regime-instability (trend-switch rate) both below expanding-past 0.8 quantiles.
RESULT (VERDICT: ABSTENTION DOES NOT RESCUE): always Sharpe -2.04 CI[-0.0086,-0.0031] traded 1.00; trend -1.39 CI[-0.0046,-0.0008] traded 0.55; abstain -1.89 CI[-0.0075,-0.0024] traded 0.75. All net-negative, all CIs strictly <0, all DSR~0. When-Alpha-Breaks two-level abstention, reimplemented on our onset signal, does NOT turn the sub-cost edge positive -- it sits between always-on and trend-gate but stays firmly negative. Consistent with TRD2. Next: EXPC info-theory positioning.
