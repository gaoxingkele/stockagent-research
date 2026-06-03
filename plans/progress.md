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
