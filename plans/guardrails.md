# Guardrails (Signs) — stockagent-research onset (启动子) algorithm work

Learned constraints read at the start of every Ralph iteration. Progress persists; failures evaporate. Append-only.

### SIGN-001: The gate is RUNS-CORRECTLY, not BEATS-BASELINE
**Trigger:** deciding whether a task passes.
**Instruction:** A task passes when its component runs and produces structurally-valid output AND its hermetic test is green. NEVER gate on "RankIC > baseline" / "beats LGBM" / "accuracy up". The signal in this domain is weak (RankIC ~0.02–0.07); a performance gate would never converge.
**Reason:** The paper's contribution is formalization + reference method + benchmark, not alpha.

### SIGN-002: Tests must be hermetic
**Trigger:** writing a task's test.
**Instruction:** Use small synthetic data generated in-test (numpy/torch). No network, no large parquet, no XPU/GPU requirement, runs in <5s. The test must FAIL before the deliverable exists and PASS after.
**Reason:** The verify gate (`pytest tests/algo`) must be fast, deterministic, and meaningful.

### SIGN-003: No look-ahead / point-in-time safety
**Trigger:** any feature, label, or evaluation builder.
**Instruction:** Never use data at or after the target/prediction date. Assert point-in-time safety in tests. This is the whole lesson of contribution C5 (temporal leakage).
**Reason:** Leakage silently inflates everything and invalidates the result.

### SIGN-004: Cluster-robust evaluation only
**Trigger:** reporting any metric difference or CI.
**Instruction:** Use date-clustered bootstrap (resample whole trading days), never anchor-independent. Report CIs on absolute differences, not ratios. (Contribution C3.)
**Reason:** Same-day anchors are correlated; naive bootstrap manufactures false significance.

### SIGN-005: Use the .venv-xpu environment
**Trigger:** running anything.
**Instruction:** Always run via `.venv-xpu\Scripts\python.exe` (Python 3.11 + torch-xpu). Do NOT use the system Python 3.14 (no torch/IPEX wheels). Models auto-select device cuda>xpu>cpu; tests must run on CPU.
**Reason:** Only .venv-xpu has a working torch.

### SIGN-006: Small, focused, committed increments
**Trigger:** finishing a task.
**Instruction:** One task = one deliverable module + its test. Run the gate, then `git add` + commit with a message referencing the task id (e.g. `feat(onset): T-003 neural intensity head`). Branch: onset/algo-impl.
**Reason:** Progress must persist across fresh iterations.

### SIGN-007: Reuse existing code, do not duplicate
**Trigger:** implementing a deliverable.
**Instruction:** Reuse src/models/tcn_cross_attn.py (encoder), src/onset/expert_pattern.py (rules), src/evaluation/c3_dose_response.py + cluster_bootstrap (bootstrap), src/eval_e3 (leakage plumbing). Import, don't reimplement.
**Reason:** The repo already has the building blocks.

### SIGN-008: IRM is not a free lunch
**Trigger:** working on T-004 (regime-invariant).
**Instruction:** Keep IRM assertions STRUCTURAL (penalty finite, decreases over steps). Do NOT assert IRM beats ERM — it only helps under diverse spurious correlations and can underperform ERM otherwise.
**Reason:** Over-claiming IRM superiority would be false and would make the test flaky.

### SIGN-R1: REAL-experiment gate = machinery-runs, finding is RECORDED not gated
**Trigger:** a task marked REAL (ID3/WS2/DB2/SYN).
**Instruction:** The task passes when its hermetic smoke test is green AND the real experiment script has been run end-to-end and its stats.json committed under results/identify/. The NUMERIC finding — including a null/zero/negative identified contribution — is written to progress.md but NEVER gates completion. A clean null is a valid, important result for this research line.
**Reason:** This line's contribution is identification/measurement, not performance. A performance gate would be dishonest and would never converge (signal is weak).

### SIGN-R2: No new LLM spend without explicit approval
**Trigger:** any task that would call the LLM API.
**Instruction:** ID3/WS2/DB2 must reuse existing scored data (results/poc_full, results/e3_*) for $0 cost. Do NOT add new LLM calls. The only task allowed to need new scoring is CF, which is skip:true until a human flips it.
**Reason:** Autonomous loop must not incur surprise API spend.

### SIGN-R3: Every identified estimate carries a clustered CI + a leakage-validity check
**Trigger:** reporting any identified contribution or corrected estimate.
**Instruction:** Use date-clustered bootstrap (src/evaluation/onset_eval.clustered_bootstrap), report absolute estimates with 95% CIs, and state whether the leakage-validity precondition (ID1) holds for the data used. An estimate without a passing validity check is not "identified" — label it accordingly.
**Reason:** The whole novelty is identification under the no-leakage condition; the CI and the validity check are what make the estimate trustworthy.

### SIGN-A1: Report tradable (economic) alpha, not only IC; shrinkage is not failure
**Trigger:** market-neutral / beta-decomposition tasks (NB-line).
**Instruction:** Report the tradable market-neutral long-short return and its Sharpe (date-clustered/block bootstrap), not only RankIC. Market/sector neutralization REMOVES the beta component, so the idiosyncratic signal is EXPECTED to be smaller than raw -- a smaller-but-stable idiosyncratic signal is the goal, and a clean null is a valid finding. Never treat shrinkage vs raw as a failure or gate on it.
**Reason:** Alpha = small, stable, leverable idiosyncratic predictability, not large beta-driven raw predictability. The honest measurement is the deliverable.

### SIGN-K1: Alpha verdict requires multi-split + net-of-cost; never trust a single split
**Trigger:** candlestick-onset REAL tasks (K-line) and any "did we find alpha" claim.
**Instruction:** Declare a REAL edge (alpha1) ONLY if the POOLED market-neutral long-short Sharpe, NET of 0.2% round-trip cost, has a clustered 95% CI excluding 0 AND the net Sharpe is positive in >= 2 of 3 walk-forward splits. A signal that only appears on one split (e.g. the NB5 split3 Sharpe 2.18 with null RankIC) is labelled 'single-split / overfit', NOT alpha. Always report GROSS and NET; report per-split AND pooled.
**Reason:** Candlestick backtests are the canonical data-snooping trap, and we already produced one single-split mirage (NB5). Multi-split + net-of-cost + pooled-CI is the bar that separates a real small edge from self-deception.

### SIGN-D1: Tradability is LONG-ONLY, net of realistic A-share cost; deployable needs multi-split + cross-period
**Trigger:** deployability tasks (LO/COST/ROB/CAP/DSYN line).
**Instruction:** A-shares mostly cannot be shorted, so the only tradable object is a LONG-ONLY top-K basket and its EXCESS over the equal-weight market. Report that, NET of the realistic A-share cost model (asymmetric stamp duty sell-only, commission, slippage), excluding un-enterable limit-up entries. An edge counts as DEPLOYABLE only if the long-only net market-excess: pooled CI excludes 0, is positive in >= 2/3 of the 2025 splits, AND holds in >= 1 additional year (2023/2024). Otherwise label 'statistically-real-but-not-tradable' or 'collapsed'. Never call a long-SHORT result tradable on A-shares.
**Reason:** The factors+candle long-short cleared SIGN-K1, but long-short is not executable on A-shares; long-only removes the short alpha and is the true deployability test.

### SIGN-P1: Test the production MECHANISM (timing + extreme filter + asymmetry), not a broad factor; optimize walk-forward only
**Trigger:** production-edge tasks (TIM/FILT/COMBO/OPT/PSYN line).
**Instruction:** The object under test is V12.31's actual edge -- DISASTER-MONTH TIMING + EXTREME FILTERING to a tiny high-conviction long-only pool + ASYMMETRIC downside avoidance -- NOT a broad cross-sectional factor (already null). Always report the Sharpe DECOMPOSITION (buy-hold vs +timing vs +timing+filter). Any knob optimization MUST be walk-forward: choose the config on TRAIN windows, judge on TEST; never report the best in-sample config as the result. Deployable verdict still requires SIGN-D1 (long-only, net realistic cost, holds cross-period).
**Reason:** Our broad cross-sectional tests kept finding null because the production edge is in timing/filtering/asymmetry, which broad RankIC/long-short averages away. And the extreme-filter pool is small -> high snooping risk; only walk-forward selection + cross-period survival counts.

### SIGN-M1: MI is positively biased -- significance ONLY from a within-stratum permutation null
**Trigger:** mutual-information tasks (onset-motif line).
**Instruction:** Binned/estimated mutual information is positively BIASED (finite-sample), so a positive raw MI is NOT evidence. Significance MUST come from a PERMUTATION null: shuffle the target WITHIN regime strata (breaking the conditional dependence while preserving the marginals) and compare the observed conditional MI to that null -> a p-value. Report MI relative to its permutation null, never raw. Conditional MI I(pattern;return|regime) is the decisive object; marginal MI is the control. Apply the same cross-period stability bar (>=2/3 years) to the INFORMATION as SIGN-D1 applies to returns.
**Reason:** Without the permutation null, MI-estimator bias would manufacture a false 'conditional information exists' just as single-split backtests manufactured false alpha. This probe is the go/no-go for the whole onset-motif upgrade -- it must be unfoolable.

### SIGN-T1: MI is sign-blind -- tradability needs conditional MEAN, not mutual information
**Trigger:** motif-tradability line (converting the stable conditional information into a deployment decision).
**Instruction:** Mutual information measures dependence but is SIGN-BLIND: I(X;Y|Z)>0 can come from E[Y|X,Z] moving with X (DIRECTIONAL, tradable long) OR from Var[Y|X,Z] moving with X (volatility/risk, NOT directly tradable long). To decide tradability you MUST use: (1) the CONDITIONAL MEAN / sign-hit-rate (directionality), (2) the conditional QUANTILE response (monotonicity -- monotone is tradable, hump/U is not), and (3) a net-of-cost, long-only, cross-sectional backtest with date-clustered CIs and per-year splits. Never infer tradability from MI. An 'information-only' (real-but-not-tradable) verdict is a valid, publishable finding -- record it, never gate on it.
**Reason:** The onset-motif line proved STABLE conditional information exists, but the whole program already showed return edges collapse after cost/long-only/cross-period. Skipping the mean/monotonicity/net-of-cost checks and trusting the MI verdict would risk building a complex motif model on sign-blind risk information that can't be traded long.

### SIGN-B1: Benchmark fairness -- reimplement on OUR protocol + always report Deflated Sharpe
**Trigger:** SOTA-comparison tasks (regime-baseline-bench line).
**Instruction:** Recent regime/uncertainty/info-theory baselines are on US markets (S&P500/NASDAQ-100), different tasks (price MAPE, network structure), and mostly have no released code. Do NOT claim to reproduce their published numbers. The fair, decisive test is to reimplement each baseline's METHOD on OUR data (D1 A-shares) under OUR honest protocol (leakage-free walk-forward, A-share round-trip cost, date-clustered CIs, point-in-time). ALWAYS report the Deflated Sharpe Ratio (Bailey & Lopez de Prado) alongside raw Sharpe to correct for the many strategy variants tried -- a baseline that wins on raw Sharpe but fails DSR is NOT a win. A clean 'the baseline is also sub-cost / null under our protocol' is a valid, publishable result.
**Reason:** The whole program's value is honest, deployment-realistic evaluation. Comparing our honest numbers to their optimistic US numbers would be apples-to-oranges and self-defeating; reimplementing on our turf with multiple-testing correction is the only fair and informative comparison.
