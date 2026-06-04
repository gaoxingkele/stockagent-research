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
