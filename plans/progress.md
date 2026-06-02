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
