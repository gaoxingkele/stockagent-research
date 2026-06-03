---
description: Run ONE Ralph iteration on the onset-algorithm backlog (plans/prd.json). Windows-native, no bash. Drive autonomously via the native /loop command.
---

# Ralph onset-algorithm iteration

You are running ONE iteration of an autonomous algorithm-implementation loop on this project. Do exactly one task's worth of work, verify it, persist progress, then stop. The harness `/loop` (or the user) will re-invoke you for the next iteration.

## How to run the loop
- Autonomous, self-paced: `/loop /ralph-loop`
- Bounded: `/loop 0 /ralph-loop` then stop when the promise appears, or just re-run `/ralph-loop` manually.
- This project has NO working bash/jq/perl, so the bash stop-hook and `scripts/ralph/*.sh` are NOT used. This command IS the loop body.

## Steps for THIS iteration

1. **Read context** (in order):
   - `plans/guardrails.md` — hard constraints. Obey every SIGN. (Especially SIGN-001: the gate is RUNS-CORRECTLY, not beats-baseline.)
   - `plans/progress.md` — what's done / what's next.
   - `plans/prd.json` — the task backlog.

2. **Pick the next task**: the lowest-`priority` task with `passes: false` and `skip` not true, whose dependencies (see notes / progress.md dependency order) are already `passes: true`. If every non-skipped task is `passes: true`, go to step 7 (completion).

3. **Set up branch** (first iteration only): ensure you are on the branch named by `branchName` in `plans/prd.json` (create from main if needed). Never commit to main.

4. **Implement** the task's `deliverable` and its `test`:
   - Reuse existing repo code per SIGN-007 (TCN encoder, expert rules, c3 bootstrap, eval_e3). Import, don't reimplement.
   - Write the hermetic test FIRST (synthetic data, CPU-only, <5s) so it fails, then implement until it passes (SIGN-002).
   - Keep changes small and focused on this one task.

5. **Verify** by running the gate via PowerShell:
   ```
   .venv-xpu\Scripts\python.exe -m pytest tests/algo -q
   ```
   If it fails, fix and re-run. Do NOT proceed until green. (If you are genuinely blocked by something requiring a human — e.g. missing credentials — set `skip: true` + `skipReason` on the task and move on.)

6. **Persist progress**:
   - Set the task's `"passes": true` in `plans/prd.json`.
   - Append a dated entry to `plans/progress.md`: what you built, key decisions, what's next.
   - If you learned a constraint worth enforcing, append a new `SIGN-xxx` to `plans/guardrails.md`.
   - Commit: `git add` the deliverable + test + plans, commit `feat(onset): T-00X <title>` (end body with the Co-Authored-By line per repo convention). Stay on `onset/algo-impl`.

7. **Completion check**: re-read `plans/prd.json`. If ALL non-skipped tasks are `passes: true` AND `pytest tests/algo` is green, output exactly:
   `<promise>ONSET_BACKLOG_COMPLETE</promise>`
   Otherwise end the iteration normally (do not output the promise) so the loop continues with the next task.

## Hard rules
- One task per iteration. Do not batch multiple tasks.
- Never gate on predictive performance (SIGN-001). The signal is weak by design.
- Never use system Python; always `.venv-xpu\Scripts\python.exe` (SIGN-005).
- Point-in-time safe, cluster-robust, hermetic tests (SIGN-002/003/004).
