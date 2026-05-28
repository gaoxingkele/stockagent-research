# Decision Log

格式：
```
YYYY-MM-DD: <决策标题>
- Context: ...
- Options considered: ...
- Decision: ...
- Reason: ...
- Affected experiments: ...
- Revisit date: ...
```

---

## 2026-05-28: 启动研究分支，与生产线物理隔离

- **Context**: 生产线 V12.31 已上线 (commit 5359e26 / tag v12.31-production-baseline), 用户决定开 paper research 分支冲 CCF A
- **Options considered**:
  - (a) 当前 repo 加 `research/` 长生命周期分支
  - (b) 新建独立仓库 `stockagent-research`
- **Decision**: (b) 新建独立仓库
- **Reason**:
  1. 生产数据有 Tushare 付费 license, 不能进开源 repo
  2. 研究代码要上 supplementary, 必须可公开复现
  3. 生产线持续迭代, 物理隔离避免污染论文实验
- **Affected experiments**: 全部
- **Revisit date**: 投稿前合并部分通用 utility

---

## 2026-05-28: 主理论框架定为 HSSM 双层 (Hierarchical Switching State-Space Model)

- **Context**: §2.7 生成模型选型, VAE / Flow / Diffusion 三选一
- **Options considered**:
  - (a) VAE (具体 SNLDS 单层)
  - (b) VAE (具体 HSSM 双层)
  - (c) Normalizing Flow
  - (d) Diffusion
- **Decision**: (b) HSSM 双层
- **Reason**:
  1. 层次结构对应市场层次 (macro regime + stock onset)
  2. 直接对接生产线市场环境感知系统 (project_market_context)
  3. 比 SNLDS 单层多一个 propositions 来源 (Regime-Conditional Asymmetry)
  4. 灾难月可解释力强
- **Fallback**: SNLDS 单层 → 经典 HSMM 三档
- **Affected experiments**: E4.x, A3, A6
- **Revisit date**: Gate 3 (M4 末)

---

## 2026-05-28: Gate 1 优先, 决定 paper 主线生死

- **Context**: paper 主张 PWC 是 backward-context 维度, 与 LdP forward triple-barrier 正交
- **Decision**: M2 末必须完成 E1.1-E1.4 四个 cell 的实验, 用 DM 检验判定 Gate 1
- **Reason**: 这是 paper 故事是否成立的核心赌注, 早做早决策
- **Affected experiments**: E1.1, E1.2, E1.3, E1.4
- **Revisit date**: 2026-07-31 (M2 末)

---

## 2026-05-29: Gate 1 FAIL — PWC sample filter NOT orthogonal to TB label

- **Context**: paper_plan §10.2 / execution_plan §3.3 Gate 1 critical experiment
- **Setup** (single split, single seed, D1 A 股 2022-2026):
  - E1.3 TB H=20 alone:                                Top20 Sharpe = 6.76
  - E1.4 TB H=20 + fixed_pwc filter (双向 |past_r5|≤8%): Sharpe = 2.56 (-62%)
  - E1.4 v2 same with 单向 past_r5≤+8%:                  Sharpe = 3.62 (-46%)
  - E1.5 TB H=5 alone (matched horizon):               Sharpe = 10.32
  - E1.6 TB H=5 + fixed_pwc filter:                    Sharpe = 3.09 (-70%)
- **Conclusion**: PWC backward sample filter (whether unidirectional or bidirectional, whether H matched or not) is consistently NEGATIVE on TB-labeled training.
- **But**: PWC AS LABEL still beats FH (E1.2 Sharpe 3.42 vs E1.1 2.00, +71%)
- **Insight**: PWC's effectiveness arises from JOINT action of (past constraint + sparse 3-class definition + drawdown-bounded path), not from backward filtering alone. TB already implicitly captures path-based filtering through its forward-barrier mechanism, so adding PWC filter is redundant and harmful (creates train/test distribution shift).
- **Paper impact** (per paper_plan §10.2 decision tree):
  - Original main claim "PWC is orthogonal backward dimension to LdP forward triple-barrier" is FALSIFIED on this dataset
  - Target downgrade: KDD A → CIKM / ICDM (still possible)
  - C2 rewrite needed: "PWC as integrated label engineering for sparse onset detection in asymmetric (short-sale-constrained) markets"
  - Adaptive PWC (HSSM-based) may or may not change this; sample filter limitation likely persists
- **Caveats before final FAIL declaration**:
  - Single split / single seed — should rerun with 5 seeds + walk-forward before paper-grade conclusion
  - Test set is unfiltered while train set is filtered (potential distribution shift); could test "filter both"
  - PWC was designed for v3c's 5-day onset definition; the filter mechanism may behave differently when adapted (HSSM Gate 3) — should not foreclose adaptive direction
- **Next**:
  - Run E1.7: PWC filter on FH baseline (does it boost weak label? — supplementary evidence)
  - Run E1.8: filter both train AND test (sanity, not for paper main)
  - Add 5-seed + walk-forward before any submission
- **Revisit date**: 2026-07-31 (M2 末 originally; now possibly earlier given main claim shift)

---

## 2026-05-28: 发现并修复生产 factor_lab 的 forward-looking 字段泄漏

- **Context**: E1.1 第一次跑出 RankIC = 0.5636, IR = 1.47, Sharpe = 20.75 — 远超合理范围 (memory feedback_st_exclude_at_source: IC>0.5 必查泄漏)
- **Diagnostic** (scripts/diag_leakage.py):
  - spearman(r5, fwd_r5) = +0.8983
  - r5 本身就是 close.shift(-5)/close - 1 (forward), 不是 past return
  - 共 10 字段泄漏: r5/r10/r20/r30/r40, dd5/dd10/dd20/dd30/dd40
- **Root cause**: 生产 factor_lab 用这些字段作为**训练 label 辅助** (pump_score 等), 在研究 panel 里被当 feature merge 进来
- **Fix**: train.py NON_FEATURE_COLS 加 10 字段黑名单
- **Future**: build_d1.py 应该在落 panel 时就 rename 这些字段为 `_label_*` 前缀, 防再次误用
- **Affected experiments**: 所有用 D1 的实验, 必须重跑
- **Revisit date**: 2026-06-15 (build_d1 重构时)
