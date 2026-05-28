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
