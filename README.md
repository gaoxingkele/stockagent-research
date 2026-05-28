# stockagent-research

**Movement Onset Classification under Short-Sale Constraints**

Research track for CCF A paper submission.
Production system (`stockagent-analysis` @ tag `v12.31-production-baseline`) is **physically isolated** from this research repo.

## Status

- 起算日期：2026-05-28
- 当前阶段：M1.1（数据 D1 整合 + LGBM/FH baseline）
- 目标 deadline：KDD'27 ADS Track (~ 2027-02)
- 生产线 baseline：`stockagent-analysis` tag `v12.31-production-baseline`

## Quick Start

```bash
# 1. 数据准备
make data_d1                # A 股 Tushare 整合 (复用生产线)
make data_d2                # Qlib Alpha158 公开版
make data_d3                # NASDAQ-100

# 2. 跑 baseline
make exp_e1_1               # LGBM + FH baseline on D1

# 3. Gate 1 关键实验
make gate_1                 # TB + Fixed-PWC 正交性验证
```

## 文档

- **理论 / 方法 / outline**: `docs/paper_plan.md`
- **数据 / 标注 / 实验 / 改进**: `docs/execution_plan.md`
- **决策日志**: `docs/decision_log.md`
- **每周进度**: `progress/YYYY-WW.md`

## 目录

```
src/
  onset/        HSSM / BOCPD onset state inference
  labels/       FH / TB / OTL / DL / PWC label generators
  models/       LGBM / HIST / MASTER / StockMixer / FactorVAE
  ranking/      ratio score / calibration
  data/         D1/D2/D3 dataset loaders
  evaluation/   metrics / DM test / bootstrap CI / FDR
  portfolio/    dual_track (移植自生产线)
configs/        Hydra yaml
experiments/    run outputs + aggregate analysis
data/           raw + processed + cache (DVC tracked)
paper/          LaTeX manuscript
tests/          pytest
```

## 5 Gates

| Gate | 节点 | 决定 |
|---|---|---|
| Gate 1 (M2 末) | TB + Fixed-PWC 正交 | paper 主线是否成立 |
| Gate 2 (M3 末) | HIST + PWC model-agnostic | 跨架构是否泛化 |
| Gate 3 (M4 末) | HSSM 双层 > 单层 | 主方法上限 |
| Gate 4 (M5 中) | NASDAQ 跨市场 | KDD ADS 可行性 |
| Gate 5 (M6 中) | 完整 T1 矩阵 | submit 决策 |
