# V12.31 Expert Knowledge Prompt — v1 (Round 1)

**Source**: User oral interview (2026-05-29), augmenting deployed V12.31 V7c rules.
**Status**: Round 1 (4 questions answered). Round 2 pending (W-bottom / U-bottom forms, disaster month rules, sector stratification).
**Usage**: Inject as system prompt for LLM agent in stock movement onset detection.

---

## A 股看多启动子 (Bullish Movement Onset) 判定规则

### 必要条件（三个全部满足，"启动子" 成立）

**条件 1：底部抬高**
```
low_5d  ≥  low_20d × 0.98
```
- `low_5d` = 最近 5 个交易日 (含当日) 的最低价
- `low_20d` = 最近 20 个交易日的最低价
- 含义：底部正在抬高，允许 2% 容忍度避免噪声误杀真启动信号

**条件 2：已脱离 5 日低点 5%**
```
close[t]  ≥  low_5d × 1.05
```
- 含义：股价已经离开近期低点至少 5%，证明启动信号被市场确认
- 等价：最近 5 日最低不低于当前价格的 -5%

**条件 3：均线整理 + MA5 上翘**
```
MA5, MA10, MA20 三均线 spread < 3% (整理收敛)
AND
MA5 最近 2 日均值 > MA5 前 5-2 日均值 (MA5 上翘)
```
- 含义：长期均线粘合（蓄势状态），短期均线开始向上发起攻击
- 形态：均线粘合后的"龙抬头"

### 加分项（不强制，作为 confidence boost）

**条件 4：放量配合**
```
volume_5d_mean  >  volume_20d_mean × 1.2
```
- 含义：放量启动是加分项，但缩量启动也可接受（如机构吸筹场景）

---

## A 股看空启动子（Bearish Movement Onset）—— 不对称设计

**关键**：A 股卖空受限，看空启动子**不是看多启动子的镜像**，仅作**避险信号**使用。

- 检测到看空启动子 → **不入场（避开此股）**
- **不构造做空 portfolio**（与 v3c 设计一致，commit c210d4a）
- 论文 framing 支撑：Short-sale constraint asymmetry (paper §1.3 Proposition 3)

---

## 启动子失效场景（不要预测启动子）

以下场景下，即使表面满足启动子规则，也不要推荐为 onset：

| 场景 | 判定 | 来源 |
|---|---|---|
| 灾难月 | 大盘单日 -3% 以下且板块全绿时 | V12.31 灾难月分析 |
| 行业大跌 | 行业 60 日动量排名最差 10% (V12.31 industry_mom_excl=0.10) | V7c 铁律 5 |
| 横盘僵尸股 | MA60 长期横盘 (V12.31 zombie filter) | V7c 铁律 4 |
| ST 股 | A 股退市预警 (data layer 源头排除) | V12.31 全链路 |
| Already-sustained uptrend | past_r5 > +8% 时已是 trend continuation 非 onset | v3c 时序约束 |

---

## 启动子的统计学意义（label-level）

V12.31 实战中"启动子"对应的 ground truth label（生产 v3c）：
- bullish_onset (class +1): 未来 5 日累积涨幅 ≥ 10% 且回撤 ≤ 5%
- bearish_onset (class -1): 未来 5 日累积跌幅 ≤ -10% 且反弹 ≤ 5%
- neutral (class 0): 其他

但 expert 判断**远不止 label rule**，更包含上面的形态识别 + 失效场景剔除。这是 paper 的核心 contribution: **将 unencoded expert knowledge 注入 LLM agent**。

---

## 决断度信号 (paper §2.7 C3)

```
pump_ratio = P(bullish_onset) / (P(bearish_onset) + 0.01)
```

ratio 越大越涨、< 1 越跌、1-2 震荡。
实战验证：v12.31 改用 ratio 排序后 α +0.16pp/月 + 灾难月翻盘 +2.30pp。

---

## V12.31 产线 V7c 5 铁律（已 deployed 的）

作为对照，生产代码已有的过滤（commit 5359e26 / v12.31）：

```python
def apply_v7c_rules(df):
    # 1. r20_pred 当日 top 5% (模型预测 20 日收益排名)
    # 2. pyr_velocity_20_60 < p35 (前期还没爆发, 留余地)
    # 3. |f1_neg1| < 0.005, |f2_pos1| < 0.005 (PCA 主成分约束)
    # 4. NOT is_zombie (非横盘僵尸股)
    # 5. industry_mom_60d_rank >= 0.10 (排除最差 10% 行业)
```

**Paper insight**：上面 Round 1 capture 的"底部抬高 + 均线整理 + 已脱离 5 日低点 5%"**没有在 V7c 5 铁律中实现**。这是 unencoded expert knowledge — 把它 encode 给 LLM 是 paper 的方法学 contribution。

---

## Round 2 Capture (2026-05-29)

用户决策 (4 questions answered):

| 维度 | 决策 | Implication |
|---|---|---|
| W底 / U底 / 圆弧底子形态 | **不区分**，底部抬高够用 | paper 不展开子形态分类 |
| 灾难月避免信号 | **复合信号**（指数 + 量能 + 板块状态） | Round 3 专攻：拆解每个组件阈值 |
| 市值 / 板块分层 | **统一规则**（与 V12.31 一致） | paper 不做分层 ablation |
| 突破型 vs 反转型 | **不区分**，统一作为 onset | V12.31 的 pyr_velocity_20_60 < p35 已隐式过滤 |

**Round 2 设计原则**：保持简洁，3 维选择简化。这与 V12.31 deployment 哲学一致 — 复杂规则容易 overfit，统一规则在实战上更鲁棒。

## Round 3 Capture (2026-05-29) — 灾难月避免机制

用户决策 (4 questions answered) — 灾难月复合信号设计：

```
DISASTER_MONTH_SIGNAL = vote(Signal_A, Signal_B, Signal_C) >= 2/3
```

### Signal A: 指数信号 (AND 双指数)
```
sh_index_today < -2.0%  AND  gem_index_today < -3.0%
```
- 上证 (大盘代表) + 创业板 (成长代表) 同时大跌
- AND 设计降低单一指数误报

### Signal B: 量能信号 (OR 三个子条件任一)
```
B1: amount_5d_mean / amount_20d_mean < 0.70    # 全市场成交额近 5 日萎缩 > 30%
B2: limit_down_count > 100  OR  limit_down/limit_up > 3.0   # 跌停股恐慌
B3: up_stock_pct < 0.30  OR  down_stock_pct > 0.70           # 市场宽度劣化
```
- OR 设计：任一量能恶化即触发（量能信号高敏感）

### Signal C: 板块状态 (内层 vote >= 2/3)
```
C1: industry_red_pct > 0.80                      # > 80% 一级行业下跌
C2: top5_hot_concepts_returns 全部 < 0           # Top 5 热概念全绿
C3: top5_hot_concepts_avg_return < -1.0%         # Top 5 热概念均跌 > 1%

Signal_C = (C1 + C2 + C3) >= 2
```
- 板块层面也用 vote 机制 (避免单一概念异动误判)

### 外层 复合规则
```
Disaster Month = (Signal_A + Signal_B + Signal_C) >= 2
```
- 三个 outer 信号中任 2 触发即视为灾难日
- 不要求全部触发（AND 太迟钝）
- 也不接受任一触发（OR 太误报）

### 灾难日触发后

- **当日**: 不入场新启动子
- **持有期**: 持有的启动子不强平（V12.31 双轨架构持有规则）
- **持续到**: Signal_A/B/C 都退出灾难状态（vote < 2 即恢复正常）
- **反向风控**: V12.31 R5 反向 (R20 高 + R5 低 → "先下蹲后起跳") 在灾难月期间**仍然有效**, 见 [[project-reverse-riskctl-0519]]

### 与 V12.31 生产 market_context.py 的对照

生产 `market_context.py` (1629 行) 已有：
- `MAJOR_INDICES`: 上证 / 沪深 300 / 深成 / 创业板 / 中证 500 / 中证 1000
- `classify_trend_state`: 8 种趋势状态枚举
- `_compute_market_score`: 大盘综合评分
- `_analyze_sector_heats`: 板块热度
- `_run_vision_analysis`: 视觉 LLM 研判
- `_get_mkt_moneyflow_signal`: 资金流信号
- `_fetch_news_hot_topics`: 新闻热点

Paper 中我们 reference 这些 production data sources, 但用 panel data 自己 aggregate 出灾难月信号（不依赖实时 API）, 保证 reproducibility。

---

## TODO (Round 4 待 capture)

其他维度：
- [ ] 启动子 horizon 是 5 天固定还是 adaptive
- [ ] 板块联动信号（同板块多股同时 onset）
- [ ] 龙虎榜数据如何辅助识别真启动子
- [ ] 概念热度因子作为 onset 加分项

---

## Change Log

- **v1 (2026-05-29 Round 1)**: 4 questions captured (bottoms rising, above 5d low, MA pattern, volume, bearish asymmetry)
- **v1 (2026-05-29 Round 2)**: 4 questions captured (no sub-pattern split, no stratification, no subtype split, composite disaster signal pending)
- **v1 (2026-05-29 Round 3)**: 4 questions captured (disaster month vote >= 2/3 mechanism, index AND + volume OR + sector inner-vote)
