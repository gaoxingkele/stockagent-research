# 2. Related Work

## 2.1 Stock Movement Prediction

**Classical statistical learning** approaches model stock movement as either a continuous regression (forecasting future returns) or as multi-class classification of price direction. Tree-based gradient boosting (LightGBM, XGBoost, CatBoost) remains the dominant production-grade baseline due to robustness against feature noise and natural handling of mixed feature types (Friedman 2001; Ke et al. 2017). Our Pattern Core (Phase 1) follows this paradigm.

**Deep learning** for stock prediction has explored recurrent and graph-based architectures: HIST (Xu et al. 2021) introduced hierarchical industry-aware GNN aggregation; MASTER (Li et al. 2024) uses market-guided Transformer encoders; StockMixer (Fan et al. 2024) adapts MLP-Mixer to multi-stock universes; FactorVAE (Duan et al. 2022) uses variational autoencoders to learn factor representations. These works typically evaluate on relatively curated, equally-weighted samples; few report performance under realistic random sampling with sparse positive events.

**Label engineering** is critical for movement prediction. López de Prado (2018) introduced the **Triple-Barrier Method (TBM)** with upper (profit), lower (stop-loss), and time-vertical barriers, producing labels in {−1, 0, +1}. The TBM is widely adopted and is one of our baseline labelers (E1.3 in our experiments). Optimal Trend Labeling (Han et al. 2023) further proposes dynamic-programming-based segmentation. Recent work also incorporates short-sale constraints and asymmetric label rules (Atilgan et al. 2022), motivating our v3c "past-window-constrained" label design.

## 2.2 LLM Agents in Finance

LLM agents are a 2024-2026 hot topic in finance. **FinAgent** (Yu et al. 2024, KDD) introduces a multi-modal foundation agent for trading that aggregates text, price charts, and economic indicators. **AlphaForge** (Hu et al. 2024, KDD) employs LLMs to generate symbolic alpha factors. **FinRobot** (Wang et al. 2024, NeurIPS workshop) provides an open-source agent platform. **TradingGPT** (Li et al. 2024) coordinates multiple LLM specialists.

Domain-tuned LLMs include **FinGPT** (Yang et al. 2023), **FinMA** (Xie et al. 2023), and **FinLLaMA** (Iacovides et al. 2024), each fine-tuned on finance-specific corpora. These works focus on language understanding (news, earnings transcripts) more than on time-series prediction.

Crucially, most prior LLM-in-finance papers evaluate on curated benchmarks (e.g., FLARE, BizBench, FinQA) or stratified subsets. Few quantify the gap between such evaluation and real production deployment. Our work directly bridges this gap by (a) anchoring all agents in a production system with documented α records and (b) reporting walk-forward random-sample results alongside stratified counterparts.

## 2.3 Multi-Agent and Reflection Architectures

The broader literature on **multi-agent LLM systems** has explored autonomous coordination (AutoGPT, BabyAGI), role-based collaboration (CAMEL, MetaGPT, AutoGen), and reflection-based self-correction (Reflexion (Shinn et al. 2023), Self-Refine (Madaan et al. 2023)). Our 4-agent decomposition draws on this lineage but differs in two respects: (i) each agent maps to a concrete production component with operational history, rather than to a designed-from-scratch role; (ii) we evaluate the system as a deployed pipeline, not on synthetic conversation traces.

## 2.4 Self-Supervised Representation Learning for Time Series

**Self-supervised time-series representation** has produced strong recent baselines: **TS2Vec** (Yue et al. 2022), **CoST** (Woo et al. 2022), and **TimeMAE** (Cheng et al. 2023) demonstrate that contrastive or reconstruction objectives yield transferable representations. **Barlow Twins** (Zbontar et al. 2021) introduces a redundancy-reduction objective that avoids negative-sample construction (problematic in finance where the very notion of a "negative sample" is contested due to cross-stock cointegration).

For time-domain encoders, **Temporal Convolutional Networks (TCN)** (Bai et al. 2018) with causal dilated convolutions provide multi-scale receptive fields without violating the no-look-ahead constraint. Cross-attention between time and feature axes is increasingly used in spatio-temporal modeling (e.g., Spacetimeformer (Grigsby et al. 2021), Crossformer (Zhang & Yan 2023)).

Our Pathway 2 plans to upgrade the Pattern Core to a TCN-Causal + Spatio-Temporal Cross-Attention architecture, and Pathway 3 to a Barlow-Twins-pretrained variant. Pathway 1 (this paper) establishes the agentic framework and walk-forward evaluation harness against which these representation upgrades can be measured.

## 2.5 Short-Sale Constraints and Asymmetric Label Design

The asymmetric impact of short-sale constraints on return predictability has been studied empirically: Atilgan, Demirtas, and Simsek (2022) document that in Chinese markets where short-selling is restricted, negative information from supply-chain peers carries disproportionately more predictive power for future returns. This finding motivates our v3c label design, which treats bearish onset as an avoidance signal rather than a tradeable direction, and our V12.31 Verifier which only constructs long-only portfolios with bearish-onset exclusion.

## 2.6 Industrial-Academic Gap and Our Positioning

Few prior works combine (i) a deployed industrial system with documented real-money α, (ii) walk-forward random-sample evaluation matching the production deployment distribution, and (iii) LLM-agent integration with explicit reflection on stratification artifacts. To our knowledge, V12-Agentic is the first such effort in the public literature. The agentic decomposition allows us to share insights with the academic community while preserving the production system's competitive moat.
