# 7. Discussion

## 7.1 Why Does Stratification Inflate LLM Advantage?

A core surprise of this paper is that the apparent +54% Top-10% advantage of hybrid LLM-LGBM routing under stratified sampling reverses to a −15% disadvantage under random sampling that matches the production deployment distribution.

Two mechanisms appear at play.

**Class-imbalance differential sensitivity.** Tree models (LGBM) handle severe class imbalance natively through leaf-level distribution learning, while LLM predictions, trained on uniform distributions across diverse tasks, exhibit miscalibration that compounds in sparse-positive regimes. When the test distribution is artificially balanced (stratification), the LLM's miscalibration is masked; when restored to the true rate, the LGBM's resilient calibration dominates.

**Top-K compression.** At onset rate $\rho = 25\%$ (stratified), the top decile contains predominantly true positives, and any reasonable ranker scores well. At $\rho = 8\%$ (random), the top decile is dominated by near-onset ambiguous anchors, where the LLM's reasoning chain produces qualitatively different (often overconfident) outputs than the LGBM's marginal-probability ranking.

**Implication for the field.** Any future paper claiming LLM-augmented improvement on a movement prediction task should report the sampling protocol, the true onset rate, and at minimum a per-stratum breakdown. We hope this paper triggers an audit of recent finance-LLM benchmarks for similar artifacts.

## 7.2 When Does LLM Add Value? Cross-Quarter Evidence

Even with the disappointing pooled walk-forward results, the per-split breakdown reveals genuine LLM contribution in specific regimes. In Split 2 (2025-Q3), where the LGBM Pattern Core fails (RankIC −0.002), the raw LLM achieves a meaningful Top-10% return of +1.10% — outperforming both LGBM (+0.88%) and all hybrid routers on this split.

We hypothesize three drivers of regime-conditional LLM value:

1. **Distribution shift the LGBM training window did not see.** Split 2's training window ends 2025-03; the test window (2025-Q3) may contain market dynamics structurally different from the training period. LLM agents, trained on broader corpora and reasoning over the per-anchor context, exhibit more graceful degradation.
2. **News-driven onsets.** While our prompts do not include news, the LLM's pretraining corpus contains historical news patterns that may inform calibration under unfamiliar volatility regimes.
3. **Conservative reasoning bias.** Under high uncertainty (Split 2 features anomalous volatility patterns), the LLM's "I don't know" prior softens its probabilities, producing better ranking than overconfident LGBM extrapolation.

Empirically, identifying which quarter falls into which regime *online* is the core challenge for any practical router. We compute the upper bound at +0.39pp Top-10% (oracle) and demonstrate that our online routers capture less than 20% of it; closing this gap is the central topic of Pathway 2.

## 7.3 The Path Forward: Pathway 2 (TCN+Cross-Attention) and Pathway 3 (Barlow Twins SSL)

The V12-Agentic framework presented here is modular by design: the Pattern Core is the natural locus for representation upgrades. We outline two future directions.

**Pathway 2: Spatio-Temporal Cross-Attention Pattern Core.** Replace LGBM with a TCN-causal-dilated encoder that processes the 60-day historical bar sequence for each stock, followed by a spatio-temporal cross-attention layer that learns dynamic factor-axis × time-axis interaction. The TCN's causal dilations capture multi-scale onset patterns (short consolidation → breakout → continuation), and the cross-attention provides factor-importance dynamics conditional on the temporal context. We project this upgrade to deliver a 0.05–0.10 absolute RankIC improvement and to materially close the +0.39pp oracle headroom.

**Pathway 3: Barlow-Twins-Pretrained Encoder.** Pretrain the Pathway 2 encoder with the Barlow Twins redundancy-reduction objective (Zbontar et al. 2021) on the full D1 unsupervised panel (4 years × 5,434 stocks ≈ 5M anchors). The Barlow Twins objective avoids the negative-sample-construction problem (problematic in finance due to cross-stock cointegration) and learns features invariant to multiple data augmentations (e.g., Gaussian noise, dropout-style masking, time-warp). We hypothesize that the Barlow-Twins-pretrained encoder will narrow the gap between in-distribution and out-of-distribution test performance, partially addressing the cross-quarter heterogeneity issue identified in this paper.

## 7.4 Limitations

We acknowledge five limitations.

1. **Single-market scope.** The walk-forward evaluation covers only Chinese A-shares. Cross-market validation (NASDAQ, KOSPI, NIFTY) is planned but not in this paper.
2. **3 splits is statistically thin.** We mitigate via 1000-resample bootstrap CIs, but encourage a 6–12 split replication.
3. **Single LLM family.** All LLM experiments use Anthropic Claude (Sonnet 4.6, Haiku 4.5, Opus 4.8). Cross-model replication with GPT-5 / Gemini / Qwen is left for follow-up.
4. **No closed-loop reflection.** The Backtest Verifier reports performance but does not currently trigger upstream agent re-elicitation. Implementing this reflection loop is part of Pathway 2.
5. **Expert prompt is China-specific.** The V12.31 expert prompt encodes A-share-specific rules (ST status, 涨跌停 limits, 龙虎榜). Adaptation to other markets requires re-elicitation; we do not claim direct transferability.
