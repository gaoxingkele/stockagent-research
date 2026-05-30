# 8. Conclusion

We have presented **V12-Agentic**, a four-agent framework for stock movement onset detection that decomposes the prediction task into Macro Regime Monitoring, Alpha Factor Exploration, Pattern Core classification, and Backtest & Verification. Each agent maps to a concrete component of a deployed quantitative trading system (V12.31, real α +2.236pp/month, Sharpe 2.20 on Chinese A-shares), and we contribute a knowledge encoding methodology that converts the production system's tacit expert knowledge into an LLM-usable prompt.

Through walk-forward random-sample evaluation on 6,000 stock-day anchors, we report three honest findings:

1. **Hybrid LLM-LGBM routing does not robustly beat LGBM** in the realistic deployment distribution. The best hybrid (G_disagreement_boost) ties LGBM's pooled Top-10% return at +1.63%, with overlapping 95% bootstrap confidence intervals.
2. **Stratified sampling overstates LLM contribution by a wide margin.** What appears as a +54% Top-10% advantage in a 25%-positive-rate stratified PoC reverses to a −15% disadvantage under the 8%-positive-rate random sample matching production deployment.
3. **Cross-quarter heterogeneity creates an oracle ceiling of +0.39pp Top-10% return** for any online regime-conditional router. Our naive routers capture less than 20% of this headroom, suggesting that richer regime conditioning (rolling RankIC tracking, sector momentum embedding, learned regime detectors) is needed to monetize the heterogeneity.

These findings constitute a methodological warning for the LLM-finance benchmarking community and a concrete research target for follow-up work.

Pathway 2 of our research program upgrades the Pattern Core to a Temporal Convolutional Network with Spatio-Temporal Cross-Attention. Walk-forward evaluation across the same three splits shows Phase 2 TCN achieves mean Top-10% return +1.61% vs LGBM +1.63% — equivalent performance with **35× less training data** — and notably **bridges the LGBM failure mode in Split 2** (RankIC +0.034 vs LGBM −0.015). Pathway 3 further pretrains the encoder with the Barlow Twins redundancy-reduction objective on the full unlabeled D1 panel, providing a data-efficient route toward matching and potentially exceeding the LGBM baseline. Both pathways are scaffolded by the V12-Agentic framework introduced here, ensuring that representation upgrades can be measured against a controlled, deployment-realistic baseline.

We release the full evaluation harness, walk-forward sampler, hybrid router suite, and the V12.31 expert prompt as supplementary material to enable reproduction and extension of these results.
