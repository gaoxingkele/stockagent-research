"""Build LLM agent prompts for stock movement onset detection.

Two conditions for the PoC:
  - expert:  system prompt loaded from prompts/v12_31_expert_v1.md
             (Round 1-3 captured user knowledge)
  - raw:     minimal system prompt without V12.31 expert knowledge

The output JSON schema requirement is appended to the USER prompt (end of
context) so a long system prompt doesn't push it out of the model's attention.

Output JSON schema (strict):
{
  "p_up":      0.0-1.0,
  "p_neutral": 0.0-1.0,
  "p_down":    0.0-1.0,
  "rationale": "≤ 30-char Chinese, 1 sentence"
}
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"

_JSON_INSTRUCTION = (
    "\n\n=== 输出要求 (严格遵守, 违反则解析失败) ===\n"
    "直接输出一个 JSON 对象, 仅含三个浮点数字段, 不要任何其他文字或 markdown。\n"
    "格式必须是:\n"
    '{"p_up": 0.XX, "p_neutral": 0.XX, "p_down": 0.XX}\n'
    "约束:\n"
    "- 三个数 ∈ [0, 1], 三者之和约等于 1\n"
    "- 不要 rationale, 不要 explanation, 不要 markdown 围栏\n"
    "- 整体输出不超过 60 字符"
)


_SHARED_SYSTEM = (
    "你是 A 股短线方向预测助手。任务: 估计给定股票未来 5 个交易日的方向概率 "
    "(p_up, p_neutral, p_down, 三者和约等于 1)。"
)


def load_expert_system_prompt() -> str:
    """Expert condition uses the SAME short system prompt as raw; the expert
    knowledge body is injected at the head of the USER prompt instead, to
    bypass any system-prompt truncation that the API gateway may apply.
    """
    return _SHARED_SYSTEM


def build_raw_system_prompt() -> str:
    return _SHARED_SYSTEM


def load_expert_knowledge_body() -> str:
    """Expert knowledge as USER-prompt prefix (replaces system-prompt approach)."""
    md = (PROMPTS_DIR / "v12_31_expert_v1.md").read_text(encoding="utf-8")
    return (
        "下面是从 deployed 量化系统 V12.31 (实战 α +2.236pp/月, Sharpe 2.20) "
        "设计师处提取的专家知识, 请基于这些规则做判断:\n\n"
        "=== V12.31 专家知识库 (Begin) ===\n\n"
        + md
        + "\n\n=== V12.31 专家知识库 (End) ===\n\n"
    )


def _fmt_history(history_records: list) -> str:
    if not history_records:
        return "(no history)"
    lines = ["日期      | 开盘   高     低     收盘   涨幅    成交量"]
    lines.append("-" * 60)
    for r in history_records[-30:]:
        lines.append(
            f"{r['trade_date']} | "
            f"{r['open']:>6.2f} {r['high']:>6.2f} {r['low']:>6.2f} {r['close']:>6.2f} "
            f"{r['pct_chg']:>+5.2f}% {int(r['vol']):>10,}"
        )
    return "\n".join(lines)


def build_user_prompt(anchor: pd.Series, expert_prefix: str = "") -> str:
    """Build the user-turn prompt for one anchor sample.

    Optional expert_prefix: V12.31 expert knowledge body, injected before the
    stock context. JSON instruction is always appended at the END.
    """
    h = anchor["_history"]
    if isinstance(h, str):
        h = json.loads(h)

    parts = []
    if expert_prefix:
        parts.append(expert_prefix)
    parts.append("## 待判定股票")
    parts.append(f"代码: {anchor['ts_code']}")
    parts.append(f"行业: {anchor.get('industry', 'unknown')}")
    parts.append(f"日期: {anchor['trade_date']}")
    parts.append(f"当前收盘: {anchor['close']:.2f}")
    parts.append("")
    parts.append("## 关键因子快照 (T 时刻可用, 无 forward look)")
    factor_lines = []
    for c in [
        "ma_ratio_5", "ma_ratio_10", "ma_ratio_20", "ma_ratio_60",
        "rsi_14", "kdj_k", "kdj_d", "macd", "macd_hist",
        "boll_pct", "atr_pct", "vol_ratio_5", "vol_ratio_20",
        "winner_rate", "main_net", "holder_pct",
        "total_mv", "pe", "pb",
        "market_score_adj", "mf_strength", "mf_consecutive",
        "bias_10", "bias_20", "lr_slope_20", "channel_pos_60",
    ]:
        v = anchor.get(c)
        if v is None or pd.isna(v):
            continue
        if isinstance(v, (int, float)):
            factor_lines.append(f"  {c}: {v:.4f}")
        else:
            factor_lines.append(f"  {c}: {v}")
    parts.append("\n".join(factor_lines) if factor_lines else "  (因子缺失)")
    parts.append("")
    parts.append("## 30 日历史 K 线")
    parts.append(_fmt_history(h))
    parts.append("")
    parts.append("## V12.31 expert_pattern 计算结果")
    parts.append(f"  bottoms_rising:        {bool(anchor['_exp_bottoms_rising'])}")
    parts.append(f"  above_5d_low_5pct:     {bool(anchor['_exp_above_5d_low_5pct'])}")
    parts.append(f"  ma_pattern_ok:         {bool(anchor['_exp_ma_pattern_ok'])}")
    parts.append(f"  volume_boost (BONUS):  {bool(anchor['_exp_volume_boost'])}")
    parts.append(f"  is_bullish_onset:      {bool(anchor['_exp_is_bullish_onset'])}")
    parts.append(f"  onset_score (0-4):     {int(anchor['_exp_onset_score'])}")
    parts.append("")
    parts.append("## 当日市场状态")
    parts.append(f"  sh_index_pct:       {anchor.get('_mkt_sh_index_pct', float('nan')):.4f}")
    parts.append(f"  gem_index_pct:      {anchor.get('_mkt_gem_index_pct', float('nan')):.4f}")
    parts.append(f"  amount_ratio_5_20:  {anchor.get('_mkt_amount_ratio_5_20', float('nan')):.3f}")
    parts.append(f"  limit_down_count:   {anchor.get('_mkt_limit_down_count', 0)}")
    parts.append(f"  up_stock_pct:       {anchor.get('_mkt_up_stock_pct', float('nan')):.2%}")
    parts.append(f"  industry_red_pct:   {anchor.get('_mkt_industry_red_pct', float('nan')):.2%}")
    parts.append(f"  signal_A_index:     {bool(anchor.get('_mkt_signal_A_index', False))}")
    parts.append(f"  signal_B_volume:    {bool(anchor.get('_mkt_signal_B_volume', False))}")
    parts.append(f"  is_disaster_month:  {bool(anchor.get('_mkt_is_disaster_month', False))}")
    body = "\n".join(parts)
    return body + _JSON_INSTRUCTION


def parse_llm_response(text: str) -> dict | None:
    """Parse LLM JSON output. Returns None if invalid."""
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1])
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        else:
            return None
    for k in ["p_up", "p_neutral", "p_down"]:
        if k not in obj or not isinstance(obj[k], (int, float)):
            return None
    return obj
