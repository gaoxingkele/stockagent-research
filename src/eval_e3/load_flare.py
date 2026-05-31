"""E3 — load + parse the FLARE stock-movement benchmarks (ACL18 / BigData22 /
CIKM18), the datasets FinBen/PIXIU use to evaluate LLM stock-movement skill.

Each FLARE example has:
  id, query (full LLM prompt: ticker + target date + price CSV + tweets),
  text (price CSV), answer ("Rise"/"Fall"), choices, gold (0/1).

We parse out a tidy frame: id, ticker, date, gold, the most-recent-day price
features (a fair tabular baseline), and keep `query` for optional LLM scoring.

Datasets are pulled as parquet via the HF datasets-server direct URLs (public,
no auth) into data/e3_flare/<name>/<split>.parquet, then parsed to
parsed_<split>.parquet.

Usage:
    .venv-xpu\\Scripts\\python.exe -m src.eval_e3.load_flare --dataset acl
"""
from __future__ import annotations
import argparse
import io
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/e3_flare"

HF = "https://huggingface.co/api/datasets/TheFinAI/flare-sm-{name}/parquet/default/{split}/0.parquet"
DATASETS = {"acl": "acl", "bigdata": "bigdata", "cikm": "cikm"}
SPLITS = ("train", "valid", "test")

FEAT_COLS = ["open", "high", "low", "close", "adj-close",
             "inc-5", "inc-10", "inc-15", "inc-20", "inc-25", "inc-30"]

_TICKER = re.compile(r"\$([A-Za-z.\-]+)")
_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})\?")
# a price row: YYYY-MM-DD followed by exactly 11 comma-separated numeric fields
_PRICE_ROW = re.compile(
    r"^(\d{4}-\d{2}-\d{2})," + ",".join([r"(-?\d+\.?\d*)"] * 11) + r"\s*$"
)


def _parse_features(text: str) -> dict:
    """Most-recent-day price features, extracting only the numeric price rows
    embedded in `text` (which may also contain interleaved tweet lines)."""
    rows = []
    for line in text.splitlines():
        m = _PRICE_ROW.match(line.strip())
        if m:
            rows.append([float(x) for x in m.groups()[1:]])
    if not rows:
        return {c: float("nan") for c in FEAT_COLS}
    df = pd.DataFrame(rows, columns=FEAT_COLS)
    last = df.iloc[-1]
    out = {c: float(last[c]) for c in FEAT_COLS}
    # a couple of window aggregates so the baseline is not strawman-weak
    for w in ("close", "inc-5", "inc-10"):
        out[f"{w}_mean"] = float(df[w].mean())
        out[f"{w}_std"] = float(df[w].std())
    out["n_days"] = int(len(df))
    return out


def _extract_date(query: str) -> str | None:
    m = _DATE.search(query)
    if m:
        return m.group(1)
    m2 = re.search(r"at (\d{4}-\d{2}-\d{2})", query)
    return m2.group(1) if m2 else None


def _extract_ticker(query: str) -> str | None:
    m = _TICKER.search(query)
    return m.group(1).lower() if m else None


def download(name: str):
    raw = BASE / DATASETS[name]
    raw.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        dst = raw / f"{s}.parquet"
        if not dst.exists():
            url = HF.format(name=DATASETS[name], split=s)
            pd.read_parquet(url).to_parquet(dst)
            print(f"  downloaded {name}/{s}")
    return raw


def parse_split(raw_dir: Path, split: str) -> pd.DataFrame:
    df = pd.read_parquet(raw_dir / f"{split}.parquet")
    feats = df["text"].apply(_parse_features).apply(pd.Series)
    out = pd.DataFrame({
        "id": df["id"],
        "ticker": df["query"].apply(_extract_ticker),
        "date": df["query"].apply(_extract_date),
        "gold": df["gold"].astype(int),
        "answer": df["answer"],
        "query": df["query"],
    })
    out = pd.concat([out, feats], axis=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="acl", choices=list(DATASETS))
    args = ap.parse_args()

    raw = download(args.dataset)
    for s in SPLITS:
        parsed = parse_split(raw, s)
        parsed.to_parquet(raw / f"parsed_{s}.parquet", index=False)
        n_date = parsed["date"].notna().sum()
        print(f"{args.dataset}/{s}: {len(parsed)} rows | "
              f"date parsed {n_date}/{len(parsed)} | "
              f"ticker parsed {parsed['ticker'].notna().sum()} | "
              f"unique dates {parsed['date'].nunique()} | "
              f"gold {parsed['gold'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
