"""Pure analysis functions over a risk-scored transactions DataFrame.

Each function takes the scored DataFrame produced by src.risk and returns
plain dicts / DataFrames that the Streamlit app (or a notebook) can render.
"""

from __future__ import annotations

import pandas as pd

HIGH_PLUS = ["High", "Critical"]


def kpi_summary(df: pd.DataFrame) -> dict:
    """Headline numbers for the dashboard KPI row."""
    fraud = df["is_fraud"].astype(bool)
    high_plus = df["risk_band"].isin(HIGH_PLUS)
    tp = int((high_plus & fraud).sum())
    return {
        "total_transactions": int(len(df)),
        "total_value_inr": float(df["amount_inr"].sum()),
        "fraud_count": int(fraud.sum()),
        "fraud_rate": float(fraud.mean()) if len(df) else 0.0,
        "fraud_value_inr": float(df.loc[fraud, "amount_inr"].sum()),
        "flagged_count": int(high_plus.sum()),
        "value_at_risk_inr": float(df.loc[high_plus, "amount_inr"].sum()),
        "rule_recall_at_high_plus": (tp / int(fraud.sum())) if fraud.sum() else 0.0,
        "rule_precision_at_high_plus": (tp / int(high_plus.sum())) if high_plus.sum() else 0.0,
        "critical_count": int((df["risk_band"] == "Critical").sum()),
        "failed_count": int((df["status"] == "FAILED").sum()),
    }


def hourly_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """24 x 7 matrix (hour x weekday) of labelled fraud *rate* (%).

    Indexed by hour 0-23, columns Mon..Sun. Cells are NaN where a bucket had
    no transactions at all.
    """
    d = df.copy()
    d["hour"] = d["timestamp"].dt.hour
    d["weekday"] = d["timestamp"].dt.dayofweek  # Mon=0
    grouped = d.groupby(["hour", "weekday"])["is_fraud"].agg(["sum", "count"])
    rate = (grouped["sum"] / grouped["count"] * 100).unstack("weekday")
    rate = rate.reindex(index=range(24), columns=range(7))
    rate.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    rate.index.name = "hour"
    return rate


def hourly_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Transactions and labelled fraud by hour of day (0-23)."""
    d = df.copy()
    d["hour"] = d["timestamp"].dt.hour
    out = d.groupby("hour").agg(
        transactions=("txn_id", "count"),
        fraud=("is_fraud", "sum"),
    ).reindex(range(24), fill_value=0)
    out["fraud_rate_pct"] = out["fraud"] / out["transactions"].clip(lower=1) * 100
    return out.reset_index()


def daily_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Per-day totals: transactions, labelled fraud, flagged (High+), value."""
    d = df.copy()
    d["date"] = d["timestamp"].dt.date
    out = d.groupby("date").agg(
        transactions=("txn_id", "count"),
        fraud=("is_fraud", "sum"),
        flagged=("risk_band", lambda s: int(s.isin(HIGH_PLUS).sum())),
        value_inr=("amount_inr", "sum"),
        fraud_value_inr=("amount_inr",
                         lambda s: float(s[d.loc[s.index, "is_fraud"]
                                           .astype(bool)].sum())),
    ).reset_index()
    out["fraud_rate_pct"] = out["fraud"] / out["transactions"] * 100
    return out


AMOUNT_BINS = [0, 100, 500, 1_000, 5_000, 10_000, 20_000, 50_000, float("inf")]
AMOUNT_LABELS = ["≤₹100", "₹101–500", "₹501–1k", "₹1k–5k", "₹5k–10k",
                 "₹10k–20k", "₹20k–50k", ">₹50k"]


def amount_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Transaction counts by amount band, split genuine vs labelled fraud."""
    d = df.copy()
    d["bin"] = pd.cut(d["amount_inr"], bins=AMOUNT_BINS, labels=AMOUNT_LABELS)
    out = (d.groupby(["bin", "is_fraud"], observed=True)["txn_id"]
           .count().unstack(fill_value=0))
    out = out.rename(columns={False: "genuine", True: "fraud"})
    for col in ["genuine", "fraud"]:
        if col not in out:
            out[col] = 0
    out["total"] = out["genuine"] + out["fraud"]
    out["fraud_rate_pct"] = out["fraud"] / out["total"].clip(lower=1) * 100
    return out.reset_index().rename(columns={"bin": "amount_band"})


def pattern_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Labelled fraud by injected pattern: count, share and ₹ value."""
    fraud = df[df["is_fraud"].astype(bool) & (df["fraud_pattern"] != "")]
    if fraud.empty:
        return pd.DataFrame(columns=["fraud_pattern", "transactions",
                                     "value_inr", "share_pct"])
    out = (fraud.groupby("fraud_pattern")
           .agg(transactions=("txn_id", "count"),
                value_inr=("amount_inr", "sum"))
           .reset_index())
    out["share_pct"] = out["transactions"] / out["transactions"].sum() * 100
    return out.sort_values("transactions", ascending=False).reset_index(drop=True)


def _top_risky(df: pd.DataFrame, col: str, n: int) -> pd.DataFrame:
    out = (df.groupby(col)
           .agg(transactions=("txn_id", "count"),
                fraud_transactions=("is_fraud", "sum"),
                flagged=("risk_band", lambda s: int(s.isin(HIGH_PLUS).sum())),
                max_risk_score=("risk_score", "max"),
                total_value_inr=("amount_inr", "sum"))
           .reset_index())
    out = out.sort_values(["fraud_transactions", "max_risk_score", "flagged"],
                          ascending=False).head(n)
    return out.reset_index(drop=True)


def top_risky_receivers(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Receivers ranked by labelled-fraud count, then peak risk score."""
    return _top_risky(df, "receiver_vpa", n)


def top_risky_senders(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Senders ranked by labelled-fraud count, then peak risk score."""
    return _top_risky(df, "sender_vpa", n)


def case_queue(df: pd.DataFrame) -> pd.DataFrame:
    """Transactions an analyst should review first: High/Critical, worst first."""
    q = df[df["risk_band"].isin(HIGH_PLUS)]
    return q.sort_values(["risk_score", "amount_inr"],
                         ascending=[False, False]).reset_index(drop=True)


def risk_band_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Counts and labelled-fraud rate per risk band (for the band chart)."""
    out = (df.groupby("risk_band")
           .agg(transactions=("txn_id", "count"),
                fraud=("is_fraud", "sum"),
                value_inr=("amount_inr", "sum"))
           .reindex(["Low", "Medium", "High", "Critical"])
           .fillna(0).reset_index())
    out["fraud_rate_pct"] = out["fraud"] / out["transactions"].clip(lower=1) * 100
    return out
