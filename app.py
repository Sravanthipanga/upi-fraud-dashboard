"""UPI Fraud Detection Dashboard — Streamlit app.

Run:  streamlit run app.py

Data is 100% SYNTHETIC (see src/generate_data.py). If data/transactions.csv
does not exist yet, it is generated on first load so the app also works
out-of-the-box on Streamlit Community Cloud.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import analysis
from src.generate_data import generate
from src.risk import RULE_DESCRIPTIONS, RULE_POINTS, load_and_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "transactions.csv"

st.set_page_config(page_title="UPI Fraud Detection Dashboard",
                   page_icon="🛡️", layout="wide")


@st.cache_data(show_spinner="Loading and risk-scoring transactions…")
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        generate().to_csv(DATA_PATH, index=False)
    return load_and_score(DATA_PATH)


def inr(x: float) -> str:
    return f"₹{x:,.0f}"


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    start, end = st.session_state["date_range"]
    out = out[(out["timestamp"].dt.date >= start)
              & (out["timestamp"].dt.date <= end)]
    types = st.session_state["txn_types"]
    out = out[out["txn_type"].isin(types)]
    bands = st.session_state["risk_bands"]
    out = out[out["risk_band"].isin(bands)]
    pattern = st.session_state["fraud_pattern"]
    if pattern != "All labelled fraud":
        out = out[out["fraud_pattern"] == pattern]
    return out


# ------------------------------------------------------------------ header
st.title("🛡️ UPI Fraud Detection Dashboard")
st.caption(
    "⚠️ **Synthetic data** generated for portfolio/analysis purposes — "
    "no real transactions, people or accounts. Fraud labels are injected, "
    "documented patterns; risk scores come from explainable rules, not ML."
)

df = load_data()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Filters")
    min_d, max_d = df["timestamp"].dt.date.min(), df["timestamp"].dt.date.max()
    st.date_input("Date range", value=(min_d, max_d), min_value=min_d,
                  max_value=max_d, key="date_range")
    st.multiselect("Transaction type", ["P2P", "P2M"], default=["P2P", "P2M"],
                   key="txn_types")
    st.multiselect("Risk band", ["Low", "Medium", "High", "Critical"],
                   default=["Low", "Medium", "High", "Critical"],
                   key="risk_bands")
    patterns = ["All labelled fraud"] + sorted(
        df.loc[df["fraud_pattern"] != "", "fraud_pattern"].unique())
    st.selectbox("Fraud pattern (labelled)", patterns, key="fraud_pattern")
    st.divider()
    st.markdown("**About**\n\n40,000 synthetic UPI transactions over 30 days "
                "with ~1.5% injected fraud across 6 documented patterns. "
                "Every transaction is scored by 7 transparent rules.")

fdf = apply_filters(df)
kpi = analysis.kpi_summary(fdf)
full_kpi = analysis.kpi_summary(df)

# ------------------------------------------------------------------ KPIs
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total transactions", f"{kpi['total_transactions']:,}")
c2.metric("Fraud rate (labelled)", f"{kpi['fraud_rate']:.2%}",
          help=f"{kpi['fraud_count']} labelled-fraud transactions in view")
c3.metric("Value at risk (₹)", inr(kpi["value_at_risk_inr"]),
          help="Total value of High + Critical transactions in view")
c4.metric("Flagged for review", f"{kpi['flagged_count']:,}",
          help="High + Critical risk band transactions")
c5.metric("Rule recall @ High+",
          f"{full_kpi['rule_recall_at_high_plus']:.1%}",
          help="On the full dataset: share of labelled fraud the rules score "
               "High or Critical. Precision @ High+ = "
               f"{full_kpi['rule_precision_at_high_plus']:.1%}.")

st.divider()

# ------------------------------------------------------------------ charts
left, right = st.columns(2)
with left:
    st.subheader("Fraud rate by hour × weekday")
    hm = analysis.hourly_matrix(fdf)
    fig = go.Figure(data=go.Heatmap(
        z=hm.values, x=list(hm.columns), y=[f"{h:02d}:00" for h in hm.index],
        colorscale="Reds", colorbar_title="fraud %",
        hovertemplate="%{y} %{x}: %{z:.2f}% fraud<extra></extra>"))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_autorange="reversed")
    st.plotly_chart(fig, width="stretch")

with right:
    st.subheader("Daily trend — volume vs labelled fraud")
    trend = analysis.daily_trend(fdf)
    fig = go.Figure()
    fig.add_bar(x=trend["date"], y=trend["transactions"], name="Transactions",
                marker_color="#4C78A8")
    fig.add_scatter(x=trend["date"], y=trend["fraud"], name="Labelled fraud",
                    mode="lines+markers", line=dict(color="#E45756", width=2),
                    yaxis="y2")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title="Transactions",
                      yaxis2=dict(title="Fraud", overlaying="y", side="right"),
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch")

left2, right2 = st.columns(2)
with left2:
    st.subheader("Amount distribution — fraud vs genuine")
    plot_df = fdf.assign(group=fdf["is_fraud"].map({True: "Fraud", False: "Genuine"}))
    fig = px.histogram(plot_df, x="amount_inr", color="group", nbins=60,
                       barmode="overlay", opacity=0.65, log_x=True,
                       labels={"amount_inr": "Amount (₹, log scale)"},
                       color_discrete_map={"Fraud": "#E45756",
                                           "Genuine": "#4C78A8"})
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch")

with right2:
    st.subheader("Labelled fraud by pattern")
    pb = analysis.pattern_breakdown(fdf)
    if pb.empty:
        st.info("No labelled fraud in the current filter view.")
    else:
        fig = px.bar(pb, x="transactions", y="fraud_pattern", orientation="h",
                     text="transactions",
                     labels={"fraud_pattern": "Pattern", "transactions": "Txns"},
                     color_discrete_sequence=["#E45756"])
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")

st.subheader("Risk bands — where the rules concentrate fraud")
bands = analysis.risk_band_summary(fdf)
fig = px.bar(bands, x="risk_band", y="transactions", text="transactions",
             color="fraud_rate_pct", color_continuous_scale="Reds",
             labels={"risk_band": "Risk band", "transactions": "Transactions",
                     "fraud_rate_pct": "fraud %"},
             category_orders={"risk_band": ["Low", "Medium", "High", "Critical"]})
fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, width="stretch")
st.caption("Colour = labelled-fraud rate inside the band. Note the jump from "
           "Low/Medium to High/Critical — that separation is what makes the "
           "review queue useful.")

st.divider()

# ------------------------------------------------------------------ tables
st.subheader("Top risky receivers")
recv = analysis.top_risky_receivers(fdf)
st.dataframe(recv.rename(columns={
    "receiver_vpa": "Receiver (masked)", "transactions": "Txns",
    "fraud_transactions": "Labelled fraud", "flagged": "Flagged (High+)",
    "max_risk_score": "Max risk score", "total_value_inr": "Total value (₹)"}),
    width="stretch", hide_index=True)

st.subheader("📋 Case Review Queue — High & Critical, worst first")
queue = analysis.case_queue(fdf)
st.caption(f"{len(queue):,} transactions need review under the current filters.")
st.dataframe(
    queue[["txn_id", "timestamp", "amount_inr", "txn_type", "sender_vpa",
           "receiver_vpa", "city", "risk_score", "risk_band", "is_fraud",
           "risk_reasons"]].head(100).rename(columns={
               "txn_id": "Txn ID", "timestamp": "Timestamp",
               "amount_inr": "Amount (₹)", "txn_type": "Type",
               "sender_vpa": "Sender (masked)", "receiver_vpa": "Receiver (masked)",
               "city": "City", "risk_score": "Risk score",
               "risk_band": "Band", "is_fraud": "Labelled fraud",
               "risk_reasons": "Why flagged"}),
    width="stretch", hide_index=True)

st.subheader("🔍 Inspect a transaction")
if len(queue):
    pick = st.selectbox("Transaction", queue["txn_id"].head(200),
                        format_func=lambda t: f"{t} — "
                        f"{queue.loc[queue['txn_id'] == t, 'risk_band'].iloc[0]} "
                        f"({queue.loc[queue['txn_id'] == t, 'risk_score'].iloc[0]})")
    row = df[df["txn_id"] == pick].iloc[0]
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(f"""
| Field | Value |
|---|---|
| Transaction ID | `{row['txn_id']}` |
| Timestamp | {row['timestamp']} |
| Amount | {inr(row['amount_inr'])} |
| Type | {row['txn_type']} |
| Status | {row['status']} |
| Sender | `{row['sender_vpa']}` ({row['sender_bank']}) |
| Receiver | `{row['receiver_vpa']}` |
| City | {row['city']} |
| Device | `{row['device_id']}`{' (first-seen)' if row['is_new_device'] else ''} |
""")
    with d2:
        st.markdown(f"**Risk score:** {row['risk_score']} — **{row['risk_band']}**")
        st.markdown("**Why it scored:**")
        if row["risk_reasons"]:
            for reason in row["risk_reasons"].split("; "):
                st.markdown(f"- {reason}")
        else:
            st.markdown("- No rules fired.")
        st.markdown(f"**Labelled fraud:** {'Yes — ' + row['fraud_pattern'] if row['is_fraud'] else 'No'}")

st.divider()

# ------------------------------------------------------------------ docs
st.header("How the risk rules work")
rules_md = "| Rule | Points | Trigger |\n|---|---:|---|\n" + "\n".join(
    f"| `{name}` | +{RULE_POINTS[name]} | {RULE_DESCRIPTIONS[name]} |"
    for name in RULE_POINTS)
st.markdown(rules_md)
st.markdown(
    f"Scores cap at 100. Bands: **Low** < 25 · **Medium** 25–49 · "
    f"**High** 50–74 · **Critical** ≥ 75.\n\n"
    f"**Measured performance on this dataset (full, unfiltered):** the rules "
    f"flag {full_kpi['flagged_count']:,} transactions as High+; of the "
    f"{full_kpi['fraud_count']} labelled-fraud transactions they catch "
    f"{full_kpi['rule_recall_at_high_plus']:.1%} (**recall**), and "
    f"{full_kpi['rule_precision_at_high_plus']:.1%} of flagged transactions "
    f"are labelled fraud (**precision**). Low precision at high recall is the "
    f"classic fraud-queue trade-off: analysts would rather review false "
    f"alarms than miss real fraud.")

st.header("Methodology & data dictionary")
st.markdown("""
**Methodology.** A seeded generator (seed 42) creates 40,000 synthetic UPI
transactions across 30 days from pools of 3,000 sender accounts and 1,200
receiver accounts. Genuine amounts are lognormal (median ≈ ₹450); timestamps
follow a realistic daily rhythm. Fraud is then *injected* as six documented
behavioural patterns, so every fraud row has known ground truth that the
risk rules are scored against. `sender_key` / `receiver_key` are the full
synthetic account identities; the VPA columns are masked for display because
that is how real dashboards must present them — behavioural rules must never
group by a masked value (many accounts share one masked string).

| Column | Meaning |
|---|---|
| `txn_id` | 12-digit UTR-style synthetic reference |
| `timestamp` | Transaction time (2026-10-14 → 2026-11-12) |
| `amount_inr` | Amount in Indian Rupees |
| `txn_type` | P2P (person→person) or P2M (person→merchant) |
| `sender_vpa` / `receiver_vpa` | **Masked** Virtual Payment Address, e.g. `ra•••@okhdfcbank` |
| `sender_key` / `receiver_key` | Full synthetic account identity (analytics key) |
| `sender_bank` | Sender's bank |
| `device_id` | Device the payment was initiated from |
| `is_new_device` | True if the device was never seen before for this sender |
| `city` | Sender's city |
| `status` | SUCCESS / FAILED (some fraud attempts fail = blocked) |
| `is_fraud` | Ground-truth label (only knowable because data is synthetic) |
| `fraud_pattern` | Which injected pattern produced the row (blank if genuine) |
| `risk_score` / `risk_band` / `risk_reasons` | Added by the explainable rule engine |
""")

st.header("Limitations")
st.markdown("""
- **Synthetic data.** Real UPI fraud is messier, adversarial and constantly
  adapting; these patterns are deliberately overt so rule performance can be
  measured against ground truth. Numbers here do **not** estimate real-world
  detection rates.
- **Rules, not ML.** The scorer is a transparent heuristic ruleset — a
  feature for explainability, a limitation for subtle fraud. A natural next
  step is benchmarking a gradient-boosted model against these rules.
- **Labels are injected patterns only.** "Genuine" rows are clean by
  construction; real datasets have unlabelled fraud hiding in the negatives.
- **Masked VPAs collide.** ~186 distinct masked sender strings represent
  3,000 accounts. Any analysis grouping by the masked value alone would be
  wrong — this project keys behaviour on the synthetic account IDs instead.
""")
