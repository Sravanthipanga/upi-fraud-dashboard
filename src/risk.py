"""Explainable, deterministic risk scoring for UPI transactions.

Every transaction is scored by seven transparent rules. Each rule that fires
adds points and appends a human-readable reason, so an analyst can always see
*why* a transaction was flagged. These are heuristics an analyst team might
ship as a first line of defence — not a machine-learning model.

Scores are capped at 100 and mapped to bands:
    Low < 25 · Medium 25-49 · High 50-74 · Critical >= 75
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RULE_POINTS = {
    "velocity_10min": 40,
    "odd_hour": 15,
    "new_device_high_value": 35,
    "high_amount": 20,
    "receiver_fan_in": 30,
    "sender_fan_out": 30,
    "micro_testing": 25,
}

RULE_DESCRIPTIONS = {
    "velocity_10min": "≥5 transactions from the same sender inside any 10-minute window",
    "odd_hour": "Transaction hour between 00:00 and 04:00",
    "new_device_high_value": "First-seen device and amount above ₹20,000",
    "high_amount": "Amount at or above the 99th percentile of genuine amounts",
    "receiver_fan_in": "Receiver collected from ≥8 unique senders on the same day",
    "sender_fan_out": "Sender paid ≥8 unique receivers within one hour",
    "micro_testing": "Transfer > ₹5,000 following ≥3 tiny (≤ ₹100) transfers from the same sender within the prior hour",
}

BANDS = ["Low", "Medium", "High", "Critical"]


def band_for(score: int) -> str:
    if score >= 75:
        return "Critical"
    if score >= 50:
        return "High"
    if score >= 25:
        return "Medium"
    return "Low"


def _window_counts_grouped(epochs_sorted: np.ndarray,
                           codes_sorted: np.ndarray,
                           window_seconds: int) -> np.ndarray:
    """Trailing-window event counts for rows sorted by (group, epoch).

    For each event, counts events of the same group in [t - window, t].
    Two-pointer sweep, O(n). Pure positional numpy — no index alignment.
    """
    n = len(epochs_sorted)
    counts = np.zeros(n, dtype=int)
    left = 0
    for i in range(n):
        if codes_sorted[i] != codes_sorted[left]:
            left = i
        while (codes_sorted[left] == codes_sorted[i]
               and epochs_sorted[i] - epochs_sorted[left] > window_seconds):
            left += 1
        counts[i] = i - left + 1
    return counts


def _group_burst_counts(df: pd.DataFrame, group_col: str,
                        window_seconds: int) -> np.ndarray:
    """Burst size through each event: the largest number of same-group events
    inside ANY window of `window_seconds` that contains the event.

    Computed on group/time-sorted rows in O(n): first the leading span L[j]
    (events in [t_j, t_j + window]) for every j, then for each event i the
    max of L[j] over all j <= i with t_i - t_j <= window, via a monotonic
    deque. This flags every member of a burst with the burst's full size.
    """
    from collections import deque

    epochs = df["_epoch"].to_numpy()
    codes, _ = pd.factorize(df[group_col])
    order = np.lexsort((epochs, codes))
    ep, cd = epochs[order], codes[order]
    n = len(ep)

    # Leading spans: L[j] = # events in [t_j, t_j + window].
    lead = np.zeros(n, dtype=int)
    right = 0
    for j in range(n):
        if right < j:
            right = j
        while (right + 1 < n and cd[right + 1] == cd[j]
               and ep[right + 1] - ep[j] <= window_seconds):
            right += 1
        if cd[right] != cd[j]:
            right = j
        lead[j] = right - j + 1

    # Sliding maximum of lead over the trailing window ending at i.
    burst_sorted = np.zeros(n, dtype=int)
    dq: deque[int] = deque()
    left = 0
    for i in range(n):
        if i > 0 and cd[i] != cd[i - 1]:
            dq.clear()
            left = i
        while ep[i] - ep[left] > window_seconds:
            if dq and dq[0] == left:
                dq.popleft()
            left += 1
        while dq and lead[dq[-1]] <= lead[i]:
            dq.pop()
        dq.append(i)
        burst_sorted[i] = lead[dq[0]]

    out = np.zeros(n, dtype=int)
    out[order] = burst_sorted
    return out


def _group_window_counts(df: pd.DataFrame, group_col: str,
                         window_seconds: int, direction: str = "trailing") -> np.ndarray:
    """Per-row trailing (or leading) window counts within each group.

    Returns an int array aligned to df's row positions. direction="leading"
    counts events in [t, t + window] instead — computed by sweeping each
    group in descending time order.
    """
    epochs = df["_epoch"].to_numpy()
    codes, _ = pd.factorize(df[group_col])
    if direction == "trailing":
        order = np.lexsort((epochs, codes))
    else:
        order = np.lexsort((-epochs, codes))
    counts_sorted = _window_counts_grouped(epochs[order], codes[order],
                                           window_seconds)
    out = np.zeros(len(df), dtype=int)
    out[order] = counts_sorted
    return out


def score_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with risk_score, risk_band and risk_reasons added."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    # Epoch seconds via total_seconds(): pandas 3 parses datetimes at
    # microsecond resolution, so astype('int64') // 10**9 would silently
    # collapse distinct timestamps into the same bucket.
    df["_epoch"] = ((df["timestamp"] - pd.Timestamp("1970-01-01"))
                    .dt.total_seconds().to_numpy(dtype="int64"))
    # Identity for behavioural rules: the full account key when available.
    # The masked display VPA (sender_vpa) is NOT an identity — many distinct
    # accounts share one masked string, so grouping by it would fabricate
    # velocity/fan-in signals. Fall back to the VPA column for hand-built
    # frames that carry no key columns.
    if "sender_key" not in df.columns:
        df["sender_key"] = df["sender_vpa"]
    if "receiver_key" not in df.columns:
        df["receiver_key"] = df["receiver_vpa"]
    n = len(df)

    scores = np.zeros(n, dtype=int)
    reasons: list[list[str]] = [[] for _ in range(n)]

    def fire(mask: pd.Series | np.ndarray, rule: str, reason_fn) -> None:
        idx = np.flatnonzero(np.asarray(mask))
        for i in idx:
            scores[i] += RULE_POINTS[rule]
            reasons[i].append(reason_fn(i))

    # Rule 1: velocity — sender burst within 10 minutes. The burst size
    # through each txn is the largest number of the sender's txns inside any
    # 10-minute window containing it, so every burst member is flagged.
    burst = _group_burst_counts(df, "sender_key", 600)
    fire(burst >= 5, "velocity_10min",
         lambda i: f"Velocity: {burst[i]} transactions in 10 minutes")

    # Rule 2: odd hour 00:00-04:00.
    hours = df["timestamp"].dt.hour.to_numpy()
    fire(hours < 4, "odd_hour", lambda i: "Odd hour (00:00–04:00) transaction")

    # Rule 3: new device + high value.
    ndhv = (df["is_new_device"].astype(bool)
            & (df["amount_inr"] > 20_000)).to_numpy()
    fire(ndhv, "new_device_high_value",
         lambda i: "New device with high-value transfer "
                   f"(₹{df['amount_inr'].iat[i]:,.0f})")

    # Rule 4: high amount — >= 99th percentile of genuine amounts.
    genuine = df.loc[~df["is_fraud"].astype(bool), "amount_inr"]
    p99 = float(genuine.quantile(0.99)) if len(genuine) else float("inf")
    fire((df["amount_inr"] >= p99).to_numpy(), "high_amount",
         lambda i: f"High amount (₹{df['amount_inr'].iat[i]:,.0f} ≥ 99th "
                   f"percentile ₹{p99:,.0f})")

    # Rule 5: receiver fan-in — >=8 unique senders to one receiver in a day.
    df["_date"] = df["timestamp"].dt.date
    fan_in = (df.groupby(["receiver_key", "_date"])["sender_key"]
              .transform("nunique").to_numpy())
    fire(fan_in >= 8, "receiver_fan_in",
         lambda i: f"Receiver fan-in: {fan_in[i]} unique senders in one day")

    # Rule 6: sender fan-out — >=8 unique receivers within one hour.
    df["_hour_bucket"] = df["timestamp"].dt.floor("h")
    fan_out = (df.groupby(["sender_key", "_hour_bucket"])["receiver_key"]
               .transform("nunique").to_numpy())
    fire(fan_out >= 8, "sender_fan_out",
         lambda i: f"Sender fan-out: {fan_out[i]} unique receivers in one hour")

    # Rule 7: micro-testing — big transfer after >=3 tiny ones within an hour.
    df["_is_tiny"] = (df["amount_inr"] <= 100).astype(int)
    epochs = df["_epoch"].to_numpy()
    codes, _ = pd.factorize(df["sender_key"])
    order = np.lexsort((epochs, codes))
    ep_s, cd_s, tiny_s = epochs[order], codes[order], df["_is_tiny"].to_numpy()[order]
    prior_tiny_sorted = np.zeros(len(df), dtype=int)
    left = 0
    running = 0
    for i in range(len(df)):
        if i > 0 and cd_s[i] != cd_s[i - 1]:
            left, running = i, 0
        while cd_s[left] == cd_s[i] and ep_s[i] - ep_s[left] > 3600:
            running -= tiny_s[left]
            left += 1
        prior_tiny_sorted[i] = running  # tiny txns strictly before this one
        running += tiny_s[i]
    prior_tiny = np.zeros(len(df), dtype=int)
    prior_tiny[order] = prior_tiny_sorted
    micro = ((prior_tiny >= 3) & (df["amount_inr"].to_numpy() > 5_000))
    fire(micro, "micro_testing",
         lambda i: f"Micro-testing: {prior_tiny[i]} tiny transfers in "
                   "the prior hour before a large transfer")

    scores = np.minimum(scores, 100)
    df["risk_score"] = scores.astype(int)
    df["risk_band"] = [band_for(s) for s in scores]
    df["risk_reasons"] = ["; ".join(r) for r in reasons]
    df = df.drop(columns=["_epoch", "_date", "_hour_bucket", "_is_tiny"])
    return df


def load_and_score(path: str | Path) -> pd.DataFrame:
    """Load the raw transactions CSV and return it risk-scored."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    # Empty CSV fields come back as NaN; the generator's contract is "".
    if "fraud_pattern" in df.columns:
        df["fraud_pattern"] = df["fraud_pattern"].fillna("")
    return score_transactions(df)


def rule_performance(df: pd.DataFrame) -> dict:
    """How well do the rules recover the injected fraud labels at High+?

    recall    = share of labelled-fraud rows scored High or Critical
    precision = share of High/Critical rows that are labelled fraud
    """
    high_plus = df["risk_band"].isin(["High", "Critical"])
    fraud = df["is_fraud"].astype(bool)
    tp = int((high_plus & fraud).sum())
    recall = tp / int(fraud.sum()) if fraud.sum() else 0.0
    precision = tp / int(high_plus.sum()) if high_plus.sum() else 0.0
    return {
        "flagged": int(high_plus.sum()),
        "fraud_total": int(fraud.sum()),
        "true_positives": tp,
        "recall_at_high_plus": round(float(recall), 4),
        "precision_at_high_plus": round(float(precision), 4),
    }
