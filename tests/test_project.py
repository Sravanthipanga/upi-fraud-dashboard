"""Test suite for the UPI fraud dashboard project.

Covers: data generation reproducibility & masking, every risk rule on
crafted frames, scoring invariants, analysis correctness on a hand-checked
fixture, DuckDB SQL execution, and end-to-end rule performance.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import analysis  # noqa: E402
from src.generate_data import generate  # noqa: E402
from src.risk import band_for, rule_performance, score_transactions  # noqa: E402

BASE = pd.Timestamp("2026-11-01 10:00:00")


def make_txn(sender="alice@okhdfc", receiver="shop@paytm", amount=500.0,
             ts=BASE, new_device=False, is_fraud=False, pattern="",
             status="SUCCESS") -> dict:
    return {"txn_id": "100000000001", "timestamp": ts, "amount_inr": amount,
            "txn_type": "P2P", "sender_vpa": "al•••@okhdfcbank",
            "receiver_vpa": "sh•••@paytm", "sender_key": sender,
            "receiver_key": receiver, "sender_bank": "HDFC Bank",
            "device_id": "DEV-1", "is_new_device": new_device,
            "city": "Mumbai", "status": status, "is_fraud": is_fraud,
            "fraud_pattern": pattern}


def frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for i, r in enumerate(rows):
        df.loc[i, "txn_id"] = f"2000000000{i:02d}"
    return df


# ---------------------------------------------------------------- generation
@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return generate()


def test_generation_shape_and_reproducibility(data):
    assert len(data) == 40_000
    again = generate()
    assert data["amount_inr"].sum() == again["amount_inr"].sum()
    assert data["is_fraud"].sum() == again["is_fraud"].sum()
    rate = data["is_fraud"].mean()
    assert 0.012 <= rate <= 0.018, f"fraud rate {rate:.3f} out of range"


def test_every_fraud_pattern_present(data):
    patterns = set(data.loc[data["is_fraud"], "fraud_pattern"])
    assert patterns == {"velocity", "odd_hour", "new_device_high_value",
                        "mule_fan_in", "mule_fan_out", "micro_testing"}
    assert (data.loc[~data["is_fraud"], "fraud_pattern"] == "").all()


def test_no_unmasked_vpa(data):
    vpa_re = re.compile(r"^[a-z]{2}•••@(okhdfcbank|okaxis|oksbi|paytm|ybl|okicici)$")
    for col in ["sender_vpa", "receiver_vpa"]:
        assert data[col].str.match(vpa_re).all(), f"unmasked VPA in {col}"
    # Identity keys must remain fine-grained (masking must not be the key).
    assert data["sender_key"].nunique() > 2_500
    assert data["receiver_key"].nunique() > 1_000


def test_status_and_failed_fraud_exist(data):
    assert set(data["status"]) <= {"SUCCESS", "FAILED"}
    assert abs((data["status"] == "SUCCESS").mean() - 0.96) < 0.03
    assert (data["is_fraud"] & (data["status"] == "FAILED")).sum() > 0


# ---------------------------------------------------------------- risk rules
def test_rule_velocity():
    rows = [make_txn(ts=BASE + pd.Timedelta(seconds=60 * i)) for i in range(6)]
    scored = score_transactions(frame(rows))
    assert (scored["risk_score"] >= 40).all()
    assert scored["risk_reasons"].str.contains("Velocity").all()


def test_rule_odd_hour():
    scored = score_transactions(frame([make_txn(ts=pd.Timestamp("2026-11-01 02:30"))]))
    assert scored.loc[0, "risk_score"] >= 15
    assert "Odd hour" in scored.loc[0, "risk_reasons"]
    noon = score_transactions(frame([make_txn(ts=pd.Timestamp("2026-11-01 12:30"))]))
    assert "Odd hour" not in noon.loc[0, "risk_reasons"]


def test_rule_new_device_high_value():
    scored = score_transactions(frame([make_txn(amount=25_000, new_device=True)]))
    assert scored.loc[0, "risk_score"] >= 35
    assert "New device" in scored.loc[0, "risk_reasons"]


def test_rule_high_amount():
    rows = [make_txn(amount=100.0) for _ in range(200)]
    rows.append(make_txn(amount=90_000.0, ts=BASE + pd.Timedelta(days=1)))
    scored = score_transactions(frame(rows))
    big = scored[scored["amount_inr"] == 90_000].iloc[0]
    assert "High amount" in big["risk_reasons"]


def test_rule_receiver_fan_in():
    rows = [make_txn(sender=f"sender{i}@okaxis", receiver="mule@paytm",
                     ts=BASE + pd.Timedelta(minutes=30 * i)) for i in range(9)]
    scored = score_transactions(frame(rows))
    assert scored["risk_reasons"].str.contains("fan-in").all()
    assert (scored["risk_score"] >= 30).all()


def test_rule_sender_fan_out():
    rows = [make_txn(receiver=f"recv{i}@ybl",
                     ts=BASE + pd.Timedelta(minutes=5 * i)) for i in range(9)]
    scored = score_transactions(frame(rows))
    assert scored["risk_reasons"].str.contains("fan-out").all()


def test_rule_micro_testing():
    rows = [make_txn(amount=10.0, ts=BASE + pd.Timedelta(minutes=5 * i))
            for i in range(3)]
    rows.append(make_txn(amount=9_000.0, ts=BASE + pd.Timedelta(minutes=40)))
    scored = score_transactions(frame(rows))
    big = scored[scored["amount_inr"] == 9_000].iloc[0]
    assert "Micro-testing" in big["risk_reasons"]
    assert big["risk_score"] >= 25


def test_score_cap_and_bands():
    # One transaction engineered to trip nearly everything at once.
    rows = [make_txn(amount=50.0, ts=pd.Timestamp("2026-11-01 01:00"))]
    rows += [make_txn(amount=25_000, new_device=True,
                      ts=pd.Timestamp("2026-11-01 01:05"))]
    scored = score_transactions(frame(rows))
    assert (scored["risk_score"] <= 100).all()
    assert band_for(0) == "Low" and band_for(24) == "Low"
    assert band_for(25) == "Medium" and band_for(49) == "Medium"
    assert band_for(50) == "High" and band_for(74) == "High"
    assert band_for(75) == "Critical" and band_for(100) == "Critical"


def test_masked_vpa_collisions_do_not_fabricate_velocity():
    # 6 DIFFERENT accounts that share one masked display string must NOT be
    # treated as one sender bursting.
    rows = [make_txn(sender=f"user{i}@okhdfc", ts=BASE + pd.Timedelta(seconds=30 * i))
            for i in range(6)]
    scored = score_transactions(frame(rows))
    assert not scored["risk_reasons"].str.contains("Velocity").any()


# ---------------------------------------------------------------- analysis
def _fixture() -> pd.DataFrame:
    rows = [
        make_txn(amount=100.0, is_fraud=False),
        make_txn(amount=200.0, is_fraud=False, ts=BASE + pd.Timedelta(hours=1)),
        make_txn(amount=300.0, is_fraud=True, pattern="velocity",
                 ts=BASE + pd.Timedelta(hours=2)),
        make_txn(amount=400.0, is_fraud=True, pattern="odd_hour",
                 ts=BASE + pd.Timedelta(hours=3)),
        make_txn(amount=500.0, is_fraud=False, ts=BASE + pd.Timedelta(hours=4)),
    ]
    df = frame(rows)
    df["risk_score"] = [0, 10, 80, 60, 30]
    df["risk_band"] = ["Low", "Low", "Critical", "High", "Medium"]
    df["risk_reasons"] = ["", "", "Velocity: 6 transactions in 10 minutes",
                          "Odd hour (00:00–04:00) transaction", ""]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def test_analysis_kpis_match_hand_computed():
    kpi = analysis.kpi_summary(_fixture())
    assert kpi["total_transactions"] == 5
    assert kpi["fraud_count"] == 2
    assert kpi["fraud_rate"] == pytest.approx(0.4)
    assert kpi["flagged_count"] == 2
    assert kpi["value_at_risk_inr"] == pytest.approx(700.0)
    assert kpi["rule_recall_at_high_plus"] == pytest.approx(1.0)
    assert kpi["rule_precision_at_high_plus"] == pytest.approx(1.0)


def test_case_queue_only_high_plus_sorted():
    q = analysis.case_queue(_fixture())
    assert list(q["risk_band"]) == ["Critical", "High"]
    assert list(q["risk_score"]) == [80, 60]


def test_pattern_breakdown_fixture():
    pb = analysis.pattern_breakdown(_fixture())
    assert set(pb["fraud_pattern"]) == {"velocity", "odd_hour"}
    assert pb["transactions"].sum() == 2


# ---------------------------------------------------------------- SQL
def _ensure_csv():
    csv = ROOT / "data" / "transactions.csv"
    if not csv.exists():
        subprocess.run([sys.executable, "-m", "src.generate_data"],
                       cwd=ROOT, check=True, capture_output=True)
    return csv


def test_duckdb_queries():
    _ensure_csv()
    con = duckdb.connect()
    rate = con.execute(
        "SELECT ROUND(100.0 * AVG(CASE WHEN is_fraud THEN 1.0 ELSE 0.0 END), 3) "
        "FROM read_csv('data/transactions.csv')").fetchone()[0]
    assert 1.2 <= rate <= 1.8
    by_hour = con.execute(
        "SELECT HOUR(timestamp), COUNT(*) FROM read_csv('data/transactions.csv') "
        "GROUP BY 1 ORDER BY 1").fetchall()
    assert len(by_hour) == 24 and sum(r[1] for r in by_hour) == 40_000
    top = con.execute(
        "SELECT receiver_key, SUM(amount_inr) v FROM read_csv('data/transactions.csv') "
        "WHERE is_fraud GROUP BY 1 ORDER BY v DESC LIMIT 10").fetchall()
    assert len(top) == 10 and all(r[1] > 0 for r in top)
    con.close()


# ---------------------------------------------------------------- end to end
def test_rule_performance_on_generated_data(data):
    scored = score_transactions(data)
    perf = rule_performance(scored)
    assert perf["recall_at_high_plus"] >= 0.5, (
        f"recall @High+ too low: {perf['recall_at_high_plus']}")
    assert perf["precision_at_high_plus"] > 0.30
    assert perf["flagged"] < len(data) * 0.05  # queue stays reviewable
