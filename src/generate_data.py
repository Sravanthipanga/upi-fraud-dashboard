"""Synthetic UPI transaction generator.

Generates 40,000 SYNTHETIC UPI transactions over 30 consecutive days ending
2026-11-12, with documented fraud patterns injected at ~1.2-1.8% of rows.

This data is entirely fictional. It exists so the fraud-detection dashboard,
risk rules and SQL queries can be developed and demonstrated without touching
any real person's financial data. VPAs are masked (e.g. "ra•••@okhdfcbank")
and names / devices / cities are drawn from small fake pools.

Run:  python -m src.generate_data
Writes: data/transactions.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_TXNS = 40_000
END_DATE = pd.Timestamp("2026-11-12 23:59:59")
START_DATE = END_DATE - pd.Timedelta(days=30) + pd.Timedelta(seconds=1)

HANDLES = ["okhdfcbank", "okaxis", "oksbi", "paytm", "ybl", "okicici"]
BANKS = ["HDFC Bank", "Axis Bank", "State Bank of India", "ICICI Bank",
         "Kotak Mahindra", "Punjab National Bank", "Bank of Baroda"]
CITIES = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Pune",
          "Kolkata", "Ahmedabad", "Jaipur", "Lucknow", "Indore", "Kochi",
          "Surat", "Nagpur", "Coimbatore", "Patna", "Bhopal", "Chandigarh"]
FIRST = ["aarav", "vivaan", "aditya", "arjun", "sai", "reyansh", "krishna",
         "ananya", "diya", "aadhya", "myra", "sara", "ira", "priya", "rahul",
         "rohit", "sneha", "kavya", "manish", "pooja", "vikram", "neha",
         "amit", "rani", "kiran", "divya", "sanjay", "meera", "ravi", "anita",
         "deepak", "shreya", "nikhil", "ritu", "varun", "isha", "karan",
         "lakshmi", "mohan", "geeta"]
LAST_INITIALS = list("abcdefghjkmnprstuvy")
MERCHANTS = ["swiggy", "zomato", "amazon", "flipkart", "bigbasket", "blinkit",
             "jiomart", "irctc", "makemytrip", "ola", "uber", "rapido",
             "paytm mall", "myntra", "nykaa", "bookmyshow", "dominos",
             "starbucks", "dmart", "apollo pharmacy", "netmeds", "licious",
             "urban company", "curefit", "practo"]

# NOTE on activity levels: with 40k transactions spread over ~3,000 senders,
# a typical genuine sender makes only ~13 transactions in 30 days, so bursts,
# fan-in/fan-out and micro-test sequences stand out cleanly — as they should:
# the injected patterns are deliberately overt so the explainable rules can be
# evaluated against known ground truth.
N_SENDERS = 3_000
N_RECEIVERS = 1_200

COLUMNS = ["txn_id", "timestamp", "amount_inr", "txn_type", "sender_vpa",
           "receiver_vpa", "sender_key", "receiver_key", "sender_bank",
           "device_id", "is_new_device", "city", "status", "is_fraud",
           "fraud_pattern"]


def _masked_vpa(local: str, rng: np.random.Generator) -> str:
    """Mask a fake VPA local part for display: keep first 2 chars, then •••.

    The handle (bank/app suffix) is random per account, so the full masked
    string identifies one synthetic account. NOTE: masking alone destroys
    identity granularity — never group risk features by the masked value.
    """
    handle = rng.choice(HANDLES)
    return f"{local[:2]}•••@{handle}"


def _account_key(local: str, handle: str) -> str:
    """Internal identity key for one synthetic account (local + handle).

    Distinct accounts may share the same masked display VPA; this full key
    (unmasked local part + handle) is unique per account and is what sender /
    receiver identity means throughout the pipeline.
    """
    return f"{local}@{handle}"


def _person_local(rng: np.random.Generator) -> str:
    name = rng.choice(FIRST)
    if rng.random() < 0.7:
        name += rng.choice(LAST_INITIALS)
    if rng.random() < 0.4:
        name += str(rng.integers(1, 999))
    return name


def _build_pools(rng: np.random.Generator):
    sender_locals = set()
    while len(sender_locals) < N_SENDERS:
        sender_locals.add(_person_local(rng))
    sender_locals = sorted(sender_locals)
    # Each account gets ONE consistent handle: its identity key is unique even
    # though many accounts share the same masked display string.
    sender_handles = [str(h) for h in rng.choice(HANDLES, size=N_SENDERS)]
    sender_keys = [_account_key(s, h) for s, h in zip(sender_locals, sender_handles)]
    sender_vpas = [f"{s[:2]}•••@{h}" for s, h in zip(sender_locals, sender_handles)]
    sender_banks = list(rng.choice(BANKS, size=N_SENDERS))
    sender_cities = list(rng.choice(CITIES, size=N_SENDERS))
    # Each sender has 1-2 habitual devices.
    sender_devices = []
    for i in range(N_SENDERS):
        devs = [f"DEV-{i:05d}-A"]
        if rng.random() < 0.3:
            devs.append(f"DEV-{i:05d}-B")
        sender_devices.append(devs)

    receiver_locals = set()
    while len(receiver_locals) < N_RECEIVERS:
        if rng.random() < 0.55:
            receiver_locals.add(rng.choice(MERCHANTS).replace(" ", ""))
        else:
            receiver_locals.add(_person_local(rng))
    receiver_locals = sorted(receiver_locals)
    recv_handles = [str(h) for h in rng.choice(HANDLES, size=N_RECEIVERS)]
    receiver_keys = [_account_key(r, h)
                     for r, h in zip(receiver_locals, recv_handles)]
    receiver_vpas = [f"{r[:2]}•••@{h}"
                     for r, h in zip(receiver_locals, recv_handles)]
    return (sender_vpas, sender_banks, sender_cities, sender_devices,
            receiver_vpas, sender_keys, receiver_keys)


# Hour weights for genuine traffic: peaks morning commute, lunch, evening.
_HOUR_W = np.array([2, 1, 1, 1, 1, 2, 4, 7, 10, 11, 11, 11, 12, 11, 10, 10,
                    10, 11, 12, 12, 11, 9, 6, 4], dtype=float)
_HOUR_W = _HOUR_W / _HOUR_W.sum()


def _random_timestamps(rng: np.random.Generator, n: int) -> pd.DatetimeIndex:
    days = rng.integers(0, 30, size=n)
    hours = rng.choice(24, size=n, p=_HOUR_W)
    mins = rng.integers(0, 60, size=n)
    secs = rng.integers(0, 60, size=n)
    base = START_DATE.normalize()
    ts = (base + pd.to_timedelta(days, unit="D") + pd.to_timedelta(hours, unit="h")
          + pd.to_timedelta(mins, unit="m") + pd.to_timedelta(secs, unit="s"))
    return pd.DatetimeIndex(ts)


def _utr(rng: np.random.Generator, n: int) -> np.ndarray:
    return np.array([f"{rng.integers(10**11, 10**12)}" for _ in range(n)])


def _genuine(rng: np.random.Generator, n: int, pools) -> pd.DataFrame:
    (sender_vpas, sender_banks, sender_cities, sender_devices, receiver_vpas,
     sender_keys, receiver_keys) = pools
    s_idx = rng.integers(0, N_SENDERS, size=n)
    r_idx = rng.integers(0, N_RECEIVERS, size=n)
    txn_type = rng.choice(["P2P", "P2M"], size=n, p=[0.55, 0.45])
    amounts = rng.lognormal(mean=np.log(450), sigma=1.05, size=n)
    amounts = np.clip(np.round(amounts), 1, 95_000)
    devices, new_dev = [], []
    for i in s_idx:
        devs = sender_devices[i]
        devices.append(devs[0] if len(devs) == 1 else devs[rng.integers(0, len(devs))])
        new_dev.append(False)
    df = pd.DataFrame({
        "txn_id": _utr(rng, n),
        "timestamp": _random_timestamps(rng, n),
        "amount_inr": amounts.astype(float),
        "txn_type": txn_type,
        "sender_vpa": [sender_vpas[i] for i in s_idx],
        "receiver_vpa": [receiver_vpas[j] for j in r_idx],
        "sender_key": [sender_keys[i] for i in s_idx],
        "receiver_key": [receiver_keys[j] for j in r_idx],
        "sender_bank": [sender_banks[i] for i in s_idx],
        "device_id": devices,
        "is_new_device": new_dev,
        "city": [sender_cities[i] for i in s_idx],
        "status": rng.choice(["SUCCESS", "FAILED"], size=n, p=[0.96, 0.04]),
        "is_fraud": False,
        "fraud_pattern": "",
    })
    return df


def _fraud_frame(rows: list[dict], rng: np.random.Generator) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS)
    if len(df):
        df["txn_id"] = _utr(rng, len(df))
    return df


def _base_ts(rng: np.random.Generator, hour_lo=0, hour_hi=24) -> pd.Timestamp:
    day = int(rng.integers(0, 30))
    hour = int(rng.integers(hour_lo, hour_hi))
    base = START_DATE.normalize()
    return (base + pd.Timedelta(days=day, hours=hour,
                                minutes=int(rng.integers(0, 60)),
                                seconds=int(rng.integers(0, 60))))


def inject_fraud(rng: np.random.Generator, pools) -> pd.DataFrame:
    """Inject the documented fraud patterns (~620 rows, ~1.5% of volume).

    Patterns are injected as whole sequences/groups and are NEVER row-sampled
    afterwards: randomly dropping individual rows would decapitate bursts and
    fan-in days, destroying the very structure the rules are meant to detect.
    """
    (sender_vpas, sender_banks, sender_cities, sender_devices, receiver_vpas,
     sender_keys, receiver_keys) = pools
    rows: list[dict] = []

    def add(ts, amount, stype, s_i, r_i, pattern, device=None, new_dev=False,
            fail_p=0.15):
        rows.append({
            "txn_id": "", "timestamp": ts, "amount_inr": float(round(amount, 2)),
            "txn_type": stype, "sender_vpa": sender_vpas[s_i],
            "receiver_vpa": receiver_vpas[r_i],
            "sender_key": sender_keys[s_i], "receiver_key": receiver_keys[r_i],
            "sender_bank": sender_banks[s_i],
            "device_id": device if device is not None else sender_devices[s_i][0],
            "is_new_device": new_dev, "city": sender_cities[s_i],
            "status": "FAILED" if rng.random() < fail_p else "SUCCESS",
            "is_fraud": True, "fraud_pattern": pattern,
        })

    # (a) velocity — bursts of 6-12 txns from one sender within 10 minutes.
    # Amounts are elevated (median ~₹6,000 vs the genuine ~₹450): draining an
    # account quickly means moving real money, so bursts typically also trip
    # the high-amount rule — that stacking is what pushes them to High+.
    for _ in range(24):
        s = int(rng.integers(0, N_SENDERS)); r = int(rng.integers(0, N_RECEIVERS))
        t0 = _base_ts(rng, 6, 24)
        for k in range(int(rng.integers(6, 13))):
            add(t0 + pd.Timedelta(seconds=int(rng.integers(0, 600))),
                rng.lognormal(np.log(6000), 0.9), "P2P", s, r, "velocity")

    # (b) odd_hour — bursts of 4-6 txns between 00:00 and 04:00, inside
    # ~10 minutes, at elevated amounts (median ~₹8,000): night-time account
    # takeover behaviour that stacks odd-hour + velocity + high-amount.
    for _ in range(22):
        s = int(rng.integers(0, N_SENDERS)); r = int(rng.integers(0, N_RECEIVERS))
        t0 = _base_ts(rng, 0, 4)
        for k in range(int(rng.integers(4, 7))):
            add(t0 + pd.Timedelta(seconds=int(rng.integers(0, 540))),
                rng.lognormal(np.log(8000), 0.9), "P2P", s, r, "odd_hour")

    # (c) new_device_high_value — first-seen device + amount > ₹20,000.
    for _ in range(50):
        s = int(rng.integers(0, N_SENDERS)); r = int(rng.integers(0, N_RECEIVERS))
        add(_base_ts(rng), rng.uniform(21_000, 95_000), "P2P", s, r,
            "new_device_high_value", device=f"DEV-NEW-{s:05d}", new_dev=True,
            fail_p=0.25)

    # (d) mule_fan_in — one receiver collects from >=10 unique senders in a day.
    for _ in range(6):
        r = int(rng.integers(0, N_RECEIVERS))
        t0 = _base_ts(rng, 7, 22)
        senders = rng.choice(N_SENDERS, size=int(rng.integers(11, 16)), replace=False)
        for s in senders:
            add(t0 + pd.Timedelta(minutes=int(rng.integers(0, 600))),
                rng.uniform(4_000, 14_900), "P2P", int(s), r, "mule_fan_in")

    # (e) mule_fan_out — one sender pays >=10 unique receivers within an hour.
    for _ in range(6):
        s = int(rng.integers(0, N_SENDERS))
        t0 = _base_ts(rng, 7, 22)
        recvs = rng.choice(N_RECEIVERS, size=int(rng.integers(11, 15)), replace=False)
        for r in recvs:
            add(t0 + pd.Timedelta(minutes=int(rng.integers(0, 55))),
                rng.uniform(5_000, 19_000), "P2P", s, int(r), "mule_fan_out")

    # (f) micro_testing — 4-5 tiny round amounts then one large transfer, the
    # whole sequence inside ~10 minutes (so it also forms a velocity burst;
    # the cash-out transfer additionally trips the micro-testing rule).
    # NOTE: the tiny probes score low individually — realistic, since a ₹10
    # transfer looks harmless alone; the sequence is caught at cash-out.
    for _ in range(20):
        s = int(rng.integers(0, N_SENDERS)); r = int(rng.integers(0, N_RECEIVERS))
        t0 = _base_ts(rng, 6, 23)
        for k in range(int(rng.integers(4, 6))):
            add(t0 + pd.Timedelta(seconds=int(rng.integers(0, 420))),
                float(rng.choice([1, 5, 10, 50, 100])), "P2P", s, r,
                "micro_testing", fail_p=0.05)
        add(t0 + pd.Timedelta(seconds=int(rng.integers(421, 540))),
            rng.uniform(15_000, 80_000), "P2P", s, r, "micro_testing",
            fail_p=0.2)

    return _fraud_frame(rows, rng)


def generate(seed: int = SEED, n_txns: int = N_TXNS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pools = _build_pools(rng)
    fraud = inject_fraud(rng, pools)
    genuine = _genuine(rng, n_txns - len(fraud), pools)
    df = pd.concat([genuine, fraud], ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df[COLUMNS]


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "data" / "transactions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(out, index=False)
    fraud = df[df["is_fraud"]]
    print(f"Wrote {out}")
    print(f"Total rows:      {len(df):,}")
    print(f"Fraud rows:      {len(fraud):,}")
    print(f"Fraud rate:      {len(fraud) / len(df):.2%}")
    print("Fraud by pattern:")
    for pat, cnt in fraud["fraud_pattern"].value_counts().items():
        print(f"  {pat:<24} {cnt}")


if __name__ == "__main__":
    main()
