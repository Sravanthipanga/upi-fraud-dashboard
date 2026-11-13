-- ============================================================================
-- UPI Fraud Detection — analyst investigation queries (DuckDB)
-- ============================================================================
-- These queries run against the RAW synthetic dataset
-- (data/transactions.csv, produced by `python -m src.generate_data`).
-- They use the is_fraud label column where useful: think of them as the
-- queries an analyst writes while *investigating* known/suspected fraud and
-- validating hypotheses, not as a production detection engine.
--
-- IDENTITY NOTE: sender_vpa / receiver_vpa are MASKED display values — many
-- distinct accounts share one masked string. Behavioural queries (velocity,
-- fan-in, fan-out) therefore group/join on sender_key / receiver_key, the
-- full synthetic account identity, and only display the masked VPA.
--
-- Run all of them:   duckdb -c ".read sql/fraud_queries.sql"
-- Or paste one at a time into any DuckDB session.
--
-- NOTE: the data is 100% synthetic. No real transactions, people or accounts.
-- ============================================================================


-- ----------------------------------------------------------------------------
-- Q1. Overall fraud rate
-- Business question: What share of our UPI volume is labelled fraud, and how
-- much money does it represent?
-- ----------------------------------------------------------------------------
SELECT
    COUNT(*)                                        AS total_transactions,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END)       AS fraud_transactions,
    ROUND(100.0 * AVG(CASE WHEN is_fraud THEN 1.0 ELSE 0.0 END), 3)
                                                    AS fraud_rate_pct,
    ROUND(SUM(amount_inr), 2)                       AS total_value_inr,
    ROUND(SUM(CASE WHEN is_fraud THEN amount_inr ELSE 0 END), 2)
                                                    AS fraud_value_inr
FROM read_csv('data/transactions.csv');


-- ----------------------------------------------------------------------------
-- Q2. Fraud rate by hour of day
-- Business question: At what hours is fraud concentrated? (Overnight spikes
-- are a classic signal of organised / scripted fraud.)
-- ----------------------------------------------------------------------------
SELECT
    HOUR(timestamp)                                 AS hour_of_day,
    COUNT(*)                                        AS transactions,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END)       AS fraud_transactions,
    ROUND(100.0 * AVG(CASE WHEN is_fraud THEN 1.0 ELSE 0.0 END), 2)
                                                    AS fraud_rate_pct
FROM read_csv('data/transactions.csv')
GROUP BY hour_of_day
ORDER BY hour_of_day;


-- ----------------------------------------------------------------------------
-- Q3. Daily fraud trend
-- Business question: Is fraud volume growing over the 30-day window, and are
-- there spike days worth a deeper look?
-- ----------------------------------------------------------------------------
SELECT
    CAST(timestamp AS DATE)                         AS day,
    COUNT(*)                                        AS transactions,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END)       AS fraud_transactions,
    ROUND(100.0 * AVG(CASE WHEN is_fraud THEN 1.0 ELSE 0.0 END), 2)
                                                    AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud THEN amount_inr ELSE 0 END), 2)
                                                    AS fraud_value_inr
FROM read_csv('data/transactions.csv')
GROUP BY day
ORDER BY day;


-- ----------------------------------------------------------------------------
-- Q4. Top receivers by fraud value
-- Business question: Which receiving VPAs accumulate the most fraudulent
-- money? (Candidate mule accounts / compromised merchants.)
-- ----------------------------------------------------------------------------
SELECT
    receiver_key,
    MAX(receiver_vpa)                               AS receiver_vpa_masked,
    COUNT(*)                                        AS fraud_transactions,
    COUNT(DISTINCT sender_key)                      AS unique_senders,
    ROUND(SUM(amount_inr), 2)                       AS fraud_value_inr
FROM read_csv('data/transactions.csv')
WHERE is_fraud
GROUP BY receiver_key
ORDER BY fraud_value_inr DESC
LIMIT 10;


-- ----------------------------------------------------------------------------
-- Q5. Velocity suspects
-- Business question: Which senders fire off bursts of transactions inside a
-- 10-minute window — behaviour typical of account takeover or scripted abuse?
-- ----------------------------------------------------------------------------
SELECT
    a.sender_key,
    MAX(a.sender_vpa)                               AS sender_vpa_masked,
    CAST(a.timestamp AS DATE)                       AS day,
    COUNT(*)                                        AS burst_transactions,
    ROUND(SUM(a.amount_inr), 2)                     AS burst_value_inr
FROM read_csv('data/transactions.csv') a
JOIN read_csv('data/transactions.csv') b
  ON a.sender_key = b.sender_key
 AND b.timestamp BETWEEN a.timestamp - INTERVAL 10 MINUTE AND a.timestamp
GROUP BY a.sender_key, day, a.txn_id
HAVING COUNT(*) >= 5
ORDER BY burst_transactions DESC
LIMIT 15;


-- ----------------------------------------------------------------------------
-- Q6. Fan-in receivers (potential mules)
-- Business question: Which receivers collect money from an unusually large
-- number of distinct senders in a single day — the fan-in signature of a
-- money-mule collection account?
-- ----------------------------------------------------------------------------
SELECT
    receiver_key,
    MAX(receiver_vpa)                               AS receiver_vpa_masked,
    CAST(timestamp AS DATE)                         AS day,
    COUNT(DISTINCT sender_key)                      AS unique_senders,
    COUNT(*)                                        AS transactions,
    ROUND(SUM(amount_inr), 2)                       AS collected_inr
FROM read_csv('data/transactions.csv')
GROUP BY receiver_key, day
HAVING COUNT(DISTINCT sender_key) >= 8
ORDER BY unique_senders DESC
LIMIT 15;


-- ----------------------------------------------------------------------------
-- Q7. High-value transfers from brand-new devices
-- Business question: How many large transfers (> ₹20,000) came from a device
-- we have never seen before? A first-seen device moving big money is one of
-- the strongest single account-takeover signals — how much value is exposed?
-- ----------------------------------------------------------------------------
SELECT
    COUNT(*)                                        AS suspicious_transfers,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END)       AS labelled_fraud,
    ROUND(SUM(amount_inr), 2)                       AS exposed_value_inr,
    ROUND(100.0 * AVG(CASE WHEN is_fraud THEN 1.0 ELSE 0.0 END), 2)
                                                    AS fraud_share_pct
FROM read_csv('data/transactions.csv')
WHERE is_new_device
  AND amount_inr > 20000;
