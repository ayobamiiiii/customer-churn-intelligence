
-- ============================================================
-- CUSTOMER CHURN INTELLIGENCE
-- PostgreSQL Exploratory Data Analysis
-- ============================================================

-- 1. Overall business summary
SELECT
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = '1' THEN 1 ELSE 0 END) AS churned,
    SUM(CASE WHEN churn = '0' THEN 1 ELSE 0 END) AS retained,
    ROUND(
        100.0 * SUM(CASE WHEN churn = '1' THEN 1 ELSE 0 END)
        / COUNT(*), 2
    ) AS overall_churn_pct,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges,
    ROUND(AVG(tenure), 2) AS avg_tenure_months
FROM customers;

-- 2. Churn by contract
SELECT
    contract,
    COUNT(*) AS customers,
    SUM(CASE WHEN churn = '1' THEN 1 ELSE 0 END) AS churned,
    ROUND(
        100.0 * AVG((churn = '1')::int), 2
    ) AS churn_pct
FROM customers
GROUP BY contract
ORDER BY churn_pct DESC;

-- 3. Churn by tenure group
SELECT
    CASE
        WHEN tenure <= 12 THEN '01. 0-12 months'
        WHEN tenure <= 24 THEN '02. 13-24 months'
        WHEN tenure <= 48 THEN '03. 25-48 months'
        ELSE '04. 49+ months'
    END AS tenure_group,
    COUNT(*) AS customers,
    SUM((churn = '1')::int) AS churned,
    ROUND(100.0 * AVG((churn = '1')::int), 2) AS churn_pct
FROM customers
GROUP BY 1
ORDER BY 1;

-- 4. Churn by payment method
SELECT
    payment_method,
    COUNT(*) AS customers,
    SUM((churn = '1')::int) AS churned,
    ROUND(100.0 * AVG((churn = '1')::int), 2) AS churn_pct
FROM customers
GROUP BY payment_method
ORDER BY churn_pct DESC;

-- 5. Churn by monthly charge band
SELECT
    CASE
        WHEN monthly_charges < 35 THEN 'Low charges'
        WHEN monthly_charges < 70 THEN 'Medium charges'
        ELSE 'High charges'
    END AS charge_group,
    COUNT(*) AS customers,
    SUM((churn = '1')::int) AS churned,
    ROUND(100.0 * AVG((churn = '1')::int), 2) AS churn_pct
FROM customers
GROUP BY 1
ORDER BY churn_pct DESC;

-- 6. Contract and payment method interaction
SELECT
    contract,
    payment_method,
    COUNT(*) AS customers,
    SUM((churn = '1')::int) AS churned,
    ROUND(100.0 * AVG((churn = '1')::int), 2) AS churn_pct
FROM customers
GROUP BY contract, payment_method
HAVING COUNT(*) >= 30
ORDER BY churn_pct DESC;

-- 7. Churn by internet service
SELECT
    internet_service,
    COUNT(*) AS customers,
    SUM((churn = '1')::int) AS churned,
    ROUND(100.0 * AVG((churn = '1')::int), 2) AS churn_pct
FROM customers
GROUP BY internet_service
ORDER BY churn_pct DESC;

-- 8. Churn by senior citizen status
SELECT
    senior_citizen,
    COUNT(*) AS customers,
    SUM((churn = '1')::int) AS churned,
    ROUND(100.0 * AVG((churn = '1')::int), 2) AS churn_pct
FROM customers
GROUP BY senior_citizen
ORDER BY churn_pct DESC;

-- 9. Churn by online security
SELECT
    online_security,
    COUNT(*) AS customers,
    SUM((churn = '1')::int) AS churned,
    ROUND(100.0 * AVG((churn = '1')::int), 2) AS churn_pct
FROM customers
GROUP BY online_security
ORDER BY churn_pct DESC;

-- 10. Churn by tech support
SELECT
    tech_support,
    COUNT(*) AS customers,
    SUM((churn = '1')::int) AS churned,
    ROUND(100.0 * AVG((churn = '1')::int), 2) AS churn_pct
FROM customers
GROUP BY tech_support
ORDER BY churn_pct DESC;

-- 11. Customer tenure and monthly charges by churn status
SELECT
    churn,
    COUNT(*) AS customers,
    ROUND(AVG(tenure), 2) AS avg_tenure,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges,
    ROUND(AVG(total_charges), 2) AS avg_total_charges
FROM customers
GROUP BY churn
ORDER BY churn;

-- 12. Missing values in important columns
SELECT
    COUNT(*) FILTER (WHERE tenure IS NULL) AS missing_tenure,
    COUNT(*) FILTER (WHERE monthly_charges IS NULL) AS missing_monthly_charges,
    COUNT(*) FILTER (WHERE total_charges IS NULL) AS missing_total_charges,
    COUNT(*) FILTER (WHERE contract IS NULL) AS missing_contract,
    COUNT(*) FILTER (WHERE payment_method IS NULL) AS missing_payment_method,
    COUNT(*) FILTER (WHERE churn IS NULL) AS missing_churn
FROM customers;

-- 13. Customer distribution by contract and tenure
SELECT
    contract,
    CASE
        WHEN tenure <= 12 THEN '0-12 months'
        WHEN tenure <= 24 THEN '13-24 months'
        WHEN tenure <= 48 THEN '25-48 months'
        ELSE '49+ months'
    END AS tenure_group,
    COUNT(*) AS customers,
    ROUND(100.0 * AVG((churn = '1')::int), 2) AS churn_pct
FROM customers
GROUP BY contract, tenure_group
ORDER BY contract, tenure_group;

