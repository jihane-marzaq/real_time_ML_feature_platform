#Construire notre table finale de features pour le ML

{{ config(materialized='view') }}

WITH stg AS (

    SELECT * 
    FROM {{ ref('stg_transactions') }}

),

card_agg AS (

    SELECT * 
    FROM {{ ref('feat_card_aggregates') }}

)

SELECT
    t.transaction_id,
    c.transaction_ts AS event_timestamp,
    t.card_id,
    t.transaction_amt,
    t.transaction_amt_log,
    t.transaction_hour,
    t.transaction_dow,
    t.card_network,
    t.card_type,
    t.product_cd,
    t.purchaser_email_domain,
    t.recipient_email_domain,
    
    -- Features agrégées jointes depuis feat_card_aggregates
    c.card_txn_count_24h,
    c.card_txn_count_1h,
    
    -- Label cible pour l'entraînement
    t.is_fraud

FROM stg t
LEFT JOIN card_agg c
    ON t.transaction_id = c.transaction_id