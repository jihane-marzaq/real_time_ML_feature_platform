#Créer des features comportementales liées à la carte :

{{ config(materialized='table') }}

WITH base AS (

    SELECT 
        *,
        -- Conversion du timestamp en secondes UNIX pour BigQuery RANGE
        UNIX_SECONDS(transaction_ts) AS ts_in_seconds
    FROM {{ ref('stg_transactions') }}

),

card_features AS (

    SELECT
        transaction_id,
        card_id,
        transaction_ts,
        transaction_amt,
        is_fraud,

        -- 24 heures = 86 400 secondes
        COUNT(*) OVER (
            PARTITION BY card_id
            ORDER BY ts_in_seconds
            RANGE BETWEEN 86400 PRECEDING AND CURRENT ROW
        ) AS card_txn_count_24h,

        -- 1 heure = 3 600 secondes
        COUNT(*) OVER (
            PARTITION BY card_id
            ORDER BY ts_in_seconds
            RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW
        ) AS card_txn_count_1h

    FROM base

)

SELECT *
FROM card_features