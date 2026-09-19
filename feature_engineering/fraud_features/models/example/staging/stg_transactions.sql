# prendre les données brutes et les rendre propres/cohérentes :
---renommer
---caster
---transformer TransactionDT
---gérer les NULL 

{{ config(materialized='view') }}

WITH raw AS (

    SELECT * 
    FROM {{ source('fraud_detection', 'raw_transactions') }}

)

SELECT
    CAST(TransactionID AS STRING) AS transaction_id,
    TIMESTAMP_ADD(
    TIMESTAMP '2017-12-01 00:00:00 UTC',
    INTERVAL CAST(TransactionDT AS INT64) SECOND
) AS transaction_ts,
    CAST(card1 AS STRING) AS card_id,
    CAST(TransactionAmt AS NUMERIC) AS transaction_amt,
    LOG(CAST(TransactionAmt AS NUMERIC) + 1) AS transaction_amt_log,
    EXTRACT(HOUR FROM TIMESTAMP_ADD(
        TIMESTAMP '2017-12-01 00:00:00 UTC',
        INTERVAL CAST(TransactionDT AS INT64) SECOND
    )) AS transaction_hour,
    EXTRACT(DAYOFWEEK FROM TIMESTAMP_ADD(
        TIMESTAMP '2017-12-01 00:00:00 UTC',
        INTERVAL CAST(TransactionDT AS INT64) SECOND
    )) AS transaction_dow,
    CAST(card4 AS STRING) AS card_network,
    CAST(card6 AS STRING) AS card_type,
    CAST(ProductCD AS STRING) AS product_cd,
    CAST(P_emaildomain AS STRING) AS purchaser_email_domain,
    CAST(R_emaildomain AS STRING) AS recipient_email_domain,
    CAST(isFraud AS INT64) AS is_fraud

FROM raw

QUALIFY ROW_NUMBER() OVER (
    PARTITION BY TransactionID 
    ORDER BY TransactionDT DESC
) = 1