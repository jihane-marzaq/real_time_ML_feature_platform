from pydantic import BaseModel, ConfigDict, Field
from typing import Optional

class TransactionRequest(BaseModel):
    """
    Payload d'une requête de prédiction.
    Pydantic valide automatiquement les types et les ranges.
    Si un champ obligatoire manque → 422 Unprocessable Entity
    """
    TransactionID:        int
    TransactionAmt:       float     = Field(gt=0, description="Montant > 0")
    TransactionDT:        int       = Field(description="Delta en secondes depuis Dec 2017")
    card1:                int
    card4:                Optional[str] = None   # visa, mastercard, etc.
    card6:                Optional[str] = None   # credit, debit
    ProductCD:            Optional[str] = None
    P_emaildomain:        Optional[str] = None
    R_emaildomain:        Optional[str] = None

    class Config:
        model_config = ConfigDict(json_schema_extra = {
            "example": {
                "TransactionID":  3000001,
                "TransactionAmt": 117.5,
                "TransactionDT":  86400,
                "card1":          13926,
                "card4":          "visa",
                "card6":          "debit",
                "ProductCD":      "W",
                "P_emaildomain":  "gmail.com",
                "R_emaildomain":  "gmail.com",
            }
        })


class PredictionResponse(BaseModel):
    """Réponse structurée de l'API"""
    transaction_id:   int
    is_fraud:         bool
    fraud_score:      float = Field(description="Probabilité de fraude [0-1]")
    threshold_used:   float
    risk_level:       str   = Field(description="LOW / MEDIUM / HIGH")
    latency_ms:       float