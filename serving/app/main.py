import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from app.schemas import TransactionRequest, PredictionResponse
from app.predictor import predictor
from app.config import DECISION_THRESHOLD


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager — remplace @app.on_event("startup").
    Le modèle est chargé UNE SEULE FOIS au démarrage.
    Toutes les requêtes suivantes utilisent le modèle en mémoire.
    """
    predictor.load()
    yield
    # Cleanup si nécessaire au shutdown


app = FastAPI(
    title="Fraud Detection API",
    description="""
    Real-Time ML Feature Platform — Détection de fraude bancaire.

    Pipeline : Transaction → Features (local + Feast) → XGBoost → Score
    Latence cible : < 100ms end-to-end
    """,
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """
    Health check endpoint.
    Cloud Run l'utilise pour savoir si le container est prêt.
    Retourne 200 si le modèle est chargé, 503 sinon.
    """
    if predictor.is_ready:
        return {"status": "healthy", "model_loaded": True}
    raise HTTPException(status_code=503, detail="Model not ready")


@app.get("/")
def root():
    return {
        "service": "Fraud Detection API",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest):
    """
    Endpoint principal de prédiction.

    Reçoit une transaction bancaire, retourne un score de fraude.
    Le seuil de décision est configurable via DECISION_THRESHOLD.
    """
    start_ms = time.time() * 1000

    try:
        result = predictor.predict(transaction.model_dump())
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    latency = time.time() * 1000 - start_ms

    return PredictionResponse(
        transaction_id = transaction.TransactionID,
        is_fraud       = result["is_fraud"],
        fraud_score    = round(result["score"], 4),
        threshold_used = DECISION_THRESHOLD,
        risk_level     = result["risk_level"],
        latency_ms     = round(latency, 2),
    )


@app.get("/model/info")
def model_info():
    """Informations sur le modèle en production."""
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {
        "model_name":  "fraud_detection_model",
        "stage":       "Production",
        "threshold":   DECISION_THRESHOLD,
        "features":    predictor.model.feature_names_in_.tolist()
                       if hasattr(predictor.model, 'feature_names_in_')
                       else "N/A",
    }