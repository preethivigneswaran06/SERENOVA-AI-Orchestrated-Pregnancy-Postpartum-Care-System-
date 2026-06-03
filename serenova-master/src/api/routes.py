from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

from src.pipeline.orchestrator import Orchestrator

app = FastAPI(title="Bloom AI Backend")

orc = Orchestrator()


# =========================
# REQUEST SCHEMA
# =========================
class SyncRequest(BaseModel):
    raw_signals: Dict[str, Any]
    maternal_context: Dict[str, Any]
    gestational_week: int
    nutrition_score: int = 70


# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def home():
    return {"status": "Bloom backend running"}


# =========================
# MAIN PIPELINE
# =========================
@app.post("/sync")
def sync(data: SyncRequest):
    payload = {
        "raw_signals": data.raw_signals,
        "maternal_context": data.maternal_context,
        "gestational_week": data.gestational_week,
        "nutrition_score": data.nutrition_score,
    }

    result = orc.run_sync(payload)

    return result


# =========================
# SIMPLE CHAT (OPTIONAL)
# =========================
@app.post("/chat")
def chat(question: str):
    return {"response": f"You asked: {question}"}