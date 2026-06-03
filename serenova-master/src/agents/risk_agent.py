from typing import Dict, Any
from src.agents.base_agent import BaseAgent
from src.rag.final_rag.main_pipeline import run_rag_pipeline  


_SYSTEM_PROMPT = """You are the Risk Assessment and Explanation Agent for Bloom.

STRICT RULES:
- Use exact numeric values
- No vague words like "appears normal"
- Output STRICT JSON only

Return:
{
  "risk_score": int,
  "risk_level": "low|medium|high",
  "alert_tone": "calm|watchful|urgent",
  "positives": [],
  "watchful_items": [],
  "alert_items": [],
  "plain_language_summary": "",
  "doctor_summary": ""
}
"""


class RiskAgent(BaseAgent):
    def __init__(self):
        super().__init__("RiskAgent", _SYSTEM_PROMPT)

    def assess(
        self,
        fetal_analysis: Dict[str, Any],
        maternal_analysis: Dict[str, Any],
        nutrition_compliance: int,
        gestational_week: int,
    ) -> Dict[str, Any]:

        # =========================
        # SAFE EXTRACTION
        # =========================
        cnn = fetal_analysis.get("cnn_raw", {}) or {}
        hypoxia = float(cnn.get("hypoxia_probability", 0))

        metrics = maternal_analysis.get("metrics", {}) or {}

        hr = metrics.get("heart_rate", {}).get("value", 0)
        spo2 = metrics.get("spo2", {}).get("value", 100)
        bp_sys = metrics.get("blood_pressure", {}).get("systolic", 120)
        bp_dia = metrics.get("blood_pressure", {}).get("diastolic", 80)
        hrv = metrics.get("hrv", {}).get("value", 60)
        glucose = metrics.get("glucose", {}).get("value", 90)

        # fallback signals
        fhr = fetal_analysis.get("fhr_mean", 140)
        fhr_std = fetal_analysis.get("fhr_std", 8)

        H, W, L = "high", "watch", "low"

        # =========================
        # ✅ FULL 17 CONDITIONS
        # =========================
        conditions = [
            {"name": "Gestational diabetes (GDM)", "status": H if glucose >= 125 else W if glucose >= 95 else L},
            {"name": "Hyperglycemia episodes", "status": W if glucose >= 110 else L},
            {"name": "Hypoglycemia", "status": H if glucose < 70 else W if glucose < 80 else L},
            {"name": "Gestational hypertension", "status": H if bp_sys >= 140 else W if bp_sys >= 130 else L},
            {"name": "Preeclampsia", "status": H if bp_sys >= 140 and spo2 < 95 else W if bp_sys >= 130 else L},
            {"name": "Anemia", "status": H if spo2 < 92 else W if spo2 < 95 else L},
            {"name": "Maternal sepsis", "status": L},
            {"name": "Deep vein thrombosis (DVT)", "status": W if hr > 95 else L},
            {"name": "Placental abruption", "status": L},
            {"name": "Fetal macrosomia", "status": W if glucose >= 110 else L},
            {"name": "Fetal growth restriction (IUGR)", "status": W if fhr_std < 6 else L},
            {"name": "Fetal hypoxia", "status": H if hypoxia >= 0.6 else W if hypoxia >= 0.35 else L},
            {"name": "Preterm labour", "status": L},
            {"name": "Cord compression", "status": H if fhr_std < 2 else W if fhr_std < 6 else L},
            {"name": "Postpartum haemorrhage (PPH)", "status": L},
            {"name": "Postpartum depression (PPMD)", "status": W if hrv < 30 else L},
            {"name": "Neonatal hypoglycemia risk", "status": W if glucose >= 95 else L},
        ]

        # =========================
        # ✅ RISK SCORE (STABLE)
        # =========================
        score = sum(
            6 if c["status"] == "high" else 3 if c["status"] == "watch" else 0
            for c in conditions
        )
        score = min(score, 100)

        if score < 30:
            level, tone = "low", "calm"
        elif score < 60:
            level, tone = "medium", "watchful"
        else:
            level, tone = "high", "urgent"

        # =========================
        # ✅ RAG (SAFE)
        # =========================
        try:
            rag_output = run_rag_pipeline({
                "conditions": conditions,
                "risk_score": score,
                "gestational_week": gestational_week
            })
        except Exception as e:
            rag_output = {"error": str(e)}

        # =========================
        # ✅ LLM EXPLANATION (SAFE)
        # =========================
        message = f"""
Week: {gestational_week}

Hypoxia: {hypoxia}
FHR: {fhr}
FHR variability: {fhr_std}
HR: {hr}
SpO2: {spo2}
BP: {bp_sys}/{bp_dia}
HRV: {hrv}
Glucose: {glucose}
Nutrition: {nutrition_compliance}

Risk score: {score}
Risk level: {level}

Explain clearly with numbers.
"""

        llm_output = self._call(message)

        # =========================
        # ✅ FINAL OUTPUT (STRICT JSON)
        # =========================
        if isinstance(llm_output, dict):
            return {
                **llm_output,
                "risk_score": score,
                "risk_level": level,
                "alert_tone": tone,
                "conditions": conditions,
                "rag": rag_output
            }

        return {
            "risk_score": score,
            "risk_level": level,
            "alert_tone": tone,
            "conditions": conditions,
            "rag": rag_output,
            "plain_language_summary": "Unable to generate explanation",
            "doctor_summary": "LLM failed"
        }