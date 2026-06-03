# src/pipeline/orchestrator.py

import asyncio
from typing import Dict, Any

from src.agents.fetal_agent import FetalAgent
from src.agents.maternal_agent import MaternalAgent
from src.agents.risk_agent import RiskAgent
from src.agents.nutrition_agent import NutritionAgent
from src.agents.labor_agent import LaborAgent
from src.agents.postpartum_agent import PostpartumAgent
from src.agents.conversational_agent import ConversationalAgent


def safe(fn, *args):
    try:
        res = fn(*args)
        return res if isinstance(res, dict) else {"error": "invalid_output"}
    except Exception as e:
        return {"error": str(e)}


def ensure_dict(x):
    return x if isinstance(x, dict) else {}


class Orchestrator:
    def __init__(self):
        self.fetal = FetalAgent()
        self.maternal = MaternalAgent()
        self.risk = RiskAgent()
        self.nutrition = NutritionAgent()
        self.labor = LaborAgent()
        self.postpartum = PostpartumAgent()
        self.chat = ConversationalAgent()

    def run_sync(self, payload: Dict[str, Any]):

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def run():

            # 🔥 CORE
            fetal_task = loop.run_in_executor(
                None, safe,
                self.fetal.analyse,
                payload.get("raw_signals", {}),
                payload.get("maternal_context", {}),
                payload.get("gestational_week", 0),
            )

            maternal_task = loop.run_in_executor(
                None, safe,
                self.maternal.analyse,
                payload.get("maternal_context", {}),
                payload.get("trend_history", []),
                payload.get("gestational_week", 0),
            )

            fetal_res, maternal_res = await asyncio.gather(fetal_task, maternal_task)

            fetal_res = ensure_dict(fetal_res)
            maternal_res = ensure_dict(maternal_res)

            # 🔥 RISK
            risk_res = safe(
                self.risk.assess,
                fetal_res,
                maternal_res,
                payload.get("nutrition_score", 70),
                payload.get("gestational_week", 0),
            )

            risk_res = ensure_dict(risk_res)

            # 🔥 PARALLEL
            nutrition_task = loop.run_in_executor(
                None, safe,
                self.nutrition.analyse,
                payload.get("nutrition_input", []),
                payload.get("supplement_log", []),
                payload.get("hydration_ml", 2000),
                payload.get("gestational_week", 0),
            )

            labor_task = loop.run_in_executor(
                None, safe,
                self.labor.predict,
                payload.get("gestational_week", 0),
                payload.get("due_date", "2026-01-01"),
                payload.get("labor_signals", {}),
            )

            postpartum_task = loop.run_in_executor(
                None, safe,
                self.postpartum.analyse,
                payload.get("birth_date", None),  # ✅ KEY FIX
                payload.get("mood_log", []),
                payload.get("sleep_data", {}),
                payload.get("lactation_log", {}),
            )

            nutrition_res, labor_res, postpartum_res = await asyncio.gather(
                nutrition_task, labor_task, postpartum_task
            )

            nutrition_res = ensure_dict(nutrition_res)
            labor_res = ensure_dict(labor_res)
            postpartum_res = ensure_dict(postpartum_res)

            # 🔥 CHAT
            chat_context = {
                "gestational_week": payload.get("gestational_week"),
                "risk_score": risk_res.get("risk_score"),
                "risk_level": risk_res.get("risk_level"),
                "alert_tone": risk_res.get("alert_tone"),
                "hypoxia_probability": (fetal_res.get("cnn_raw") or {}).get("hypoxia_probability"),
                "fetal_status": fetal_res.get("fetal_status"),
                "maternal_summary": maternal_res.get("overall_status"),
                "nutrition_compliance": nutrition_res.get("compliance_score"),
            }

            chat_res = safe(
                self.chat.respond,
                payload.get("question", "How am I doing?"),
                chat_context,
            )

            chat_res = ensure_dict(chat_res)

            return {
                "fetal": fetal_res,
                "maternal": maternal_res,
                "risk": risk_res,
                "nutrition": nutrition_res,
                "labor": labor_res,
                "postpartum": postpartum_res,
                "conversation": chat_res,
            }

        return loop.run_until_complete(run())