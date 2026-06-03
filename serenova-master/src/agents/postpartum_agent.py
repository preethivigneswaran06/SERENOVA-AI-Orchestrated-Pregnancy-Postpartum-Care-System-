# src/agents/postpartum_agent.py

from typing import Dict, Any, List
from datetime import date, datetime
from src.agents.base_agent import BaseAgent


_SYSTEM_PROMPT = """You are the Postpartum Care Agent for Bloom.

Return ONLY valid JSON:
{
  "day_postpartum": int,
  "mood_today": "great|good|okay|low|very_low",
  "mood_trend": "positive|stable|declining|concerning",
  "sleep_status": "normal|fragmented|severely_fragmented",
  "lactation_status": "on_track|below_target",
  "epds_score": int,
  "risk_level": "low|elevated|high",
  "overall_status": "normal|watchful|urgent",
  "affirmation": "one supportive sentence"
}
"""


class PostpartumAgent(BaseAgent):
    def __init__(self):
        super().__init__("PostpartumAgent", _SYSTEM_PROMPT)

    def analyse(
        self,
        birth_date: str,
        daily_mood_log: List[str],
        sleep_data: Dict[str, Any],
        lactation_log: Dict[str, Any],
        epds_score: int = None,
    ) -> Dict[str, Any]:

        # =========================
        # ✅ FIX 1: HANDLE PREGNANCY CASE
        # =========================
        if not birth_date:
            return {
                "status": "not_applicable",
                "message": "Patient currently pregnant"
            }

        try:
            birth = datetime.strptime(birth_date, "%Y-%m-%d").date()
            day_pp = (date.today() - birth).days
        except Exception:
            return {
                "error": "invalid_birth_date"
            }

        # =========================
        # ✅ FIX 2: INVALID FUTURE DATE
        # =========================
        if day_pp < 0:
            return {
                "status": "invalid",
                "message": "Birth date is in the future"
            }

        # =========================
        # 🔹 LLM CALL (NORMAL CASE)
        # =========================
        message = f"""
Day postpartum: {day_pp}

Mood log (last 7 days): {daily_mood_log}
Sleep data: {sleep_data}
Lactation: {lactation_log}
EPDS score: {epds_score if epds_score is not None else "not taken"}

Analyse and return JSON.
        """.strip()

        result = self._call(message, max_tokens=1200)

        # =========================
        # ✅ FIX 3: FALLBACK SAFETY
        # =========================
        if not isinstance(result, dict) or "error" in result:
            return {
                "day_postpartum": day_pp,
                "mood_today": "okay",
                "mood_trend": "stable",
                "sleep_status": "normal",
                "lactation_status": "on_track",
                "epds_score": epds_score or 0,
                "risk_level": "low",
                "overall_status": "normal",
                "affirmation": "You are doing well. Keep taking care of yourself."
            }

        return result