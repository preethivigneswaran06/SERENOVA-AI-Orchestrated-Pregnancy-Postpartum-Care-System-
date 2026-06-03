from typing import Dict, Any
from src.agents.base_agent import BaseAgent
from datetime import date


_SYSTEM_PROMPT = """You are the Labor Prediction Agent.

You predict likelihood of labor in next 7 days.

Return ONLY JSON:
{
  "active": bool,
  "probability_7d": int,
  "peak_days": ["YYYY-MM-DD"],
  "summary": "short explanation"
}
"""


class LaborAgent(BaseAgent):
    def __init__(self):
        super().__init__("LaborAgent", _SYSTEM_PROMPT)

    def predict(self, gestational_week: int, due_date: str, signals: Dict[str, Any]):

        if gestational_week < 36:
            return {"active": False}

        message = f"""
Today: {date.today()}
Week: {gestational_week}
Due date: {due_date}
Signals: {signals}

Predict labor probability.
        """

        return self._call(message)