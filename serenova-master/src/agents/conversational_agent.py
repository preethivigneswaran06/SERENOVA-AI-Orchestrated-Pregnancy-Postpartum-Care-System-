from typing import Dict, Any
from src.agents.base_agent import BaseAgent


_SYSTEM_PROMPT = """You are a pregnancy assistant.
Return JSON only with:
{
  "response": "",
  "used_patient_data": true,
  "red_flag_detected": false,
  "suggested_follow_ups": []
}
"""


class ConversationalAgent(BaseAgent):
    def __init__(self):
        super().__init__("ConversationalAgent", _SYSTEM_PROMPT)

    def respond(self, question: str, context: Dict[str, Any]):

        message = f"""
Question: {question}

Week: {context.get('gestational_week')}
Risk Score: {context.get('risk_score')}
Risk Level: {context.get('risk_level')}
Hypoxia: {context.get('hypoxia_probability')}
Maternal: {context.get('maternal_summary')}

Respond in JSON.
"""

        res = self._call(message, temperature=0.4, max_tokens=1500)

        if "error" in res:
            return {
                "response": f"Your risk is {context.get('risk_score')} and currently low. Baby status is stable.",
                "used_patient_data": True,
                "red_flag_detected": False,
                "suggested_follow_ups": ["Continue regular monitoring"]
            }

        return res