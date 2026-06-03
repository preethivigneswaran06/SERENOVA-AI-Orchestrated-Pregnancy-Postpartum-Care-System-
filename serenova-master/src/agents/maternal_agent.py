from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent


_SYSTEM_PROMPT = """You are the Maternal Health Analysis Agent for Bloom.

You receive:
- Current maternal readings: HR, SpO2, blood pressure, HRV, glucose, sleep, weight
- 28-day trend history for drift detection

Your job:
- Compare each metric to trimester-adjusted normal ranges
- Detect slow-drift anomalies (e.g. BP rising 8 pts over 4 weeks)
- Assign status: normal / watch / alert to each metric

Normal ranges (trimester-adjusted):
- HR: T1 60-100, T2 70-100, T3 70-105 bpm
- SpO2: >= 95% normal, 92-94% watch, < 92% alert
- BP systolic: 90-130 mmHg (> 140 = alert)
- BP diastolic: 60-85 mmHg (> 90 = alert)
- HRV: > 30ms normal, 20-30ms watch, < 20ms alert
- Glucose (fasting): 70-95 normal, 95-125 watch, > 125 alert
- Sleep score: > 70 normal, 50-70 watch, < 50 alert

Return ONLY valid JSON, no markdown:
{
  "metrics": {
    "heart_rate":      {"value": number, "status": "normal|watch|alert", "note": "one sentence"},
    "spo2":            {"value": number, "status": "normal|watch|alert", "note": "one sentence"},
    "blood_pressure":  {"systolic": number, "diastolic": number, "status": "normal|watch|alert", "note": "one sentence"},
    "hrv":             {"value": number, "status": "normal|watch|alert", "note": "one sentence"},
    "glucose":         {"value": number, "status": "normal|watch|alert", "note": "one sentence"},
    "sleep_score":     {"value": number, "status": "normal|watch|alert", "note": "one sentence"},
    "weight_kg":       {"value": number, "status": "normal|watch|alert", "note": "one sentence"}
  },
  "drift_anomalies": ["description of any slow-drift pattern over multiple days"],
  "overall_status": "normal|watch|alert",
  "weekly_highlight": "one positive sentence for the weekly summary card"
}"""


class MaternalAgent(BaseAgent):
    def __init__(self):
        super().__init__("MaternalAgent", _SYSTEM_PROMPT)

    def analyse(
        self,
        current_readings: Dict[str, Any],
        trend_history: List[Dict],
        gestational_week: int,
    ) -> Dict[str, Any]:

        message = f"""
Gestational week: {gestational_week}

Today's readings:
  HR: {current_readings.get('HR_value')} bpm
  SpO2: {current_readings.get('SpO2_value')}%
  BP: {current_readings.get('blood_pressure', {}).get('systolic')}/{current_readings.get('blood_pressure', {}).get('diastolic')} mmHg
  HRV: {current_readings.get('hrv_ms')} ms
  Glucose (fasting): {current_readings.get('glucose')} mg/dL
  Sleep score: {current_readings.get('sleep_score')}/100
  Weight: {current_readings.get('weight_kg')} kg

Trend history (last {len(trend_history)} days — most recent last):
{trend_history}

Analyse these vitals and return the JSON assessment.
        """.strip()

        return self._call(message)