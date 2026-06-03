import numpy as np
from typing import Dict, Any

from src.agents.base_agent import BaseAgent
from src.model.infer import FetalHypoxiaPredictor
from src.preprocessing.signal_processor import extract_scalar_features


_SYSTEM_PROMPT = """(same as yours — keep unchanged)"""


class FetalAgent(BaseAgent):
    def __init__(self):
        super().__init__("FetalAgent", _SYSTEM_PROMPT)
        self.predictor = FetalHypoxiaPredictor()

    def analyse(
        self,
        raw_signals: Dict[str, np.ndarray],
        maternal_context: Dict[str, Any],
        gestational_week: int,
        kick_count_per_hour: float = None,
    ) -> Dict[str, Any]:

        cnn_result = self.predictor.predict(raw_signals)
        scalar_features = extract_scalar_features(raw_signals)

        message = f"""
Gestational week: {gestational_week}

CNN model output:
  hypoxia_probability: {cnn_result['hypoxia_probability']}
  hypoxia_flag: {cnn_result['hypoxia_flag']}

Signal scalar features:
  FHR mean: {scalar_features.get('FHR_mean', 'N/A')} bpm
  FHR std: {scalar_features.get('FHR_std', 'N/A')} bpm
  ECG mean: {scalar_features.get('ECG_mean', 'N/A')}
  PLETH mean: {scalar_features.get('PLETH_mean', 'N/A')}
  RESP mean: {scalar_features.get('RESP_mean', 'N/A')}

Maternal context:
  HR: {maternal_context.get('HR_value')} bpm
  SpO2: {maternal_context.get('SpO2_value')}%
  RESP rate: {maternal_context.get('RESP_rate')} breaths/min
  HRV: {maternal_context.get('hrv_ms')} ms

Kick count: {kick_count_per_hour if kick_count_per_hour else 'not recorded'}

Return JSON.
        """.strip()

        result = self._call(message)
        result["cnn_raw"] = cnn_result
        return result