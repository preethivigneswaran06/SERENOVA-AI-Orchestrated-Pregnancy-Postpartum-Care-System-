import os
import json
import requests
from typing import Any, Dict
from dotenv import load_dotenv

load_dotenv()


def safe_parse(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raw = raw.strip()
        open_b = raw.count('{') - raw.count('}')
        open_sq = raw.count('[') - raw.count(']')
        raw += ']' * open_sq + '}' * open_b
        try:
            return json.loads(raw)
        except:
            return {"error": "non_json", "raw_output": raw[:300]}


class BaseAgent:
    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt

        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY missing")

        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def _call(self, user_message: str, temperature: float = 0.2, max_tokens: int = 2000):

        try:
            res = requests.post(
                self.url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "openrouter/auto",
                    "messages": [
                        {
                            "role": "system",
                            "content": self.system_prompt + "\nReturn ONLY valid JSON.",
                        },
                        {"role": "user", "content": user_message},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=30,
            )

            data = res.json()

            text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )

            if text.startswith("```"):
                text = text.replace("```json", "").replace("```", "").strip()

            return safe_parse(text)

        except Exception as e:
            return {"error": str(e)}