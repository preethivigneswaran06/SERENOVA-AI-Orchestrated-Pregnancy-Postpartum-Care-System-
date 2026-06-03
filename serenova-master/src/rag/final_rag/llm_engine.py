from transformers import pipeline

# Load model (first time will download)
generator = pipeline("text2text-generation", model="google/flan-t5-small")


def generate_explanation(summary, rules):
    prompt = f"""
Patient condition:
{summary}

Medical rules:
{rules}

Explain the condition clearly and give recommendation.
"""

    result = generator(prompt, max_length=150)

    return result[0]["generated_text"]


# Test
if __name__ == "__main__":
    print(generate_explanation("HR high, SpO2 low", "Hypoxia risk"))