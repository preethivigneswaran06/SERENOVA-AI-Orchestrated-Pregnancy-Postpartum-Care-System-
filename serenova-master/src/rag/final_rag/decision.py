def decide_action(summary):
    summary = summary.lower()

    if "low" in summary or "high" in summary:
        return "EMERGENCY 🚨"

    elif "moderate" in summary:
        return "ALERT ⚠️"

    else:
        return "MONITOR ✅"