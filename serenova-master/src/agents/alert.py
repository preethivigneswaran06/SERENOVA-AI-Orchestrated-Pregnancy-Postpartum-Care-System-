def generate_alert(decision):

    if decision == "🚨 EMERGENCY":
        return "CALL DOCTOR IMMEDIATELY"

    elif decision == "⚠️ NEEDS ATTENTION":
        return "Notify doctor"

    return "Normal monitoring"