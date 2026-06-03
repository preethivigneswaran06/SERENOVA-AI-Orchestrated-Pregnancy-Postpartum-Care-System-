def map_conditions(features):
    H, W, L = "high", "watch", "low"

    f = features

    conditions = [

        {"name": "Fetal hypoxia",
         "status": H if f["hypoxia"] >= 0.6 else W if f["hypoxia"] >= 0.35 else L},

        {"name": "Preeclampsia",
         "status": H if f["bp_sys"] >= 140 or f["bp_dia"] >= 90
         else W if f["bp_sys"] >= 130 else L},

        {"name": "Gestational hypertension",
         "status": H if f["bp_sys"] >= 140 else W if f["bp_sys"] >= 130 else L},

        {"name": "Anemia",
         "status": H if f["spo2"] < 92 else W if f["spo2"] < 95 else L},

        {"name": "Gestational diabetes",
         "status": H if f["glucose"] >= 125 else W if f["glucose"] >= 95 else L},

        {"name": "Hypoglycemia",
         "status": H if f["glucose"] < 70 else W if f["glucose"] < 80 else L},

        {"name": "Infection risk",
         "status": H if f.get("temp", 36) > 38.5 else W if f.get("temp", 36) > 37.5 else L},

        {"name": "Preterm labor risk",
         "status": H if f.get("contractions", 0) >= 6 else W if f.get("contractions", 0) >= 3 else L},

        {"name": "Cord compression",
         "status": H if f["fhr_std"] < 2 else W if f["fhr_std"] < 6 else L},

        {"name": "Fetal tachycardia",
         "status": H if f["fhr"] > 160 else W if f["fhr"] > 150 else L},

        {"name": "Fetal bradycardia",
         "status": H if f["fhr"] < 110 else W if f["fhr"] < 120 else L},

        {"name": "IUGR",
         "status": W if f["fhr_std"] < 6 else L},

        {"name": "Placental issues",
         "status": W if f["hypoxia"] > 0.35 else L},

        {"name": "DVT risk",
         "status": W if f["hr"] > 95 else L},

        {"name": "Maternal stress",
         "status": H if f["hrv"] < 20 else W if f["hrv"] < 30 else L},

        {"name": "Neonatal hypoglycemia risk",
         "status": W if f["glucose"] >= 95 else L},

        {"name": "Overall fetal well-being",
         "status": H if f["hypoxia"] >= 0.6 else W if f["hypoxia"] >= 0.35 else L},
    ]

    return conditions