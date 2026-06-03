import pickle
import random


# ==============================
# LOAD DATA
# ==============================
def load_all():
    pcg = pickle.load(open("data/fetal_pcg.pkl", "rb"))
    ecg = pickle.load(open("data/fetal_ecg.pkl", "rb"))
    ctg = pickle.load(open("data/ctg.pkl", "rb"))
    bidmc = pickle.load(open("data/bidmc.pkl", "rb"))

    return pcg, ecg, ctg, bidmc


# ==============================
# GET RANDOM WINDOW
# ==============================
def get_window(dataset):
    sample = random.choice(dataset)

    if len(sample["windows"]) == 0:
        return None

    return random.choice(sample["windows"])


# ==============================
# CLEAN FEATURES
# ==============================
def clean_features(feat):
    clean = {}

    for k, v in feat.items():

        # fix key names
        key = k.replace(",", "").replace(" ", "_")

        # flatten MFCC
        if key == "mfcc":
            for i, val in enumerate(v):
                clean[f"mfcc_{i}"] = float(val)

        else:
            try:
                clean[key] = float(v)
            except:
                continue

    return clean


# ==============================
# BUILD ONE SAMPLE
# ==============================
def build_sample(pcg, ecg, ctg, bidmc, idx):

    pcg_w = get_window(pcg)
    ecg_w = get_window(ecg)
    ctg_w = get_window(ctg)
    bidmc_w = get_window(bidmc)

    if None in [pcg_w, ecg_w, ctg_w, bidmc_w]:
        return None

    # 🔷 MERGE FEATURES
    features = {}

    features.update(clean_features(pcg_w["features"]))
    features.update(clean_features(ecg_w["features"]))
    features.update(clean_features(ctg_w["features"]))
    features.update(clean_features(bidmc_w["features"]))

    # ==============================
    # LABEL ENGINE (SAFE)
    # ==============================
    labels = {}

    # 🔴 fetal hypoxia (safe check)
    fhr = features.get("fhr_mean", None)

    if fhr is None or fhr == 0:
        labels["fetal_hypoxia"] = 0
    elif fhr < 110:
        labels["fetal_hypoxia"] = 1
    else:
        labels["fetal_hypoxia"] = 0

    # 🔴 stress
    resp = features.get("RESP_value", 0)
    hr = features.get("HR_value", 0)

    if resp > 24 or hr > 100:
        labels["stress"] = 1
    else:
        labels["stress"] = 0

    return {
        "id": idx,
        "features": features,
        "labels": labels
    }


# ==============================
# BUILD DATASET
# ==============================
def build_dataset(n=5000):

    pcg, ecg, ctg, bidmc = load_all()

    dataset = []

    for i in range(n):
        sample = build_sample(pcg, ecg, ctg, bidmc, i)

        if sample:
            dataset.append(sample)

    return dataset


# ==============================
# SAVE
# ==============================
def save_dataset(data):
    with open("data/final_dataset.pkl", "wb") as f:
        pickle.dump(data, f)

    print("✅ FINAL DATASET READY")


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    data = build_dataset()
    save_dataset(data)