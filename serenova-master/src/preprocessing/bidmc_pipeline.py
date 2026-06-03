import os
import wfdb
import numpy as np
import pickle
from tqdm import tqdm


# ==============================
# LOAD WFDB RECORD
# ==============================
def load_record(path):
    record = wfdb.rdrecord(path)
    return record.p_signal, record.fs, record.sig_name


# ==============================
# LOAD BREATH FILE
# ==============================
def load_breath(path):
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r") as f:
            data = f.readlines()
        return len(data)  # simple count
    except:
        return None


# ==============================
# WINDOWING
# ==============================
def create_windows(signal, fs, sec=30):
    size = int(sec * fs)
    windows = []

    for i in range(0, len(signal), size):
        w = signal[i:i+size]
        if len(w) == size:
            windows.append(w)

    return windows


# ==============================
# FEATURE EXTRACTION
# ==============================
def extract_features(window, sig_names):
    features = {}

    for i, name in enumerate(sig_names):
        col = window[:, i]

        features[f"{name}_mean"] = float(np.mean(col))
        features[f"{name}_std"] = float(np.std(col))

    return features


# ==============================
# MAIN PIPELINE
# ==============================
def process_bidmc(data_path):
    dataset = []

    files = os.listdir(data_path)

    # get base records (ignore 'n' versions)
    records = []
    for f in files:
        if f.endswith(".hea") and not f.endswith("n.hea"):
            name = f.replace(".hea", "")
            if name + ".dat" in files:
                records.append(name)

    print(f"✅ Records found: {len(records)}")

    for rec in tqdm(records):
        try:
            main_path = os.path.join(data_path, rec)
            num_path = os.path.join(data_path, rec + "n")

            # load main
            signal, fs, sig_names = load_record(main_path)

            # load numerics (optional)
            if os.path.exists(num_path + ".hea"):
                num_signal, _, num_names = load_record(num_path)
            else:
                num_signal = None
                num_names = []

            # load breath
            breath_count = load_breath(os.path.join(data_path, rec + ".breath"))

            windows = create_windows(signal, fs)

            sample = {
                "id": rec,
                "windows": []
            }

            for i, w in enumerate(windows):
                feat = extract_features(w, sig_names)

                # add numerics
                if num_signal is not None:
                    nw = num_signal[i] if i < len(num_signal) else None
                    if nw is not None:
                        for j, n in enumerate(num_names):
                            feat[f"{n}_value"] = float(nw[j])

                # add breath info
                if breath_count is not None:
                    feat["breath_events"] = breath_count

                sample["windows"].append({
                    "window_id": i,
                    "features": feat
                })

            dataset.append(sample)

        except Exception as e:
            print(f"❌ Error: {rec} → {e}")

    return dataset


# ==============================
# SAVE
# ==============================
def save_dataset(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)

    print(f"✅ Saved: {path}")