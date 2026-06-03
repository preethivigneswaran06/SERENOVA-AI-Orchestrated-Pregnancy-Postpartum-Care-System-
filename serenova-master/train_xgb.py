import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier

# ==============================
# LOAD DATA
# ==============================
data = pickle.load(open("data/final_dataset.pkl", "rb"))

# ==============================
# CONVERT TO TABLE
# ==============================
rows = []
labels = []

for d in data:
    rows.append(d["features"])
    labels.append(d["labels"]["fetal_hypoxia"])

df = pd.DataFrame(rows)
y = labels

print("Shape:", df.shape)

# ==============================
# HANDLE MISSING
# ==============================
df = df.fillna(0)

# ==============================
# 🚨 REMOVE LEAKING FEATURES
# ==============================
leak_cols = [
    "fhr_mean",
    "fhr_variability",
    "RESP_value",
    "HR_value"
]

for col in leak_cols:
    if col in df.columns:
        df = df.drop(columns=[col])

print("After removing leakage:", df.shape)

# ==============================
# TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=42
)

# ==============================
# MODEL
# ==============================
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.08,
    scale_pos_weight=3.5,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# ==============================
# PREDICT
# ==============================
y_pred = model.predict(X_test)

# ==============================
# RESULTS
# ==============================
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==============================
# SAVE MODEL
# ==============================
pickle.dump(model, open("data/xgb_model.pkl", "wb"))

print("\n✅ Model saved")