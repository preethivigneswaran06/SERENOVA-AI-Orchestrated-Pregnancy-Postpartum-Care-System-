import pickle
from src.pipeline.orchestrator import Orchestrator

data = pickle.load(open("data/final_dataset.pkl", "rb"))

sample = data[0]
features = sample["features"]

feature_keys = sorted(features.keys())

system = Orchestrator(feature_keys)
output = system.run(features)

print("\n==============================")
print("🧠 SERENOVA FINAL OUTPUT")
print("==============================")

if "status" in output and output["status"] == "rejected":
    print("❌ Rejected")
else:
    print("Conditions:", output["conditions"])
    print("Decision:", output["decision"])
    print("Alert:", output["alert"])
    print("Explanation:", output["explanation"])