import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

N = 25000  # approximate number of simulated patients
rng = np.random.default_rng(42)

model = joblib.load("outputs/model.joblib")
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target

# Simulate patients from the statistics (mean + covariance) of the real data, per class
parts = []
for cls in [0, 1]:
    Xc = X[y == cls]
    n = int(N * (y == cls).mean())
    sim = rng.multivariate_normal(Xc.mean().values, np.cov(Xc.values, rowvar=False), size=n)
    sim = np.clip(sim, Xc.min().values, Xc.max().values)  # keep values in realistic range
    df = pd.DataFrame(sim, columns=X.columns)
    df["simulated_class"] = data.target_names[cls]
    parts.append(df)

sim_df = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
features = sim_df[X.columns]

# Predict for all patients at once
preds = model.predict(features)
result = pd.DataFrame({
    "patient_id": range(1, len(sim_df) + 1),
    "predicted": [data.target_names[p] for p in preds],
})
if hasattr(model, "predict_proba"):
    result["confidence"] = model.predict_proba(features).max(axis=1).round(4)

result.to_csv("outputs/simulated_predictions.csv", index=False)

print(f"Predicted {len(result)} simulated patients\n")
print(result.head(10).to_string(index=False))
print("\nPrediction counts:")
print(result["predicted"].value_counts())
print("\nSaved to outputs/simulated_predictions.csv")
print("NOTE: patients are SIMULATED. Do not report accuracy from this data.")
