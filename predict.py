import joblib
from sklearn.datasets import load_breast_cancer

model = joblib.load("outputs/model.joblib")
data = load_breast_cancer(as_frame=True)

samples = data.data.iloc[:5]
actual = data.target.iloc[:5]
preds = model.predict(samples)

for i in range(5):
    line = f"Patient {i+1}: predicted = {data.target_names[preds[i]]}"
    if hasattr(model, "predict_proba"):
        conf = model.predict_proba(samples)[i].max()
        line += f" (confidence {conf:.2%})"
    line += f", actual = {data.target_names[actual.iloc[i]]}"
    print(line)