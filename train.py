import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, RepeatedStratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, ConfusionMatrixDisplay, RocCurveDisplay,
)

os.makedirs("outputs", exist_ok=True)

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target  # 0 = malignant, 1 = benign
print("Dataset shape:", X.shape)

# ---------------------------------------------------------------
# 2. TRAIN / TEST SPLIT
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------------------------------------------
# 3. TUNE ALL 3 MODELS WITH GRIDSEARCHCV
# ---------------------------------------------------------------
search_space = {
    "Logistic Regression": (
        Pipeline([("scaler", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=5000))]),
        {"clf__C": [0.01, 0.1, 1, 10, 100]},
    ),
    "SVM": (
        Pipeline([("scaler", StandardScaler()),
                  ("clf", SVC(random_state=42))]),
        {"clf__C": [0.1, 1, 10, 100],
         "clf__gamma": ["scale", 0.01, 0.001]},
    ),
    "Random Forest": (
        Pipeline([("scaler", StandardScaler()),
                  ("clf", RandomForestClassifier(random_state=42))]),
        {"clf__n_estimators": [100, 200, 300],
         "clf__max_depth": [None, 5, 10],
         "clf__min_samples_split": [2, 5]},
    ),
}

tuned = {}
cv_means = {}
print("\n--- Tuning each model (5-fold CV, F1 score) ---")
for name, (pipe, grid_params) in search_space.items():
    grid = GridSearchCV(pipe, grid_params, cv=5, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)
    tuned[name] = grid.best_estimator_
    cv_means[name] = grid.best_score_
    print(f"{name}: best CV F1 = {grid.best_score_:.4f}, params = {grid.best_params_}")

# Bar chart of tuned CV scores
plt.figure(figsize=(7, 4))
plt.bar(list(cv_means.keys()), list(cv_means.values()),
        color=["#4C72B0", "#55A868", "#C44E52"])
plt.ylim(0.9, 1.0)
plt.title("Tuned Models: 5-Fold CV F1 Score")
plt.ylabel("F1 score")
plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 4. PICK THE BEST MODEL AUTOMATICALLY
# ---------------------------------------------------------------
best_name = max(cv_means, key=cv_means.get)
best_model = tuned[best_name]
print(f"\nBest model: {best_name}")

# ---------------------------------------------------------------
# 5. FINAL EVALUATION ON UNSEEN TEST DATA
# ---------------------------------------------------------------
y_pred = best_model.predict(X_test)
if hasattr(best_model, "predict_proba"):
    y_score = best_model.predict_proba(X_test)[:, 1]
else:
    y_score = best_model.decision_function(X_test)

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "ROC-AUC": roc_auc_score(y_test, y_score),
}
print("\n--- Test set results ---")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
print("\n", classification_report(y_test, y_pred, target_names=data.target_names))

# More reliable estimate: 5-fold CV repeated 3 times on ALL data
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
rep_scores = cross_val_score(best_model, X, y, cv=rcv, scoring="accuracy")
print(f"Repeated CV accuracy (all data, 15 runs): "
      f"{rep_scores.mean():.4f} (+/- {rep_scores.std():.4f})")
metrics["Repeated CV Accuracy"] = rep_scores.mean()

pd.Series(metrics).round(4).to_csv("outputs/metrics.csv", header=["score"])

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=data.target_names, cmap="Blues"
)
plt.title(f"Confusion Matrix ({best_name})")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)
plt.close()

# ROC curve
RocCurveDisplay.from_estimator(best_model, X_test, y_test)
plt.title(f"ROC Curve ({best_name})")
plt.tight_layout()
plt.savefig("outputs/roc_curve.png", dpi=150)
plt.close()

# Feature importance (Random Forest) or coefficients (Logistic Regression)
clf = best_model.named_steps["clf"]
if hasattr(clf, "feature_importances_"):
    imp = pd.Series(clf.feature_importances_, index=X.columns)
    title = "Top 10 Most Important Features"
elif hasattr(clf, "coef_"):
    imp = pd.Series(np.abs(clf.coef_[0]), index=X.columns)
    title = "Top 10 Features by |Coefficient|"
else:
    imp = None
if imp is not None:
    imp = imp.sort_values(ascending=False).head(10)
    plt.figure(figsize=(8, 5))
    imp[::-1].plot(kind="barh", color="teal")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150)
    plt.close()
    print("\nTop features:", list(imp.index[:3]))

# ---------------------------------------------------------------
# 6. SAVE MODEL
# ---------------------------------------------------------------
joblib.dump(best_model, "outputs/model.joblib")
print("\nDone! Model and plots saved in the 'outputs' folder.")