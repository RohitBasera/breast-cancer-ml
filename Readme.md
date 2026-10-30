# Breast Cancer Prediction using Machine Learning

Classifies tumors as malignant or benign using the scikit-learn Breast Cancer Wisconsin dataset (569 samples, 30 features).

## What it does
- Tunes Logistic Regression, SVM and Random Forest with GridSearchCV (5-fold CV)
- Automatically selects the best model by cross-validated F1 score
- Evaluates on an unseen test set (accuracy, precision, recall, F1, ROC-AUC, confusion matrix)
- Validates with repeated stratified cross-validation on all data (5 folds x 3 repeats)
- Shows the most influential features

## How to run
pip install -r requirements.txt
python train.py
python predict.py

## Results

Best model: [PASTE BEST MODEL NAME]

Tuned model comparison (5-fold CV, F1):

| Model | CV F1 |
|---|---|
| Logistic Regression | [paste] |
| SVM | [paste] |
| Random Forest | [paste] |

Final model on the unseen test set (114 samples):

| Metric | Score |
|---|---|
| Accuracy | 0.974 |
| Precision (benign) | 0.973 |
| Recall (benign) | 0.986 |
| F1 (benign) | 0.979 |
| ROC-AUC | 0.996 |
| Repeated CV accuracy (all data) | 0.974 (+/- 0.013) |
## Scalability demo
To demonstrate scalability, the trained model was applied to 24,999 simulated patients generated from the statistics of the real dataset (8,841 predicted malignant, 16,158 predicted benign). This is a demonstration only; all reported accuracy comes from the real 569 patients. Predictions are in `outputs/simulated_predictions.csv`.


Most influential features: worst texture, worst radius, worst concave points.

Plots are in the `outputs/` folder.


## Author 
ROHIT BASERA