import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

# Load dataset
df = pd.read_csv("Train/ml_features.csv")

X = df.drop(columns=["status"])
y = df["status"]

# Five-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Out-of-fold predictions and probabilities
y_pred = cross_val_predict(
    model, X, y, cv=cv, method="predict"
)

y_prob = cross_val_predict(
    model, X, y, cv=cv, method="predict_proba",
    n_jobs=-1
)[:, 1]

# Confusion matrix
cm = confusion_matrix(y, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non-alcoholic", "Alcoholic"]
)

disp.plot(cmap="Blues", values_format="d")
plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()
plt.savefig("Train/confusion_matrix.png", dpi=300)
plt.close()

# ROC curve and AUROC
fpr, tpr, thresholds = roc_curve(y, y_prob)
auc = roc_auc_score(y, y_prob)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"Random Forest (AUROC = {auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Chance level")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Random Forest - ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("Train/roc_curve.png", dpi=300)
plt.close()

print("Confusion matrix:")
print(cm)

print(f"\nAUROC: {auc:.4f}")

print("\nFigures saved:")
print("Train/confusion_matrix.png")
print("Train/roc_curve.png")