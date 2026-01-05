import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# =========================
# CONFIG
# =========================
DATA_PATH = "data/parkinsons.csv"
MODEL_PATH = "src/model.joblib"
RANDOM_STATE = 42

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

# Some versions include a 'name' column, some don't
if "name" in df.columns:
    X = df.drop(columns=["name", "status"])
else:
    X = df.drop(columns=["status"])

y = df["status"]

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

# =========================
# SCALE FEATURES
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# LOAD MODEL
# =========================
model = joblib.load(MODEL_PATH)

# =========================
# PREDICTIONS
# =========================
y_pred = model.predict(X_test_scaled)

# Some models support predict_proba, others decision_function
if hasattr(model, "predict_proba"):
    y_scores = model.predict_proba(X_test_scaled)[:, 1]
else:
    y_scores = model.decision_function(X_test_scaled)

# =========================
# METRICS
# =========================
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# =========================
# PRINT RESULTS (PASTE THIS TO ME)
# =========================
print("\n==============================")
print("NEUROSENSE MODEL EVALUATION")
print("==============================\n")

print(f"Accuracy: {accuracy:.6f}\n")

print("Confusion Matrix:")
print(cm, "\n")

print("Classification Report:")
print(report)

print(f"ROC AUC: {roc_auc:.6f}")

# =========================
# SAVE FIGURES
# =========================
plt.figure()
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=300)
plt.close()

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("results/roc_curve.png", dpi=300)
plt.close()

print("\nSaved figures:")
print(" - results/confusion_matrix.png")
print(" - results/roc_curve.png")
