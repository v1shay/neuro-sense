import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# =========================
# Paths (robust & portable)
# =========================
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "parkinsons.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")


# =========================
# Load dataset
# =========================
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["name", "status"])
y = df["status"]


# =========================
# Train / test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# Train model
# =========================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)


# =========================
# Evaluate model
# =========================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Model accuracy: {accuracy:.4f}")


# =========================
# Save trained model
# =========================
joblib.dump(model, MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
