import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

print("🚀 Starting NeuroSense training")

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "parkinsons.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")

print("📂 Loading dataset:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["name", "status"])
y = df["status"]

print("🔀 Splitting data")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("🌲 Training RandomForest")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print("🧪 Evaluating")
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print(f"✅ Accuracy: {acc:.4f}")

joblib.dump(model, MODEL_PATH)
print(f"💾 Model saved to {MODEL_PATH}")
