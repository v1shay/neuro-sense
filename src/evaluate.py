import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Config
# ----------------------------
DATA_PATH = "data/parkinsons.csv"
MODEL_PATH = "src/model.joblib"
RANDOM_STATE = 42

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["name", "status"])
y = df["status"]

# ----------------------------
# Train/test split (MUST match training)
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

# ----------------------------
# Scale features
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# Load trained model
# ----------------------------
model = joblib.load(MODEL_PATH)

# ----------------------------
# Evaluate
# ----------------------------
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# ----------------------------
# Print results
# ----------------------------
print("\n=== MODEL EVALUATION ===")
print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)
