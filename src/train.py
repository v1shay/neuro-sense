print("NEUROSENSE START")
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# =============================
# Absolute paths 
# =============================

# This finds the project root no matter where you run the file from
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "parkinsons.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "model.joblib")


# =============================
# Training logic
# =============================

def main():
    print("Starting NeuroSense training...")
    print("Project root:", PROJECT_ROOT)
    print("Data path:", DATA_PATH)
    print("Model will be saved to:", MODEL_PATH)

    # Load dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print("Dataset loaded with shape:", df.shape)

    # Split features and label
    X = df.drop(columns=["name", "status"])
    y = df["status"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)
    print("Model trained successfully.")

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Save model
    joblib.dump(model, MODEL_PATH)
    print("Model saved successfully.")


# =============================
# Entry point 
# =============================

if __name__ == "__main__":
    main()
