import warnings
import mlflow
import mlflow.sklearn
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

# =========================================================
# SET MLFLOW + DAGSHUB
# =========================================================

mlflow.set_tracking_uri("https://dagshub.com/lisaapLI/ModellingExspLilisApr.mlflow")
mlflow.set_experiment("Diabetes_Classification_CI")

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv('diabetes_clean.csv')

print("Ukuran Data :", df.shape)
print(df.head())

# =========================================================
# FEATURE DAN TARGET
# =========================================================

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================================
# TRAINING + LOGGING
# =========================================================

mlflow.sklearn.autolog()

with mlflow.start_run(nested=False, run_name="RandomForest_Diabetes_CI"):

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    print("\n=== HASIL EVALUASI ===")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    print("\n=== CONFUSION MATRIX ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred))