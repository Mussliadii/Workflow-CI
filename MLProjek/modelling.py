import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from mlflow.models.signature import infer_signature

# === Setup path ===
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "titanic_preprocessing", "titanic_preprocessed_train.csv")

# === Load dataset ===
df = pd.read_csv(csv_path)

# === Encode categorical features ===
label_cols = df.select_dtypes(include='object').columns
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# === Handle integer columns with missing values ===
for col in df.select_dtypes(include='int').columns:
    if df[col].isnull().any():
        df[col] = df[col].astype(float)

# === Split features and target ===
X = df.drop(columns=["Survived"])
y = df["Survived"]

print("Fitur:", X.columns.tolist())
print("Jumlah fitur:", X.shape[1])

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === MLflow tracking ===
with mlflow.start_run() as run:
    
    # Buat pipeline dengan scaler dan model
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500)
    )
    
    # Latih model
    model.fit(X_train, y_train)
    
    # Prediksi
    y_pred = model.predict(X_test)

    # Evaluasi model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log parameter dan metrik
    mlflow.log_param("model_type", "Logistic Regression with Scaler")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Log signature dan contoh input
    signature = infer_signature(X_train, model.predict(X_train))
    input_example = X_train.head(5)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )

    # Output hasil
    print(f"Akurasi   : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"Run ID    : {run.info.run_id}")
