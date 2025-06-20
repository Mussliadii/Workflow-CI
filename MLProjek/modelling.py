import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from mlflow.models.signature import infer_signature

# =======================
# SETUP & DATA LOADING
# =======================
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "titanic_preprocessing", "titanic_preprocessed_train.csv")

df = pd.read_csv(csv_path)

# =======================
# DATA PREPROCESSING
# =======================
# Encode categorical columns
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col].astype(str))

# Convert integer columns with NaN to float
int_cols = df.select_dtypes(include='int').columns
for col in int_cols:
    if df[col].isnull().any():
        df[col] = df[col].astype(float)

# =======================
# FEATURE TARGET SPLIT
# =======================
X = df.drop(columns=["Survived"])
y = df["Survived"]

print("🧾 Fitur digunakan:", X.columns.tolist())
print(f"🔢 Jumlah fitur: {X.shape[1]}")

# =======================
# SPLIT TRAIN - TEST
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =======================
# MODEL TRAINING + MLFLOW
# =======================
with mlflow.start_run() as run:
    
    # Buat pipeline preprocessing + model
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500)
    )
    
    # Latih model
    pipeline.fit(X_train, y_train)

    # Prediksi
    y_pred = pipeline.predict(X_test)

    # Evaluasi
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    # Logging ke MLflow
    mlflow.log_param("model_type", "Logistic Regression with Scaler")
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)

    # Signature & contoh input
    signature = infer_signature(X_train, pipeline.predict(X_train))
    input_example = X_train.head(5)

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )

    # Output hasil evaluasi
    print("\n📊 Hasil Evaluasi Model:")
    for name, val in metrics.items():
        print(f"{name.capitalize():<10}: {val:.4f}")

    print(f"\n🧪 Run ID: {run.info.run_id}")
