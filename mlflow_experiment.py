
import pandas as pd
import mlflow
import mlflow.sklearn
import boto3
from mlflow import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("http://localhost:5000")

# -----------------------------
# 1. Load data
# -----------------------------

data = pd.read_csv("data/students.csv")

X = data[
    [
        "hours_studied",
        "attendance",
        "previous_score",
        "sleep_hours"
    ]
]

y = data["passed"]


# -----------------------------
# 2. Split data
# -----------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42
)


# -----------------------------
# 3. MLflow experiment
# -----------------------------

mlflow.set_experiment("student-prediction")

c_values = [0.01, 0.1, 1, 10, 100]

best_accuracy = 0
best_c = None
best_model = None
best_run_id = None


# -----------------------------
# 4. Train experiments
# -----------------------------

for c in c_values:

    with mlflow.start_run() as run:

        model = LogisticRegression(C=c)

        # Train
        model.fit(X_train, y_train)

        # Validate
        val_predictions = model.predict(X_val)

        val_accuracy = accuracy_score(
            y_val,
            val_predictions
        )

        # Log experiment information
        mlflow.log_param("C", c)

        mlflow.log_metric(
            "validation_accuracy",
            val_accuracy
        )

        mlflow.sklearn.log_model(
            model,
            "model"
        )

        print(
            f"C={c:<6} "
            f"Validation Accuracy={val_accuracy:.3f}"
        )

        # Track best model
        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy
            best_c = c
            best_model = model
            best_run_id = run.info.run_id


# -----------------------------
# 5. Show best model
# -----------------------------

print("\n==============================")
print("BEST MODEL")
print("==============================")

print(f"Best C: {best_c}")
print(f"Validation Accuracy: {best_accuracy:.3f}")
print(f"MLflow Run ID: {best_run_id}")


# -----------------------------
# 6. Test ONLY the best model
# -----------------------------

test_predictions = best_model.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

print(f"Test Accuracy: {test_accuracy:.3f}")

# -----------------------------
# 7. Register the best model
# -----------------------------

model_uri = f"runs:/{best_run_id}/model"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name="StudentPassPredictor"
)

print("\n==============================")
print("MODEL REGISTERED")
print("==============================")

print(f"Model: StudentPassPredictor")
print(f"Version: {registered_model.version}")

# -----------------------------
# 8. Set model alias
# -----------------------------

client = MlflowClient()

client.set_registered_model_alias(
    name="StudentPassPredictor",
    alias="champion",
    version=registered_model.version
)

print(f"Alias 'champion' -> Version {registered_model.version}")
