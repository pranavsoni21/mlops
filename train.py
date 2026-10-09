
import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


DATA_PATH = Path(os.getenv("DATA_PATH", "data/students.csv"))
EXPERIMENT_NAME = "student-prediction"
MIN_VALIDATION_ACCURACY = float(
    os.getenv("MIN_VALIDATION_ACCURACY", "0.85")
)
C_VALUES = [0.01, 0.1, 1, 10, 100]

FEATURES = [
    "hours_studied",
    "attendance",
    "previous_score",
    "sleep_hours",
]
TARGET = "passed"


def load_data():
    """Load and validate the training dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + [TARGET]
    missing = set(required_columns) - set(data.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing)}"
        )

    if data[required_columns].isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    if not set(data[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target column must contain only 0 and 1.")

    if len(data) < 10:
        raise ValueError("Dataset is too small for training.")

    return data[FEATURES], data[TARGET]


def split_data(X, y):
    """Create reproducible 60/20/20 train/validation/test splits."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=42,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_candidates(X_train, y_train, X_val, y_val):
    """Train candidates and return the best validation run."""
    best_accuracy = -1.0
    best_c = None
    best_run_id = None

    for c in C_VALUES:
        with mlflow.start_run() as run:
            model = LogisticRegression(C=c, max_iter=1000)
            model.fit(X_train, y_train)

            predictions = model.predict(X_val)
            val_accuracy = accuracy_score(y_val, predictions)

            mlflow.log_params({
                "C": c,
                "model_type": "LogisticRegression",
                "random_state": 42,
                "train_rows": len(X_train),
                "validation_rows": len(X_val),
            })
            mlflow.log_metric(
                "validation_accuracy", val_accuracy
            )
            mlflow.set_tag("git_commit", os.getenv("GITHUB_SHA", "local"))

            mlflow.sklearn.log_model(
                model,
                name="model",
                input_example=X_train.head(3),
            )

            print(
                f"C={c:<6} "
                f"Validation Accuracy={val_accuracy:.3f}"
            )

            # Strict > keeps the first candidate if scores tie.
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_c = c
                best_run_id = run.info.run_id

    return best_c, best_accuracy, best_run_id


def write_github_output(run_id, best_c, accuracy):
    """Expose results to later GitHub Actions steps/jobs."""
    output_file = os.getenv("GITHUB_OUTPUT")

    if output_file:
        with open(output_file, "a", encoding="utf-8") as file:
            file.write(f"best_run_id={run_id}\n")
            file.write(f"best_c={best_c}\n")
            file.write(f"validation_accuracy={accuracy}\n")


def main():
    tracking_uri = os.getenv("MLFLOW_SERVER_URL")

    if not tracking_uri:
        raise ValueError(
            "Set MLFLOW_SERVER_URL to your MLflow tracking server URL."
        )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    best_c, best_accuracy, best_run_id = train_candidates(
        X_train, y_train, X_val, y_val
    )

    print("\n========== BEST CANDIDATE ==========")
    print(f"Best C: {best_c}")
    print(f"Validation Accuracy: {best_accuracy:.3f}")
    print(f"MLflow Run ID: {best_run_id}")

    if best_accuracy < MIN_VALIDATION_ACCURACY:
        print(
            f"QUALITY GATE FAILED: {best_accuracy:.3f} "
            f"< {MIN_VALIDATION_ACCURACY:.3f}"
        )
        sys.exit(1)

    print("QUALITY GATE PASSED")

    # Deliberately do not evaluate the test set during routine CI.
    # Keep it held out for final evaluation after model selection.
    print(
        f"Held-out test set: {len(X_test)} rows "
        "(not evaluated in this CI run)."
    )

    write_github_output(best_run_id, best_c, best_accuracy)


if __name__ == "__main__":
    main()
