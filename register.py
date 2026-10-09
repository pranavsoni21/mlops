
import argparse
import os
import sys

import mlflow
from mlflow import MlflowClient


MODEL_NAME = "StudentPassPredictor"


def main():
    parser = argparse.ArgumentParser(
        description="Register an MLflow model candidate."
    )
    parser.add_argument(
        "--run-id",
        default=os.getenv("MODEL_RUN_ID"),
        help="MLflow run ID produced by train.py",
    )
    parser.add_argument(
        "--min-validation-accuracy",
        type=float,
        default=float(
            os.getenv("MIN_VALIDATION_ACCURACY", "0.85")
        ),
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Move the champion alias to this model version.",
    )
    args = parser.parse_args()

    tracking_uri = os.getenv("MLFLOW_SERVER_URL")
    if not tracking_uri:
        parser.error("Set MLFLOW_SERVER_URL.")

    if not args.run_id:
        parser.error("Provide --run-id or set MODEL_RUN_ID.")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Ensure the run exists and passed the quality gate.
    run = client.get_run(args.run_id)
    validation_accuracy = run.data.metrics.get(
        "validation_accuracy"
    )

    if validation_accuracy is None:
        raise ValueError(
            f"Run {args.run_id} has no validation_accuracy metric."
        )

    print(f"Candidate run: {args.run_id}")
    print(f"Validation accuracy: {validation_accuracy:.3f}")

    if validation_accuracy < args.min_validation_accuracy:
        print("Candidate failed the quality gate; not registering.")
        sys.exit(1)

    model_uri = f"runs:/{args.run_id}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    print(f"Registered model: {MODEL_NAME}")
    print(f"Version: {registered_model.version}")

    if not args.promote:
        print(
            "Candidate registered but not promoted. "
            "Use --promote after release approval."
        )
        return

    # Explicit release action: move the alias only when requested.
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=registered_model.version,
    )

    print(
        f"Alias 'champion' -> version {registered_model.version}"
    )


if __name__ == "__main__":
    main()
