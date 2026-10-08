import mlflow
import mlflow.sklearn
import pandas as pd

model = mlflow.sklearn.load_model(
    "models:/StudentPassPredictor@champion"
)

student = pd.DataFrame([{
    "hours_studied": 6.0,
    "attendance": 85.0,
    "previous_score": 75.0,
    "sleep_hours": 7.0
}])

prediction = model.predict(student)

print("Prediction:", prediction[0])

if prediction[0] == 1:
    print("Student is predicted to PASS")
else:
    print("Student is predicted to FAIL")
