import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load data
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


# Split data
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


# Hyperparameters we want to test
c_values = [0.01, 0.1, 1, 10, 100]


print("Validation data:")
print(
    pd.DataFrame({
        "hours": X_val["hours_studied"].values,
        "attendance": X_val["attendance"].values,
        "actual": y_val.values
    })
)

best_accuracy = 0
best_c = None
best_model = None

# Run experiments
for c in c_values:

    model = LogisticRegression(C=c)

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_c = c
        best_model = model

print("\nBest model:")
print(f"C = {best_c}")
print(f"Validation Accuracy = {best_accuracy:.2f}")


test_predictions = best_model.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

print(f"Test Accuracy = {test_accuracy:.2f}")
