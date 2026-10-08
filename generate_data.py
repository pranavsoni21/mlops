import numpy as np
import pandas as pd

np.random.seed(42)

n_students = 1000

# Generate features
hours_studied = np.random.uniform(1, 10, n_students)
attendance = np.random.uniform(40, 100, n_students)
previous_score = np.random.uniform(30, 100, n_students)
sleep_hours = np.random.uniform(4, 9, n_students)

# Create a score that determines probability of passing
score = (
    0.35 * hours_studied
    + 0.02 * attendance
    + 0.03 * previous_score
    + 0.05 * sleep_hours
    + np.random.normal(0, 0.5, n_students)
)

# Convert score into pass/fail
passed = (score > 5.5).astype(int)

# Create dataframe
data = pd.DataFrame({
    "hours_studied": hours_studied,
    "attendance": attendance,
    "previous_score": previous_score,
    "sleep_hours": sleep_hours,
    "passed": passed
})

# Save dataset
data.to_csv("data/students.csv", index=False)

print(f"Generated {len(data)} students")
print("\nFirst 5 rows:")
print(data.head())

print("\nClass distribution:")
print(data["passed"].value_counts())
