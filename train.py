import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# You need your CSV here too
df = pd.read_csv("student_cgpa_dataset.csv")

X = df[['prev_cgpa','attendance_percent','assignments_completed',
        'midterm_score','lab_score','extracurricular_activities']]

y = df['predicted_cgpa']  # or 'current_cgpa' — check your CSV column name

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, "cgpa_model.pkl")
print("Model saved!")