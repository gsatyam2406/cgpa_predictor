from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def train_model():
    df = pd.read_csv("student_cgpa_dataset.csv")
    X = df[['prev_cgpa','attendance_percent','assignments_completed',
            'midterm_score','lab_score','extracurricular_activities']]
    y = df['predicted_cgpa']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, "cgpa_model.pkl")
    return model

try:
    model = joblib.load("cgpa_model.pkl")
except:
    model = train_model()

class Student(BaseModel):
    prev_cgpa: float
    attendance_percent: float
    assignments_completed: float
    midterm_score: float
    lab_score: float
    extracurricular_activities: float

@app.get("/")
def home():
    return {"message": "CGPA Predictor API running"}

@app.post("/predict")
def predict(student: Student):
    data = [[
        student.prev_cgpa,
        student.attendance_percent,
        student.assignments_completed,
        student.midterm_score,
        student.lab_score,
        student.extracurricular_activities
    ]]
    prediction = model.predict(data)
    return {"predicted_cgpa": round(float(prediction[0]), 2)}
