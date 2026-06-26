from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import joblib
from pydantic import BaseModel


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("cgpa_model.pkl")


class Student(BaseModel):

    prev_cgpa: float
    attendance_percent: float
    assignments_completed: float
    midterm_score: float
    lab_score: float
    extracurricular_activities: float



@app.post("/predict")

def predict(student:Student):


    data=[[
        student.prev_cgpa,
        student.attendance_percent,
        student.assignments_completed,
        student.midterm_score,
        student.lab_score,
        student.extracurricular_activities
    ]]


    prediction=model.predict(data)


    return {

        "predicted_cgpa":
        round(float(prediction[0]),2)

    }