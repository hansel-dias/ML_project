from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel,ValidationError
from src.pipeline.train_pipeline import PredictPipeline
from src.logger import logging
from src.exception import CustomException
import sys
import joblib
import uvicorn
import pandas as pd




app = FastAPI()

# Load the trained model
model = joblib.load("trained_models\model.pkl")


# Mount the "static" directory to serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Use Jinja2Templates to render HTML templates
templates = Jinja2Templates(directory="templates")



class HeartData(BaseModel):
    Age: int
    RestingBP: int
    Cholesterol: int
    MaxHR: int
    Oldpeak: float
    FastingBS:int
    Sex: str
    ChestPainType: str
    RestingECG: str
    ST_Slope:str
    ExerciseAngina: str

# Define the root route
@app.get("/")
async def root(request: Request):
    # Prepare the context to render the HTML template
    context = {"request": request}

    # Render the HTML template with the context
    return templates.TemplateResponse("index.html", context)

@app.post("/predict")
async def process_predict_form(request: Request):
    form_data = await request.form()
    try:
        try:
            heart_data = HeartData(
                Age=int(form_data["age"]),
                RestingBP=int(form_data["resting_bp"]),
                Cholesterol=int(form_data["cholesterol"]),
                MaxHR=int(form_data["max_hr"]),
                Oldpeak=float(form_data["oldpeak"]),
                FastingBS=int(form_data["binary"]),
                Sex=form_data["sex"],
                ChestPainType=form_data["chest_pain_type"],
                RestingECG=form_data["resting_ecg"],
                ST_Slope=form_data["st_slope"],
                ExerciseAngina=form_data["exercise_angina"],
            )

            logging.info(heart_data)
        except ValidationError as e:
            return {"error": e.errors()} 

    except Exception as e:
        raise CustomException(e ,sys)

    predict_data_dict = heart_data.dict()
    predict_data_df = pd.DataFrame(predict_data_dict,index=[0])
    # Predict the outcome

    pred_pipe = PredictPipeline()

    results = pred_pipe.predict(predict_data_df)
    logging.info(results)

    results_list = results.tolist()
    
    for i in results_list:
        if i == 1.0:
            return {"prediction": "Heartdiease Positive"}
        else:
            return {"prediction": "Heartdiease Negative"}



if __name__=="__main__":
    uvicorn.run(app, host="localhost", port=8000)    