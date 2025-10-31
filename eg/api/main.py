from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd

app = FastAPI()

model = xgb.XGBRegressor()
model.load_model("../models/consumption_model.json")

class Input(BaseModel):
    population: float
    gdp: float
    gdp_per_capita: float
    growth_rate: float

@app.post("/forecast")
def forecast(data: Input):
    df = pd.DataFrame([data.dict()])
    pred = model.predict(df)[0]
    savings = pred * 0.25
    return {
        "predicted_TWh": round(pred, 2),
        "savings_TWh": round(savings, 2),
        "advice": "Go green!" if pred > 500 else "Sustainable"
    }
