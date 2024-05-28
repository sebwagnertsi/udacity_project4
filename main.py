from fastapi import FastAPI
from typing import Union
from pydantic import BaseModel

from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from ml.model import inference_from_df, convert_inf_results_to_label
import pandas as pd

class Data(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss:int
    hours_per_week: int
    native_country: str

# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.
@app.get("/", response_class=HTMLResponse)
async def say_hello():
    #return {"greeting": "Hello dear Reviewing team! This is the web interface for my project 4 submission."}
    return """<html><body>
            <h1>Welcome</h1>
            <p>Hello dear Reviewing team! This is the web interface for my project 4 submission.</p>
            Please refer to <a href='docs'>docs</a> for the endpoint documentation.
            </body></html>
            """

def conv_item_to_df(item: Data) -> pd.DataFrame:
    data = []
    data.append(dict(item))
    data_df = pd.DataFrame(data)
    return data_df

@app.post("/predict/")
async def predict(item: Data):

    result = convert_inf_results_to_label(inference_from_df(conv_item_to_df(item)))

    return result[0]
