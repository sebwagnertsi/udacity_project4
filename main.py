from fastapi import FastAPI
from typing import Union
from pydantic import BaseModel

from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from ml.model import inference_from_df, inference_from_df_with_labelconversion
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
    capital_loss: int
    hours_per_week: int
    native_country: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    'age': 25,
                    'workclass': 'State-gov',
                    'fnlgt': 77516,
                    'education': 'Bachelors',
                    'education_num': 13,
                    'marital_status': 'Never-married',
                    'occupation': 'Adm-clerical',
                    'relationship': 'Not-in-family',
                    'race': 'White',
                    'sex': 'Male',
                    'capital_gain': 2174,
                    'capital_loss': 0,
                    'hours_per_week': 40,
                    'native_country': 'United-States'
                },
                {'age': 55,
                 'workclass': 'Private',
                 'fnlgt': 159449,
                 'education': 'Masters',
                 'education_num': 13,
                 'marital_status': 'Married-civ-spouse',
                 'occupation': 'Exec-managerial',
                 'relationship': 'Husband',
                 'race': 'White',
                 'sex': 'Male',
                 'capital_gain': 7500,
                 'capital_loss': 0,
                 'hours_per_week': 40,
                 'native_country': 'United-States'
                 }
            ]
        }
    }


# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.


@app.get("/", response_class=HTMLResponse)
async def say_hello():
    # return {"greeting": "Hello dear Reviewing team! This is the web interface for my project 4 submission."}
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

    print(item)
    result = inference_from_df_with_labelconversion(conv_item_to_df(item))

    return result[0]
