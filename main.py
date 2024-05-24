from fastapi import FastAPI
from typing import Union
from pydantic import BaseModel

from fastapi import HTTPException


class Data(BaseModel):
    feature_1: float
    feature_2: str

# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}


@app.post("/predict/")
async def predict(item: Data):

    # if item.feature_1 <0:
    #     raise HTTPException(status_code=400, detail="Feature 1 must be greater than 0")
    # if len(item.feature_2)>280:
    #     raise HTTPException(status_code=400, detail="Feature 2 must be less than 280 characters")
    return item
