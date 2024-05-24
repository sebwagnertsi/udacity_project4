from fastapi import FastAPI
# Import Union since our Item object will have tags that can be strings or a list.
from typing import Union 
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel

from fastapi import HTTPException

# # Declare the data object with its components and their type.
# class TaggedItem(BaseModel):
#     name: str
#     tags: Union[str, list] 
#     item_id: int

# class Data(BaseModel):
#     feature_1: float
#     feature_2: str

# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}


# @app.post("/ingest_data/")
# async def ingest_data(item: Data):

#     if item.feature_1 <0:
#         raise HTTPException(status_code=400, detail="Feature 1 must be greater than 0")
#     if len(item.feature_2)>280:
#         raise HTTPException(status_code=400, detail="Feature 2 must be less than 280 characters")
#     return item

# @app.get("/items/{item_id}")
# async def get_items(item_id: int, count: int = 1):
#     return {"fetch": f"Fetched {count} of {item_id}"}