import json
import pytest
from fastapi.testclient import TestClient

from main import app
client = TestClient(app)

@pytest.fixture
def request_object():
    return {
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
        }

def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200

def test_prediction():
    data = json.dumps(request_object)
    r = client.post("/predict", data=data)
    assert r.json()["native_country"] == 'United-States'

def test_prediction_fail_too_many_attributes(request_object):
    data = json.dumps(request_object)
    r = client.post("/predict", data=data)
    assert r.status_code == 400

# def test_get():
#     r = client.get("/items/0")
#     assert r.status_code == 200
#     assert r.json()["fetch"] == "Fetched 1 of 0"

# def test_get_path():
#     r = client.get("/items/42")
#     assert r.status_code == 200
#     assert r.json() == {"fetch": "Fetched 1 of 42"}


# def test_get_path_query():
#     r = client.get("/items/42?count=5")
#     assert r.status_code == 200
#     assert r.json() == {"fetch": "Fetched 5 of 42"}


# def test_get_malformed():
#     r = client.get("/items")
#     assert r.status_code != 200
    