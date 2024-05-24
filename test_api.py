import json
from fastapi.testclient import TestClient

from main import app
client = TestClient(app)

def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200

def test_prediction():
    data = json.dumps({
        'feature_1': 20,
        'feature_2': 'hello this is a string'
        })
    r = client.post("/predict", data=data)
    assert r.json()["feature_2"] == 'hello this is a string'

def test_prediction_fail_too_many_attributes():
    data = json.dumps({
        'feature_1': -50,
        'feature_2': 'hello this is a string'
        })
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
    