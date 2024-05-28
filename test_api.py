import json
import pytest
from fastapi.testclient import TestClient
from main import conv_item_to_df

from main import app
client = TestClient(app)

@pytest.fixture
def request_person_poor():
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

@pytest.fixture
def request_person_rich():
    return {
            'age': 55,
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

def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200

def test_prediction_rich(request_person_rich):
    data = json.dumps(request_person_rich)
    r = client.post("/predict", data=data)
    assert r.text == '">50K"'

def test_prediction_poor(request_person_poor):
    data = json.dumps(request_person_poor)
    r = client.post("/predict", data=data)
    assert r.text == '"<=50K"'

def test_prediction_fail_improper_attribute_values(request_person_poor):
    data = request_person_poor
    data['marital_status'] = 42 # improper type for marital status

    r = client.post("/predict", data=data)

    assert r.status_code >= 300  # should be some kind of error

def test_dataframe_conversion(request_person_poor):    
    data_df = conv_item_to_df(request_person_poor)
    assert data_df.shape == (1, 14)
    assert data_df['age'][0] == 25
    assert data_df['workclass'][0] == 'State-gov'
    assert data_df['fnlgt'][0] == 77516
    assert data_df['education'][0] == 'Bachelors'
    assert data_df['education_num'][0] == 13
    assert data_df['marital_status'][0] == 'Never-married'
    assert data_df['occupation'][0] == 'Adm-clerical'
    assert data_df['relationship'][0] == 'Not-in-family'
