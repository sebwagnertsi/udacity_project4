import requests
from main import conv_item_to_df
from ml.model import inference_from_df_with_labelconversion



def test_api_request_unsuccessful():
    url = 'http://localhost:8000/predict'
    # url = 'https://udacity-project4-mamn.onrender.com/predict'
    data = request_person_two()
    data['additional_attribute'] = 'this should not be here'

    response = requests.post(url, json=data)
    print(response.text)
    print(response.json())

def request_person_one():
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
def request_person_two():
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

def test_inference_from_df(request_object):
    # res = inference_from_df(conv_item_to_df(request_object))
    res = inference_from_df_with_labelconversion(conv_item_to_df(request_object))
    print(res)

def test_api_request(data):
    url = 'https://udacity-project4-mamn.onrender.com/predict'
    response = requests.post(url, json=data)
    print("Status code:", response.status_code)
    # print("Response text:", response.text)
    print("Response json:", response.json())


if __name__=='__main__':
    
    print("##### Test API Request:")
    test_api_request({
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
    })