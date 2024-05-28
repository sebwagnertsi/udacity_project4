import pytest
import numpy as np
from ml.data import preprocess_data
from ml.model import inference_from_df_with_labelconversion, train_model, compute_model_metrics
from main import conv_item_to_df


@pytest.fixture
def entity():
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
def X():
    return np.array([
        [1, 2, 1],
        [2, 3, 1],
        [4, 5, 0],
        [2, 3, 1],
        [2, 7, 1],
        [9, 6, 6],
    ])


@pytest.fixture
def y():
    return np.array([0, 0, 1, 0, 1, 1])


@pytest.fixture
def preds():
    return np.array([1, 0, 1, 1, 0, 1])


def test_preprocessing(entity):
    df = conv_item_to_df(entity)
    X, y, _, _ = preprocess_data(df)

    assert X.shape[0] == 1
    assert X.shape[1] == 108  # after conversion we expect 108 columns


def test_preprocessing_with_y(entity):
    entity['salary'] = '<50K'
    df = conv_item_to_df(entity)
    X, y, _, _ = preprocess_data(df, training=True)

    assert y.shape[0] == 1
    assert y == 0


def test_inference(entity):

    res = inference_from_df_with_labelconversion(conv_item_to_df(entity))
    # doesnt matter if <= or > 50K as long as it is one of them
    assert '50K' in res[0]


def test_train_model(X, y):
    model = train_model(X, y)
    assert model is not None


def test_metrics(y, preds):

    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.6666666)
    assert fbeta == pytest.approx(0.5714286)
