from train_model import train_and_store_model
from preprocessing import clean_data
from evaluation import run_all_slices_evaluation, run_train_test_evaluation, run_cross_validation, run_validation
import warnings

warnings.filterwarnings('ignore')


def run_full_process():
    print("Running full process")

    # Preprocess the data
    print("Cleaning and preprocessing data.")
    X, y = clean_data()

    # Evaluate the model performance
    print("Evaluating model performance ######################")
    precision, recall, fbeta = run_train_test_evaluation()

    run_cross_validation()

    # Train final model
    model = train_and_store_model()

    print("Validating model performance ######################")
    run_validation(model)

    # Run the all slices evaluation with model
    run_all_slices_evaluation(model)


if __name__ == '__main__':
    run_full_process()
