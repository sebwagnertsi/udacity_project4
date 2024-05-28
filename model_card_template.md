# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

A Random Forest Model using the standard scikit-learn parameters.
Trained with scikit-learn 1.3.0

## Intended Use

Classify income in two groups (<50k and >= 50k) based on the attributes: age,workclass,fnlgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country

## Training Data

The census data as provided by the project starter kit. 
Original source: https://archive.ics.uci.edu/dataset/20/census+income
Initial Modifications: Removed whitespaces from the file.

## Evaluation Data

I split the data into a train/test and validation set.
The train/test set contains 27093 entries, the validation set contains 5468 entries.

## Metrics

When trained with 70% of the (train/test) data, the remaining 30% yield the following results:
Precision: 0.7376330619912336, Recall: 0.5943491422805247, F1: 0.6582844369935738

10-Fold Cross Validation yields the following F1 scores: [0.6460251  0.6705298  0.68320926 0.65767285 0.66947014 0.669967
 0.67304625 0.67619849 0.70042918 0.66832918]

The results of the Model on the validation set are:
Precision: 0.7347826086956522, Recall: 0.6310679611650486, F1: 0.678987545198875

## Ethical Considerations

Some attributes are ethically problematic and should not be used.
