# import matplotlib
# matplotlib.use("TkAgg")

import pandas as pd
import sklearn.linear_model as linear_model
import sklearn.cross_validation as cross_validation
import input_preproc
import matplotlib.pyplot as plt
import calculate_metrics
from sklearn import ensemble


CONFIG_EDUCATION = {
    'TARGET': "../../data/anonymization/adults_target_education_num/",
    'OUTPUT': "../../output/anonymization/adults_target_education_num/",
    'INPUT_COLS': [
        "age",
        "fnlwgt",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "workclass",
        "native-country",
        "sex",
        "race",
        "marital-status",
        "relationship",
        "occupation",
        "income",
        "education-num"
     ],
    'TARGET_COL': "education-num"
}


CONFIG_MARITAL = {
    'TARGET': "../../data/anonymization/adults_target_marital_status/",
    'OUTPUT': "../../output/anonymization/adults_target_marital_status/",
    'INPUT_COLS': [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "workclass",
        "native-country",
        "sex",
        "race",
        "relationship",
        "occupation",
        "income",
        "marital-status"
     ],
    'TARGET_COL': "marital-status"
}

config = CONFIG_EDUCATION

INPUT_FILE = "adults_original_dataset.csv"

INPUT_CSV = config['TARGET'] + INPUT_FILE


def runLogisticRegression(input_file):

  encoded_data = input_preproc.readFromDataset(
    input_file,
    config['INPUT_COLS'],
    config['TARGET_COL']
  )

  # LOGISTIC REGRESSION Model
  cls = linear_model.LogisticRegression(
    class_weight="balanced",  # default = None
    max_iter=1000,  # default = 100
    solver="liblinear",  # default = liblinear (can only handle on-vs-rest)
    multi_class="ovr",
    n_jobs=-1
  )


  training_columns = encoded_data.columns.difference([config['TARGET_COL']])

  # DIVIDE THE DATASET INTO TRAIN AND TEST SETS
  X_train, X_test, y_train, y_test = cross_validation.train_test_split(
      encoded_data[training_columns],
      encoded_data[config['TARGET_COL']], train_size=0.80)

  predictions = cls.fit(X_train, y_train).predict(X_test)

  coefs = pd.Series(cls.coef_[0], index=training_columns)
  coefs.sort_values(inplace=True)

  bottom_5_coefs = coefs[:5]
  # print "Least important columns:"
  # print bottom_5_coefs

  top_5_coefs = coefs[len(coefs)-5:]
  # print "Most important columns:"
  # print top_5_coefs

  coefs_to_display = pd.concat([bottom_5_coefs, top_5_coefs])
  print "Most significant coefficients"
  print coefs_to_display

  precision, recall, f1 = calculate_metrics.calculateMetrics(predictions, y_test)
  print "intermediary results (precision / recall / F1 Score):"
  print("%.6f %.6f %.6f" % (precision, recall, f1))

  # fig = plt.figure()
  fig, ax = plt.subplots()
  rect = fig.patch
  # rect.set_facecolor('white')
  coefs_to_display.plot(kind="bar")
  fig.tight_layout()
  # fig.subplots_adjust(bottom=0.5)
  plt.xticks(rotation=60)
  plt.show()


def computeOriginalData():
  runLogisticRegression(INPUT_CSV)



if __name__ == "__main__":
  computeOriginalData()
  # computeAllResults()
