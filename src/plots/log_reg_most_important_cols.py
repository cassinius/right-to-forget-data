# import matplotlib
# matplotlib.use("TkAgg")

import pandas as pd
import sklearn.linear_model as linear_model
import sklearn.cross_validation as cross_validation
import src.multi_class.input_preproc
import matplotlib.pyplot as plt
import src.multi_class.calculate_metrics
import sklearn.preprocessing as preprocessing
import numpy as np
from sklearn.model_selection import KFold

CROSS_VALIDATION_K = 10

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

  encoded_data = src.multi_class.input_preproc.readFromDataset(
    input_file,
    config['INPUT_COLS'],
    config['TARGET_COL']
  )

  # LOGISTIC REGRESSION Model
  cls = linear_model.LogisticRegression(
    class_weight="balanced",  # default = None
    max_iter=1000,  # default = 100
    solver="liblinear",  # default = liblinear (can only handle one-vs-rest)
    multi_class="ovr",
    n_jobs=-1
  )

  # Split into predictors and target
  X = np.array(encoded_data[encoded_data.columns.difference([config['TARGET_COL']])])
  y = np.array(encoded_data[config['TARGET_COL']])
  kf = KFold(n_splits=CROSS_VALIDATION_K, shuffle=True)

  training_columns = encoded_data.columns.difference([config['TARGET_COL']])

  X = np.array(encoded_data[encoded_data.columns.difference([config['TARGET_COL']])])
  y = np.array(encoded_data[config['TARGET_COL']])
  kf = KFold(n_splits=CROSS_VALIDATION_K, shuffle=True)

  cls_coefs_sum = np.zeros(len(encoded_data.columns)-1)
  # print len(cls_coefs_sum)
  idx = 1

  for train_index, test_index in kf.split(X):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]

    scaler = preprocessing.StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train))  # , columns=X_train.columns)
    X_test = scaler.transform(X_test)

    # Calculate coefficients...
    cls.fit(X_train, y_train)

    # Sum up the coefs
    # print len(cls.coef_[0])
    cls_coefs_sum += cls.coef_[0]
    print("Finished iteration: %d" %(idx))
    idx += 1


  coefs = pd.Series(cls_coefs_sum / CROSS_VALIDATION_K, index=training_columns)
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

  # precision, recall, f1 = src.multi_class.calculate_metrics.calculateMetrics(predictions, y_test)
  # print "intermediary results (precision / recall / F1 Score):"
  # print("%.6f %.6f %.6f" % (precision, recall, f1))

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
