import pandas as pd
import numpy as np
import sklearn as skl
from sklearn import svm
from sklearn import ensemble
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as ms
import sklearn.metrics as metrics
import sklearn.tree as tree
import math
import os, csv, glob
import sklearn.linear_model as linear_model
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.svm import SVC


INPUT_CSV_INCOME = '../../data/anonymization/adults_target_income/original/'
OUTPUT_CSV = '../../output/anonymization/adults_target_income/original/'


def runClassifier(input_file):

  original_data = pd.read_csv(
      input_file,
      names=[
        "age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week", "workclass",
        "native-country", "sex", "race", "marital-status", "relationship", "occupation", "income"
      ],
      header=0,
      # index_col=0,
      sep=r'\s*,\s*',
      engine='python',
      na_values="?")


  # Binary features
  binary_data = pd.get_dummies(original_data)


  # Let's fix the Target as it will be converted to dummy vars too
  binary_data["income"] = binary_data["income_>50K"]
  del binary_data["income_<=50K"]
  del binary_data["income_>50K"]


  # Split training and test sets
  X_train, X_test, y_train, y_test = ms.train_test_split(binary_data[binary_data.columns.difference(["income"])],
                                                                       binary_data["income"], train_size=0.80)
  scaler = preprocessing.StandardScaler()
  X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
  X_test = scaler.transform(X_test)


  # LOGISTIC REGRESSION
  # cls = linear_model.LogisticRegression()

  # LINEAR SVC
  # cls = svm.LinearSVC()

  # SVC
  # Too bad results
  # cls = svm.SVC(kernel="rbf", verbose=2)


  # ENSEMBLE SVM
  # n_estimators = 10
  # cls = OneVsRestClassifier(
  #   BaggingClassifier(SVC(kernel='linear', probability=True, class_weight='balanced'), max_samples=1.0 / n_estimators,
  #                     n_estimators=n_estimators))


  # GRADIENT BOOSTING
  # cls = ensemble.GradientBoostingClassifier(learning_rate=0.1, max_depth=5, verbose=0)

  # RANDOM FOREST
  cls = ensemble.RandomForestClassifier(n_estimators=100, criterion="gini", max_features=None, verbose=0, n_jobs=-1)

  # Run the actual training and prediction phases
  cls.fit(X_train, y_train)
  y_pred = cls.predict(X_test)

  precision = skl.metrics.precision_score(y_test, y_pred)
  recall = skl.metrics.recall_score(y_test, y_pred)
  f1 = skl.metrics.f1_score(y_test, y_pred)
  accuracy = metrics.accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)

  return [precision, recall, f1, accuracy]


# COMPILING RESULTS


def computeOriginalData():
  print runClassifier( INPUT_CSV_INCOME + 'adults_original_dataset.csv' )


def computeAllResults():
  filelist = [f for f in sorted(os.listdir(INPUT_CSV_INCOME)) if f.endswith(".csv")]
  with open(OUTPUT_CSV + "results_random_forest.csv", 'wb') as fout:
    writer = csv.writer(fout, lineterminator='\n')
    writer.writerow(["dataset", "precision", "recall", "F1 score", "accuracy"])

    final_results = {}
    intermediary_results = {}
    for input_file in filelist:
      intermediary_results[input_file] = {}
      intermediary_results[input_file]["precision"] = []
      intermediary_results[input_file]["recall"] = []
      intermediary_results[input_file]["f1"] = []
      intermediary_results[input_file]["accuracy"] = []
      for i in range(0,10):
        scores = runClassifier(INPUT_CSV_INCOME + input_file)
        intermediary_results[input_file]["precision"].append(scores[0])
        intermediary_results[input_file]["recall"].append(scores[1])
        intermediary_results[input_file]["f1"].append(scores[2])
        intermediary_results[input_file]["accuracy"].append(scores[3])
        # writer.writerow( [input_file,
        #                   i,
        #                   intermediary_results[input_file]["precision"],
        #                   intermediary_results[input_file]["recall"],
        #                   intermediary_results[input_file]["f1"],
        #                   intermediary_results[input_file]["accuracy"]] )

    for input_file in filelist:
      final_results[input_file] = {}
      final_results[input_file]["precision"] = np.mean(intermediary_results[input_file]["precision"])
      final_results[input_file]["recall"] = np.mean(intermediary_results[input_file]["recall"])
      final_results[input_file]["f1"] = np.mean(intermediary_results[input_file]["f1"])
      final_results[input_file]["accuracy"] = np.mean(intermediary_results[input_file]["accuracy"])
      writer.writerow( [input_file,
                        final_results[input_file]["precision"],
                        final_results[input_file]["recall"],
                        final_results[input_file]["f1"],
                        final_results[input_file]["accuracy"]])
    print final_results


if __name__ == "__main__":
  # computeOriginalData()
  computeAllResults()
