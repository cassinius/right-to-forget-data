# import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import statsmodels as sm
import sklearn as skl
from sklearn import svm
from sklearn import ensemble
import sklearn.preprocessing as preprocessing
import sklearn.cross_validation as cross_validation
import sklearn.metrics as metrics
import sklearn.tree as tree
import seaborn as sns
import math
import os, csv, glob


INPUT_CSV = '../data/'
OUTPUT_CSV = '../output/'


def runSVMClassifier(input_file):

  original_data = pd.read_csv(
      input_file,
      names = [
          "nodeID", "age", "workclass", "native-country", "sex", "race", "marital-status", "relationship", "income"
      ],
      header=0,
      index_col=0,
      sep=r'\s*,\s*',
      engine='python',
      na_values="?")


  # Binary features
  binary_data = pd.get_dummies(original_data)
  # Let's fix the Target as it will be converted to dummy vars too
  binary_data["income"] = binary_data["income_>50K"]
  del binary_data["income_<=50K"]
  del binary_data["income_>50K"]


  # Use binary for SVM
  X_train, X_test, y_train, y_test = cross_validation.train_test_split(binary_data[binary_data.columns - ["income"]], binary_data["income"], train_size=0.80)
  scaler = preprocessing.StandardScaler()
  X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
  X_test = scaler.transform(X_test)

  # print X_train
  # print y_train

  # SVM CLASSIFIER
  # cls = svm.LinearSVC()
  # cls = svm.SVC(kernel="rbf", gamma=0.01, C=3, verbose=2)
  # cls.fit(X_train, y_train)
  # y_pred = cls.predict(X_test)

  # Gradient Boosting / Random forest classifier
  cls = ensemble.GradientBoostingClassifier(learning_rate=0.1, max_depth=5, verbose=0)
  # cls = ensemble.RandomForestClassifier(n_estimators=100, criterion="gini", max_features=None, verbose=0)
  cls.fit(X_train, y_train)
  y_pred = cls.predict(X_test)

  precision = skl.metrics.precision_score(y_test, y_pred)
  recall = skl.metrics.recall_score(y_test, y_pred)
  f1 = skl.metrics.f1_score(y_test, y_pred)

  return [precision, recall, f1]


# COMPILING RESULTS


def computeOriginalData():
  print runSVMClassifier(INPUT_CSV + "0_adults_sanitized_new.csv")

  # print runSVMClassifier(INPUT_CSV + "adults_relationship_Own-child_0.2.csv")
  # print runSVMClassifier(INPUT_CSV + "adults_relationship_Own-child_0.4.csv")
  # print runSVMClassifier(INPUT_CSV + "adults_relationship_Own-child_0.6.csv")
  # print runSVMClassifier(INPUT_CSV + "adults_relationship_Own-child_0.8.csv")
  # print runSVMClassifier(INPUT_CSV + "adults_relationship_Own-child_1.csv")

  # print runSVMClassifier(INPUT_CSV + "adults_marital-status_Married-civ-spouse_0.1.csv")
  # print runSVMClassifier(INPUT_CSV + "adults_marital-status_Married-civ-spouse_0.2.csv")
  # print runSVMClassifier(INPUT_CSV + "adults_marital-status_Married-civ-spouse_0.3.csv")
  # print runSVMClassifier(INPUT_CSV + "adults_marital-status_Married-civ-spouse_0.4.csv")
  # print runSVMClassifier(INPUT_CSV + "adults_marital-status_Married-civ-spouse_0.5.csv")

  # print runSVMClassifier(INPUT_CSV + "adults_marital-status_Never-married_0.1.csv")
  # print runSVMClassifier(INPUT_CSV + "adults_marital-status_Never-married_0.2.csv")
  # print runSVMClassifier(INPUT_CSV + "adults_marital-status_Never-married_0.3.csv")
  # print runSVMClassifier(INPUT_CSV + "adults_marital-status_Never-married_0.4.csv")
  # print runSVMClassifier(INPUT_CSV + "adults_marital-status_Never-married_0.5.csv")


def computeAllResults():
  filelist = [ f for f in sorted(os.listdir(INPUT_CSV)) if f.endswith(".csv") ]
  with open(OUTPUT_CSV + "results.csv", 'wb') as fout:
    writer = csv.writer(fout, lineterminator='\n')
    writer.writerow(["dataset", "precision", "recall", "F1 score"])

    for input_file in filelist:
      scores = runSVMClassifier(INPUT_CSV + input_file)
      scores.insert(0, input_file)
      writer.writerow(scores)
      print scores


if __name__ == "__main__":
  computeOriginalData()
  # computeAllResults()
