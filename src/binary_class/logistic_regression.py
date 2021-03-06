# import matplotlib
# matplotlib.use("TkAgg")

import pandas as pd
import numpy as np
import statsmodels as sm
import sklearn as skl
from sklearn import svm
import sklearn.preprocessing as preprocessing
import sklearn.linear_model as linear_model
import sklearn.model_selection as ms
import sklearn.metrics as metrics
import sklearn.tree as tree
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import math
import os, csv, glob


INPUT_CSV = '../../data/anonymization/adults_target_income/original/'
OUTPUT_CSV = '../../output/anonymization/adults_target_income/original/'

input_cols = [
                "age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week", "workclass",
                "native-country", "sex", "race", "marital-status", "relationship", "occupation", "income"
             ]

def runLogisticRegression(input_file):

  original_data = pd.read_csv(
      input_file,
      names = input_cols,
      header=0,
      # index_col=0,
      sep=r'\s*,\s*',
      engine='python',
      na_values="?")



  # PLOT THE ORIGINAL VALUE DISTRIBUTION
  fig = plt.figure(figsize=(20,30))
  matplotlib.rcParams.update({'font.size': 32})
  rows = 4
  cols = 4
  for i, column in enumerate(input_cols): # enumerate(["capital-gain", "education-num", "marital-status", "relationship", "occupation", "hours-per-week"]):
      ax = fig.add_subplot(rows, cols, i + 1)
      ax.set_title(column, fontweight="bold", size=20)

      if original_data.dtypes[column] == np.object:
          original_data[column].value_counts().plot(kind="bar", axes=ax)
      else:
          original_data[column].hist(axes=ax)

      plt.xticks(rotation=45, ha="right", fontsize=14)

  plt.subplots_adjust(top=.95, bottom=.15, hspace=0.55, wspace=0.2)
  plt.show()



  # print (original_data["native-country"].value_counts() / original_data.shape[0]).head()


  # Encode the categorical features as numbers
  def number_encode_features(df):
      result = df.copy()
      encoders = {}
      for column in result.columns:
          if result.dtypes[column] == np.object:
              encoders[column] = preprocessing.LabelEncoder()
              result[column] = encoders[column].fit_transform(result[column])

      return result, encoders


  # PLOT CORRELATION OF ORIGINAL FEATURES
  plt.subplots(figsize=(10,10))
  # Calculate the correlation and plot it
  encoded_data, _ = number_encode_features(original_data)
  plt.xticks(rotation=90)
  plt.yticks(rotation=0)
  sns.heatmap(encoded_data.corr(), square=True)
  plt.show()


  # ENCODE FEATURES AS NUMBERS
  encoded_data, encoders = number_encode_features(original_data)
  fig = plt.figure(figsize=(20,15))
  rows = 4
  cols = 4
  # rows = math.ceil(float(encoded_data.shape[1]) / cols)
  for i, column in enumerate(input_cols): # enumerate(["capital-gain", "education-num", "marital-status", "relationship", "occupation", "hours-per-week"]):
      ax = fig.add_subplot(rows, cols, i + 1)
      ax.set_title(column)
      encoded_data[column].hist(axes=ax)
      plt.xticks(rotation=90)
  # plt.subplots_adjust(top=0.95, bottom=-0.25, hspace=0.2, wspace=0.5)
  plt.show()


  # DIVIDE THE DATASET INTO TRAIN AND TEST SETS
  X_train, X_test, y_train, y_test = ms.train_test_split(
      encoded_data[encoded_data.columns.difference(["income"])],
      encoded_data["income"], train_size=0.80)

  scaler = preprocessing.StandardScaler()
  X_train = pd.DataFrame(scaler.fit_transform(X_train.astype("float64")), columns=X_train.columns)
  X_test = scaler.transform(X_test.astype("float64"))


  # LOGISTIC REGRESSION
  cls = linear_model.LogisticRegression()
  cls.fit(X_train, y_train)
  y_pred = cls.predict(X_test)

  print( encoders )


  cm = metrics.confusion_matrix(y_test, y_pred)
  plt.figure(figsize=(12,12))
  plt.subplot(2,1,1)
  sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoders["income"].classes_, yticklabels=encoders["income"].classes_)
  plt.ylabel("Real value")
  plt.xlabel("Predicted value")

  # print "Precision: %f" % skl.metrics.precision_score(y_test, y_pred)
  # print "Recall: %f" % skl.metrics.recall_score(y_test, y_pred)
  # print "F1 score: %f" % skl.metrics.f1_score(y_test, y_pred)

  coefs = pd.Series(cls.coef_[0], index=X_train.columns)
  coefs.sort_values(inplace=True)
  plt.subplot(2,1,2)
  coefs.plot(kind="bar")
  plt.show()


  # Binary features
  binary_data = pd.get_dummies(original_data)
  # Let's fix the Target as it will be converted to dummy vars too
  binary_data["income"] = binary_data["income_>50K"]
  del binary_data["income_<=50K"]
  del binary_data["income_>50K"]
  plt.subplots(figsize=(20,20))
  sns.heatmap(binary_data.corr(), square=True)

  plt.xticks(rotation=90)
  plt.yticks(rotation=0)
  plt.show()


  # Use binary for Logistic regression
  X_train, X_test, y_train, y_test = ms.train_test_split(
      binary_data[binary_data.columns.difference(["income"])],
      binary_data["income"], train_size=0.80)


  # Scale attribute ranges
  scaler = preprocessing.StandardScaler()
  X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
  X_test = scaler.transform(X_test)


  # Re-run logistic regression again
  cls = linear_model.LogisticRegression()
  cls.fit(X_train, y_train)
  y_pred = cls.predict(X_test)


  cm = metrics.confusion_matrix(y_test, y_pred)
  plt.figure(figsize=(20,20))
  plt.subplot(2,1,1)
  sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoders["income"].classes_, yticklabels=encoders["income"].classes_)
  plt.ylabel("Real value")
  plt.xlabel("Predicted value")
  plt.show()


  coefs = pd.Series(cls.coef_[0], index=X_train.columns)
  coefs.sort_values(inplace=True)
  ax = plt.subplot(2,1,1)
  coefs.plot(kind="bar", rot=90)
  plt.show()


  precision = skl.metrics.precision_score(y_test, y_pred)
  # print "Precision: %f" % precision
  recall = skl.metrics.recall_score(y_test, y_pred)
  # print "Recall: %f" % recall
  f1 = skl.metrics.f1_score(y_test, y_pred)
  # print "F1 score: %f" % f1

  return [precision, recall, f1]


# COMPILING RESULTS
filelist = [ f for f in sorted(os.listdir(INPUT_CSV)) if f.endswith(".csv") ]


def computeOriginalData():
  # print runLogisticRegression(INPUT_CSV + "adults_original_dataset.csv")
  print( runLogisticRegression(INPUT_CSV + "adults_original_dataset.csv") )


def computeAllResults():
  with open(OUTPUT_CSV + "results.csv", 'w') as fout:
    writer = csv.writer(fout, lineterminator='\n')
    writer.writerow(["dataset", "precision", "recall", "F1 score"])

    for input_file in filelist:
      # print input_file
      scores = runLogisticRegression(INPUT_CSV + input_file)
      scores.insert(0, input_file)
      writer.writerow(scores)
      print( scores )


if __name__ == "__main__":
  computeOriginalData()
  # computeAllResults()
