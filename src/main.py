# import matplotlib
# matplotlib.use("TkAgg")

import pandas as pd
import numpy as np
import statsmodels as sm
import sklearn as skl
import sklearn.preprocessing as preprocessing
import sklearn.linear_model as linear_model
import sklearn.cross_validation as cross_validation
import sklearn.metrics as metrics
import sklearn.tree as tree
import seaborn as sns
import matplotlib.pyplot as plt
import math


INPUT_FILE = '../data/adults_sanitized.csv'
INPUT_ANON_K7_EQUAL = '../data/adults_anonymized_k7.csv'
INPUT_ANON_K7_RACE = '../data/adults_anonymized_k7_race_important.csv'
INPUT_ANON_K7_AGE = '../data/adults_anonymized_k7_age_important.csv'
INPUT_ANON_K13_EQUAL = '../data/adults_anonymized_k13.csv'
INPUT_ANON_K13_RACE = '../data/adults_anonymized_k13_race_important.csv'
INPUT_ANON_K13_AGE = '../data/adults_anonymized_k13_age_important.csv'
INPUT_ANON_K19_EQUAL = '../data/adults_anonymized_k19.csv'
INPUT_ANON_K19_RACE = '../data/adults_anonymized_k19_race_important.csv'
INPUT_ANON_K19_AGE = '../data/adults_anonymized_k19_age_important.csv'
INPUT_ANON_K31_EQUAL = '../data/adults_anonymized_k31.csv'
INPUT_ANON_K31_RACE = '../data/adults_anonymized_k31_race_important.csv'
INPUT_ANON_K31_AGE = '../data/adults_anonymized_k31_age_important.csv'


original_data = pd.read_csv(
    INPUT_ANON_K31_AGE,
    names = [
        "age", "workclass", "native-country", "sex", "race", "marital-status", "income" # "nodeID",
    ],
    header=0,
    # index_col=0,
    sep=r'\s*,\s*',
    engine='python',
    na_values="?")
print original_data.tail()


fig = plt.figure(figsize=(20,15))
cols = 3
rows = math.ceil(float(original_data.shape[1]) / cols)
for i, column in enumerate(original_data.columns):
    if i == len(original_data.columns) - 1:
        break
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    if original_data.dtypes[column] == np.object:
        original_data[column].value_counts().plot(kind="bar", axes=ax)
    else:
        original_data[column].hist(axes=ax)

    plt.xticks(rotation=55)

plt.subplots_adjust(top=0.9, bottom=0, hspace=0.4, wspace=0.2)
# plt.show()


print (original_data["native-country"].value_counts() / original_data.shape[0]).head()


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
# plt.show()


# ENCODE FEATURES AS NUMBERS
encoded_data, encoders = number_encode_features(original_data)
fig = plt.figure(figsize=(20,15))
cols = 3
rows = math.ceil(float(encoded_data.shape[1]) / cols)
for i, column in enumerate(encoded_data.columns):
    if i == len(encoded_data.columns) - 1:
        break
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    encoded_data[column].hist(axes=ax)
    plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.2, wspace=0.2)
# plt.show()


# TRAIN A LOGISTIC REGRESSION MODEL
X_train, X_test, y_train, y_test = cross_validation.train_test_split(encoded_data[encoded_data.columns - ["income"]], encoded_data["income"], train_size=0.80)
scaler = preprocessing.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train.astype("float64")), columns=X_train.columns)
X_test = scaler.transform(X_test.astype("float64"))


# LOGISTIC REGRESSION
cls = linear_model.LogisticRegression()

cls.fit(X_train, y_train)
y_pred = cls.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12,12))
plt.subplot(2,1,1)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoders["income"].classes_, yticklabels=encoders["income"].classes_)
plt.ylabel("Real value")
plt.xlabel("Predicted value")
print "F1 score: %f" % skl.metrics.f1_score(y_test, y_pred)
coefs = pd.Series(cls.coef_[0], index=X_train.columns)
coefs.sort()
plt.subplot(2,1,2)
coefs.plot(kind="bar")
# plt.show()


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
# plt.show()


# Use binary for Logistic regression
X_train, X_test, y_train, y_test = cross_validation.train_test_split(binary_data[binary_data.columns - ["income"]], binary_data["income"], train_size=0.80)
scaler = preprocessing.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = scaler.transform(X_test)


# Plot logistic regression again
cls = linear_model.LogisticRegression()
cls.fit(X_train, y_train)
y_pred = cls.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(20,20))
plt.subplot(2,1,1)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoders["income"].classes_, yticklabels=encoders["income"].classes_)
plt.ylabel("Real value")
plt.xlabel("Predicted value")
print "F1 score: %f" % skl.metrics.f1_score(y_test, y_pred)
coefs = pd.Series(cls.coef_[0], index=X_train.columns)
coefs.sort()
ax = plt.subplot(2,1,2)
coefs.plot(kind="bar")
plt.show()
