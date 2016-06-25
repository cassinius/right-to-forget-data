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

# names = [
#             "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
#             "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
#             "Hours per week", "Country", "Target"],


original_data = pd.read_csv(
    INPUT_FILE,
    names = [
        "nodeID", "age", "workclass", "native-country", "sex", "race", "marital-status", "income"
    ],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
print original_data.tail()


fig = plt.figure(figsize=(20,15))
cols = 5
rows = math.ceil(float(original_data.shape[1]) / cols)
for i, column in enumerate(original_data.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    if original_data.dtypes[column] == np.object:
        original_data[column].value_counts().plot(kind="bar", axes=ax)
    else:
        original_data[column].hist(axes=ax)
        plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.7, wspace=0.2)