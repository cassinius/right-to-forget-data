from sklearn import datasets
import sklearn.cross_validation as cross_validation
import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing


def readIris():
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    # Return Train / Test split
    return cross_validation.train_test_split(X, y, train_size=0.80)


def readFromDataset(input_file, input_cols, target_col):
    original_data = pd.read_csv(
        input_file,
        names=input_cols,
        header=0,
        # index_col=0,
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")

    binary_data = pd.get_dummies(original_data)

    if target_col == "marital-status":
      restoreMaritalStatus(original_data, binary_data)

    # print binary_data

    # Encode the categorical features as numbers
    def number_encode_features(df):
        result = df.copy()
        encoders = {}
        for column in result.columns:
            if result.dtypes[column] == np.object:
                encoders[column] = preprocessing.LabelEncoder()
                result[column] = encoders[column].fit_transform(result[column])
        return result, encoders


    # ENCODE FEATURES AS NUMBERS
    encoded_data, encoders = number_encode_features(binary_data)

    # print encoded_data
    # print encoders


    if target_col == "education-num":
      groupEducationLevels(binary_data)
      # groupEducationLevels(encoded_data)


    # print binary_data
    # print encoded_data

    return binary_data


'''
    For target education-num, group values into 4 baskets
    0 = Pre-School - 4th class
    1 = 5th - 12th class (including HS graduates)
    2 = Some college up to bachelors
    3 = Advanced studies
'''
def groupEducationLevels(encoded_data):
  for i, row in encoded_data.iterrows():
    if row["education-num"] < 3:
      encoded_data.loc[i, "education-num"] = 0
    elif row["education-num"] < 10:
      encoded_data.loc[i, "education-num"] = 1
    elif row["education-num"] < 14:
      encoded_data.loc[i, "education-num"] = 2
    else:
      encoded_data.loc[i, "education-num"] = 3


def restoreMaritalStatus(original_data, binary_data):
  del binary_data["marital-status_Divorced"]
  del binary_data["marital-status_Married-AF-spouse"]
  del binary_data["marital-status_Married-civ-spouse"]
  del binary_data["marital-status_Married-spouse-absent"]
  del binary_data["marital-status_Never-married"]
  del binary_data["marital-status_Separated"]
  del binary_data["marital-status_Widowed"]
  binary_data["marital-status"] = original_data["marital-status"]


if __name__ == "__main__":
    data = readFromDataset('../../data/adults_target_education_num/adults_original_dataset.csv',
             [
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
             "education-num"
    )

    # print data

    # print data["education-num"]