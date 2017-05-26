from sklearn import datasets
import sklearn.cross_validation as cross_validation
import pandas as pd


def readIris():
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    # Return Train / Test split
    return cross_validation.train_test_split(X, y, train_size=0.80)


def returnOriginalDataset(input_file, input_cols, target_col):
    return pd.read_csv(
        input_file,
        names=input_cols,
        header=0,
        # index_col=0,
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")


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


    # Useful for random forest anonymization?
    # binary_data.drop_duplicates(inplace=True)

    if target_col == "marital-status":
      restoreMaritalStatus(original_data, binary_data)


    if target_col == "education-num":
      groupEducationLevels(binary_data)
      # groupEducationLevels(encoded_data)


    if target_col == "income":
      restoreIncome(original_data, binary_data)

    return binary_data


def restoreIncome(original_data, binary_data):
      del binary_data["income_>50K"]
      del binary_data["income_<=50K"]
      binary_data["income"] = original_data["income"]


'''
    For target education-num, group values into 4 baskets
    0 = Pre-School - 4th class
    1 = 5th - 12th class (including HS graduates)
    2 = Some college up to bachelors
    3 = Advanced studies
'''
def groupEducationLevels(data):
  for i, row in data.iterrows():
    if row["education-num"] < 3:
      data.loc[i, "education-num"] = 0
    elif row["education-num"] < 10:
      data.loc[i, "education-num"] = 1
    elif row["education-num"] < 14:
      data.loc[i, "education-num"] = 2
    else:
      data.loc[i, "education-num"] = 3


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
    data = readFromDataset('../../data/anonymization/adults_target_education_num/original/adults_original_dataset.csv',
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

    print data

    # print data["education-num"]