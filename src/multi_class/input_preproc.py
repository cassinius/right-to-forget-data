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
    encoded_data, encoders = number_encode_features(original_data)

    if target_col == "education-num":
        for i, row in encoded_data.iterrows():
            if row["education-num"] < 3:
                encoded_data.loc[i, "education-num"] = 0
            elif row["education-num"] < 10:
                encoded_data.loc[i, "education-num"] = 1
            elif row["education-num"] < 14:
                encoded_data.loc[i, "education-num"] = 2
            else:
                encoded_data.loc[i, "education-num"] = 3

    return encoded_data


if __name__ == "__main__":
    data = readFromDataset('../../data/adults_target_education_num/adults_anonymized_k3_equal.csv',
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

    for i, row in data.iterrows():
        if row["education-num"] < 3:
            data.loc[i, "education-num"] = 0
        elif row["education-num"] < 10:
            data.loc[i, "education-num"] = 1
        elif row["education-num"] < 14:
            data.loc[i, "education-num"] = 2
        else:
            data.loc[i, "education-num"] = 3
    print data["education-num"]