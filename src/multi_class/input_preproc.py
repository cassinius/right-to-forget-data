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

    return cross_validation.train_test_split(
        encoded_data[encoded_data.columns.difference([target_col])],
        encoded_data[target_col], train_size=0.80)
