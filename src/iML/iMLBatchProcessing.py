import os, csv, glob
import pandas as pd
import numpy as np
import importlib
from src.multi_class import input_preproc
from sklearn.model_selection import KFold
import sklearn.preprocessing as preprocessing

INPUT_DIR = "../../iml_input/"
OUTPUT_DIR = "../../iml_output/"
ORIGINAL_DATA_FILE = "original_data_5000_rows.csv"

CROSS_VALIDATION_K = 10

TARGETS = ["education-num", "marital-status", "income"]

INPUT_COLS = {
    "education-num": [
        "age",
        "hours-per-week",
        "workclass",
        "native-country",
        "sex",
        "race",
        "relationship",
        "occupation",
        "marital-status",
        "income",
        "education-num"
    ],
    "marital-status": [
        "age",
        "hours-per-week",
        "education-num",
        "workclass",
        "native-country",
        "sex",
        "race",
        "relationship",
        "occupation",
        "income",
        "marital-status"
    ],
    "income": [
        "age",
        "hours-per-week",
        "education-num",
        "workclass",
        "native-country",
        "sex",
        "race",
        "relationship",
        "occupation",
        "marital-status",
        "income"
    ],
    "original": [
        "age",
        "education-num",
        "hours-per-week",
        "workclass",
        "native-country",
        "sex",
        "race",
        "relationship",
        "occupation",
        "income",
        "marital-status"
    ]
}

ALGORITHMS = [
    'linear_svc',
    'logistic_regression',
    'gradient_boosting',
    'random_forest'
]


def original_data():
    for target in TARGETS:
        for algo_str in ALGORITHMS:
            algorithm = importlib.import_module('src.multi_class.' + algo_str)
            encoded_data = input_preproc.readFromDataset(
                INPUT_DIR + ORIGINAL_DATA_FILE,
                INPUT_COLS['original'],
                target
            )
            # Split into predictors and target
            X = np.array(encoded_data[encoded_data.columns.difference([target])])
            y = np.array(encoded_data[target])
            kf = KFold(n_splits=CROSS_VALIDATION_K, shuffle=True)

            f1s = []

            for train_index, test_index in kf.split(X):
                X_train, y_train = X[train_index], y[train_index]
                X_test, y_test = X[test_index], y[test_index]

                scaler = preprocessing.StandardScaler()
                X_train = pd.DataFrame(scaler.fit_transform(X_train))  # , columns=X_train.columns)
                X_test = scaler.transform(X_test)

                precision, recall, f1_score, accuracy = algorithm.runClassifier(X_train, X_test, y_train, y_test)
                f1s.append(f1_score)

            final_f1 = sum(f1s) / len(f1s)
            print("\n================================")
            print("%s, %s, F1 Score: %.6f" % (target, algo_str, final_f1))
            print("================================\n")




def main_workflow(filelist, out_dir):
    for algo_str in ALGORITHMS:
        algorithm = importlib.import_module('src.multi_class.' + algo_str)

        with open(out_dir + 'results_' + algo_str + "_income.csv", 'w') as income_out, \
                open(out_dir + 'results_' + algo_str + "_marital.csv", 'w') as marital_out, \
                open(out_dir + 'results_' + algo_str + "_education.csv", 'w') as education_out:

            income_writer = csv.writer(income_out, lineterminator='\n')
            marital_writer = csv.writer(marital_out, lineterminator='\n')
            education_writer = csv.writer(education_out, lineterminator='\n')

            income_writer.writerow(["weight_category", "k-factor", "f1_score"])
            marital_writer.writerow(["weight_category", "k-factor", "f1_score"])
            education_writer.writerow(["weight_category", "k-factor", "f1_score"])

            for input_file in filelist:
                # We only need the original result once, which we will hardcode somewhere..
                if input_file == ORIGINAL_DATA_FILE:
                    continue

                # Split filename to see what we're dealing with here...
                file_components = input_file.split('_')
                k_factor = file_components[len(file_components)-1].split('.')[0]
                weight_category = file_components[len(file_components)-2]
                target = file_components[len(file_components)-3]

                print(algo_str)
                print(weight_category)
                print(target)
                print(k_factor)

                encoded_data = input_preproc.readFromDataset(
                    INPUT_DIR + input_file,
                    INPUT_COLS[target],
                    target
                )

                # Split into predictors and target
                X = np.array(encoded_data[encoded_data.columns.difference([target])])
                y = np.array(encoded_data[target])
                kf = KFold(n_splits=CROSS_VALIDATION_K, shuffle=True)

                f1s = []

                for train_index, test_index in kf.split(X):
                    X_train, y_train = X[train_index], y[train_index]
                    X_test, y_test = X[test_index], y[test_index]

                    scaler = preprocessing.StandardScaler()
                    X_train = pd.DataFrame(scaler.fit_transform(X_train))  # , columns=X_train.columns)
                    X_test = scaler.transform(X_test)

                    precision, recall, f1_score, accuracy = algorithm.runClassifier(X_train, X_test, y_train, y_test)
                    f1s.append(f1_score)

                final_f1 = sum(f1s) / len(f1s)
                print("\n================================")
                print("F1 Score: %.6f" % (final_f1))
                print("================================\n")

                if target == 'income':
                    income_writer.writerow([weight_category, k_factor, final_f1])
                elif target == 'marital-status':
                    marital_writer.writerow([weight_category, k_factor, final_f1])
                elif target == 'education-num':
                    education_writer.writerow([weight_category, k_factor, final_f1])
                else:
                    raise Exception("Unknown weight category. Are you overweight???")


if __name__ == "__main__":
    filelist_anon = [f for f in sorted(os.listdir(INPUT_DIR)) if f.endswith(".csv")]
    # main_workflow(filelist_anon, OUTPUT_DIR)
    original_data()
