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
    ]
}

ALGORITHMS = [
    'linear_svc',
    'logistic_regression',
    'gradient_boosting',
    'random_forest'
]


def main_workflow():
    # CREATING FILELIST
    filelist = [f for f in sorted(os.listdir(INPUT_DIR)) if f.endswith(".csv")]

    for algo_str in ALGORITHMS:
        algorithm = importlib.import_module('src.multi_class.' + algo_str)

        with open(OUTPUT_DIR + 'results_' + algo_str + "_equal.csv", 'w') as equal_out, \
                open(OUTPUT_DIR + 'results_' + algo_str + "_bias.csv", 'w') as bias_out, \
                open(OUTPUT_DIR + 'results_' + algo_str + "_iml.csv", 'w') as iml_out:

            equal_writer = csv.writer(equal_out, lineterminator='\n')
            bias_writer = csv.writer(bias_out, lineterminator='\n')
            iml_writer = csv.writer(iml_out, lineterminator='\n')

            equal_writer.writerow(["target", "k-factor", "f1_score"])
            bias_writer.writerow(["target", "k-factor", "f1_score"])
            iml_writer.writerow(["target", "k-factor", "f1_score"])

            for input_file in filelist:
                # We only need the original result once, which we will hardcode somewhere..
                if input_file == ORIGINAL_DATA_FILE:
                    continue

                # Split filename to see what we're dealing with here...
                file_components = input_file.split('_')
                k_factor = file_components[len(file_components)-1].split('.')[0]
                weight_category = file_components[len(file_components)-2]
                target = file_components[len(file_components)-3]
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

                if weight_category == 'equal':
                    equal_writer.writerow([target, k_factor, final_f1])
                elif weight_category == 'bias':
                    bias_writer.writerow([target, k_factor, final_f1])
                elif weight_category == 'iml':
                    iml_writer.writerow([target, k_factor, final_f1])
                else:
                    raise Exception("Unknown weight category. Are you overweight???")

if __name__ == "__main__":
  main_workflow()