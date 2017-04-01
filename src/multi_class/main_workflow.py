'''
The whole workflow consists of the following steps:

1. Choose an appropriate target
    => e.g. "adults_target_education_num"
2. Choose an algorithm
    => e.g. gradient boosting, ANN with keras etc.
3. Open an appropriate output file
    => e.g. "results_random_forest_multiclass.csv
4. write header line to output file
    => e.g. "dataset,precision,recall,F1 score"
5. Read a series of datasets (anonymization, perturbation)
    => e.g. "adults_anonymized_k7_equal.csv"
    - for each input dataset:
        a. run the choosen algorithm returning [precision, recall, f1_score]
        b. append that line to the open output file
6. Close the output file


What is happening where?

-) Reading an input file => extra, imported module
-) Instantiating & running a chosen algorithm => extra, imported module (maybe 2 methods)
    => including specialized settings / k-fold setting, special metrics computation etc.
-) Overall workflow => here
    a. setting target
    b. setting algorithm
    c. handling output file
'''


import os, csv, glob
import pandas as pd
import numpy as np
import importlib
import input_preproc
import sklearn.cross_validation as cross_validation
from sklearn.model_selection import KFold
import sklearn.preprocessing as preprocessing


CROSS_VALIDATION_K = 10

# MODE = 'anonymization'
MODE = 'perturbation'

CONFIG_EDUCATION = {
    'TARGET': "../../data/" + MODE + "/adults_target_education_num/",
    'OUTPUT': "../../output/" + MODE + "/adults_target_education_num/",
    'INPUT_COLS': [
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
    'TARGET_COL': "education-num"
}


CONFIG_MARITAL = {
    'TARGET': "../../data/" + MODE + "/adults_target_marital_status/",
    'OUTPUT': "../../output/" + MODE + "/adults_target_marital_status/",
    'INPUT_COLS': [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "workclass",
        "native-country",
        "sex",
        "race",
        "relationship",
        "occupation",
        "income",
        "marital-status"
     ],
    'TARGET_COL': "marital-status"
}


ALGORITHMS = [
    # 'linear_svc',
    'logistic_regression',
    'gradient_boosting',
    'random_forest',
    # 'nn_keras', ## TOO SLOW...
    # 'bagging_svc' ## WAY TOO SLOW...
]

config = CONFIG_MARITAL


def main_workflow():
    print "Starting main workflow..."
    print "Running on target: " + config['TARGET']

    # CREATING FILELIST
    filelist = [f for f in sorted(os.listdir(config['TARGET'])) if f.endswith(".csv")]

    for algo_str in ALGORITHMS:

        algorithm = importlib.import_module(algo_str)

        with open(config['OUTPUT'] + 'results_' + algo_str + ".csv", 'wb') as fout:
            writer = csv.writer(fout, lineterminator='\n')
            writer.writerow(["dataset", "precision", "recall", "f1"])

            for input_file in filelist:

                encoded_data = input_preproc.readFromDataset(
                    config['TARGET'] + input_file,
                    config['INPUT_COLS'],
                    config['TARGET_COL']
                )
                # print y_test

                print "Running algorithm: " + algo_str + " on: " + input_file

                # Split into predictors and target
                X = np.array( encoded_data[encoded_data.columns.difference([config['TARGET_COL']])] )
                y = np.array( encoded_data[config['TARGET_COL']] )
                kf = KFold(n_splits=CROSS_VALIDATION_K, shuffle=True)

                precisions = []
                recalls = []
                f1s = []

                for train_index, test_index in kf.split(X):
                    # print "train_index: " + str(train_index)
                    # print "test_index: " + str(test_index)
                    X_train, y_train = X[train_index], y[train_index]
                    X_test, y_test = X[test_index], y[test_index]

                    scaler = preprocessing.StandardScaler()
                    X_train = pd.DataFrame(scaler.fit_transform(X_train))  # , columns=X_train.columns)
                    X_test = scaler.transform(X_test)

                    precision, recall, f1_score = algorithm.runClassifier(X_train, X_test, y_train, y_test)
                    precisions.append(precision)
                    recalls.append(recall)
                    f1s.append(f1_score)

                final_precision, final_recall, final_f1 = sum(precisions)/len(precisions), sum(recalls)/len(recalls), sum(f1s)/len(f1s)
                print "\n================================"
                print "Precision / Recall / F1 Score: "
                print("%.6f %.6f %.6f" % (final_precision, final_recall, final_f1))
                print "================================\n"

                writer.writerow([input_file, final_precision, final_recall, final_f1])



if __name__ == "__main__":
  main_workflow()

