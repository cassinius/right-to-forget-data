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
from src.multi_class import input_preproc
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import sklearn.preprocessing as preprocessing

CROSS_VALIDATION_K = 10

MODE = 'anonymization'
# MODE = 'perturbation'
# MODE = 'outliers'

# OUTLIER_TARGET = ''
# OUTLIER_TARGET = 'outliers/'
# OUTLIER_TARGET = 'random_comparison/'
OUTLIER_TARGET = 'original/'
# OUTLIER_TARGET = 'outliers_removed/'


CONFIG_EDUCATION = {
    'TARGET': "../../data/" + MODE + "/adults_target_education_num/" + OUTLIER_TARGET,
    'OUTPUT': "../../output/" + MODE + "/adults_target_education_num/" + OUTLIER_TARGET,
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
    'TARGET': "../../data/" + MODE + "/adults_target_marital_status/" + OUTLIER_TARGET,
    'OUTPUT': "../../output/" + MODE + "/adults_target_marital_status/" + OUTLIER_TARGET,
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


CONFIG_INCOME = {
    'TARGET': "../../data/" + MODE + "/adults_target_income/" + OUTLIER_TARGET,
    'OUTPUT': "../../output/" + MODE + "/adults_target_income/" + OUTLIER_TARGET,
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
        "marital-status",
        "relationship",
        "occupation",
        "income"
     ],
    'TARGET_COL': "income"
}


ALGORITHMS = [
    # 'linear_svc',
    'logistic_regression',
    # 'gradient_boosting',
    # 'random_forest',
    # 'nn_keras', ## TOO SLOW...
    # 'bagging_svc' ## WAY TOO SLOW...
]

config = CONFIG_INCOME


def main_workflow():
    print( "Starting main workflow..." )
    print( "Running on target: " + config['TARGET'] )

    # CREATING FILELIST
    filelist = [f for f in sorted(os.listdir(config['TARGET'])) if f.endswith(".csv")]

    for algo_str in ALGORITHMS:

        algorithm = importlib.import_module(algo_str)

        with open(config['OUTPUT'] + 'results_' + algo_str + ".csv", 'w') as algo_out, \
             open(config['OUTPUT'] + 'data_stats' + ".csv", 'w') as stats_out:
            algo_writer = csv.writer(algo_out, lineterminator='\n')
            algo_writer.writerow(["dataset", "precision", "recall", "f1", "std", "var"])

            stats_writer = csv.writer(stats_out, lineterminator='\n')
            stats_writer.writerow(["dataset", "size", "std", "var"])

            for input_file in filelist:

                encoded_data = input_preproc.readFromDataset(
                    config['TARGET'] + input_file,
                    config['INPUT_COLS'],
                    config['TARGET_COL']
                )

                print( "Running algorithm: " + algo_str + " on: " + input_file )

                # Split into predictors and target
                X = np.array( encoded_data[encoded_data.columns.difference([config['TARGET_COL']])] )
                y = np.array( encoded_data[config['TARGET_COL']] )
                kf = KFold(n_splits=CROSS_VALIDATION_K, shuffle=True)
                # kf = StratifiedKFold(n_splits=CROSS_VALIDATION_K)

                # We want to know the variance of the training data set only
                print("Input data SIZE: %.2f" % (len(X)))
                print("Input data STD.DEV.: %.2f" % (np.std(X)))
                print("Input data VARIANCE: %.2f" % (np.var(X)))
                stats_writer.writerow([input_file, len(X), np.std(X), np.var(X)])

                precisions = []
                recalls = []
                f1s = []
                accuracies = []

                for train_index, test_index in kf.split(X, y):
                    X_train, y_train = X[train_index], y[train_index]
                    X_test, y_test = X[test_index], y[test_index]

                    scaler = preprocessing.StandardScaler()
                    X_train = pd.DataFrame(scaler.fit_transform(X_train))  # , columns=X_train.columns)
                    X_test = scaler.transform(X_test)

                    precision, recall, f1_score, accuracy = algorithm.runClassifier(X_train, X_test, y_train, y_test)
                    precisions.append(precision)
                    recalls.append(recall)
                    f1s.append(f1_score)
                    accuracies.append(accuracy)

                final_precision, final_recall, final_f1, final_acc = sum(precisions)/len(precisions), sum(recalls)/len(recalls), sum(f1s)/len(f1s), sum(accuracies)/len(accuracies)
                print("\n================================")
                print("Precision | Recall | F1 Score | Accuracy: ")
                print("%.6f %.6f %.6f %.6f" % (final_precision, final_recall, final_f1, final_acc))
                print( "================================\n" )

                algo_writer.writerow([input_file, final_precision, final_recall, final_f1])



if __name__ == "__main__":
  main_workflow()

