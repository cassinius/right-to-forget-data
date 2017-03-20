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
import importlib
import logistic_regression
import linear_svc
import gradient_boosting
import random_forest
import nn_keras


CONFIG_EDUCATION = {
    'TARGET': "../../data/adults_target_education_num/",
    'OUTPUT': "../../output/adults_target_education_num/"
}

CONFIG_MARITAL = {
    'TARGET': "../../data/adults_target_marital_status/",
    'OUTPUT': "../../output/adults_target_marital_status/"
}

ALGORITHMS = [
    'linear_svc',
    'nn_keras',
    # 'logistic_regression',
    # 'gradient_boosting',
    # 'random_forest'
]

config = CONFIG_EDUCATION


def main_workflow():
    print "Starting main workflow..."
    print "Running on target: " + config['TARGET']

    # CREATING FILELIST
    filelist = [f for f in sorted(os.listdir(config['TARGET'])) if f.endswith(".csv")]

    for algo_str in ALGORITHMS:

        i = importlib.import_module(algo_str)

        with open(config['OUTPUT'] + 'results_' + algo_str + ".csv", 'wb') as fout:
            writer = csv.writer(fout, lineterminator='\n')
            writer.writerow(["dataset", "precision", "recall", "F1 score"])

            for input_file in filelist:

                print "Running algorithm: " + algo_str + "on: " + input_file

                precision, recall, f1_score = 0., 0., 0.

                print "\n================================"
                print "Precision / Recall / F1 Score: "
                print("%.6f %.6f %.6f" % (precision, recall, f1_score))
                print "================================\n"

                writer.writerow([input_file, precision, recall, f1_score])



if __name__ == "__main__":
  main_workflow()