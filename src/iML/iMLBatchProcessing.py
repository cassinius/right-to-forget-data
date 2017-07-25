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

CONFIG_EDUCATION = {
    'INPUT_COLS': [
        "age",
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
    'INPUT_COLS': [
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
     ],
    'TARGET_COL': "marital-status"
}

CONFIG_INCOME = {
    'INPUT_COLS': [
        "age",
        "education-num",
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

            equal_writer.writerow(["k-factor", "f1_score"])
            bias_writer.writerow(["k-factor", "f1_score"])
            iml_writer.writerow(["k-factor", "f1_score"])

            for input_file in filelist:
                # Split filename to see what we're dealing with here...
                if ( input_file == ORIGINAL_DATA_FILE ):
                    continue
                print(input_file)











if __name__ == "__main__":
  main_workflow()