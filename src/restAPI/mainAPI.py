import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import sklearn.preprocessing as preprocessing
from flask import Flask

from src.multi_class import main_workflow
from src.multi_class import random_forest
from src.multi_class import input_preproc

config = main_workflow.CONFIG_INCOME
# input_file = "adults_original_dataset.csv"
input_file = "adults_anonymized_k07_equal.csv"

CROSS_VALIDATION_K = 10


app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"


@app.route("/testread")
def sendFileRead():
    encoded_data = input_preproc.readFromDataset(
        config['TARGET'] + input_file,
        config['INPUT_COLS'],
        config['TARGET_COL']
    )
    encoded_data = encoded_data.head(500)

    # Split into predictors and target
    X = np.array(encoded_data[encoded_data.columns.difference([config['TARGET_COL']])])
    y = np.array(encoded_data[config['TARGET_COL']])
    kf = KFold(n_splits=CROSS_VALIDATION_K, shuffle=True)

    # We want to know the variance of the training data set only
    print("INPUT DATA VARIANCE: %.2f" % (np.var(X)))

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

        precision, recall, f1_score = random_forest.runClassifier(X_train, X_test, y_train, y_test)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1_score)

    final_precision, final_recall, final_f1 = sum(precisions) / len(precisions), sum(recalls) / len(recalls), sum(
        f1s) / len(f1s)
    print "\n================================"
    print "Precision / Recall / F1 Score: "
    print("%.6f %.6f %.6f" % (final_precision, final_recall, final_f1))
    print "================================\n"

    results = {
        'precision': precision,
        'recall': recall,
        'f1': f1_score
    }

    return json.dumps(results)


if __name__ == "__main__":
    app.run()
