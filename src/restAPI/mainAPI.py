import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import sklearn.preprocessing as preprocessing
from flask import Flask, request, jsonify
from InvalidUsage import InvalidUsage
import importlib

from src.multi_class import main_workflow
from src.multi_class import random_forest
from src.multi_class import input_preproc
from flask_cors import CORS, cross_origin
from iml_config import INPUT_COLS, CROSS_VALIDATION_K, ALGORITHMS


app = Flask(__name__)
CORS(app)

@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route("/")
def hello():
    return "Hello World!"


@app.route("/anonML", methods=['POST'])
def sendResults():
    if request.method == 'POST':
        print "Cient POST..."
    else:
        raise InvalidUsage('This route can only be accessed via POST requests', status_code=500)

    # TODO set those to request values
    target_col = request.json.get('target')

    # print str( request.json.get('grouptoken') )
    # print str( request.json.get('csvdata') )

    encoded_data = input_preproc.readFromString(
        request.json.get('csvdata'),
        INPUT_COLS[target_col],
        target_col
    )

    results = {}

    for algo_str in ALGORITHMS:
        print "Running on algorithm: " + algo_str
        algorithm = importlib.import_module("src.multi_class." + algo_str)
        results[algo_str] = computeResultsFrom( encoded_data, algorithm, target_col )

    return json.dumps(results)



def computeResultsFrom( encoded_data, algorithm, target_col ):
    precisions = []
    recalls = []
    f1s = []
    accuracies = []

    # Split into predictors and target
    X = np.array(encoded_data[encoded_data.columns.difference([target_col])])
    y = np.array(encoded_data[target_col])
    kf = KFold(n_splits=CROSS_VALIDATION_K, shuffle=True)

    for train_index, test_index in kf.split(X):
        # print "train_index: " + str(train_index)
        # print "test_index: " + str(test_index)
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        scaler = preprocessing.StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train))  # , columns=X_train.columns)
        X_test = scaler.transform(X_test)

        precision, recall, f1_score, accuracy = random_forest.runClassifier(X_train, X_test, y_train, y_test)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1_score)
        accuracies.append(accuracy)

    final_precision = sum(precisions) / len(precisions)
    final_recall = sum(recalls) / len(recalls)
    final_f1 = sum(f1s) / len(f1s)
    final_accuracy = sum(accuracies) / len(accuracies)

    print "\n================================"
    print "Precision | Recall | F1 Score | Accuracy"
    print("%.6f %.6f %.6f %.6f" % (final_precision, final_recall, final_f1, final_accuracy))
    print "================================\n"

    algo_results = {
        'precision': final_precision,
        'recall': final_recall,
        'f1': final_f1,
        'accuracy': final_accuracy,
        'plot_url': "http://berndmalle.com/imlanon/groupfolder/user_target_results.jpg"
    }
    return algo_results




def plotAndWriteResultsToFS():
    print "plotting..."


if __name__ == "__main__":
    app.run()
