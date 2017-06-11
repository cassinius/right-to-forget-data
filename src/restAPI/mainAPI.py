import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import sklearn.preprocessing as preprocessing
from flask import Flask, request, jsonify
from src.restAPI.InvalidUsage import InvalidUsage
import importlib
import datetime
from src.restAPI import dbConnection

from src.multi_class import main_workflow
from src.multi_class import random_forest
from src.multi_class import input_preproc
from flask_cors import CORS, cross_origin
from src.restAPI.iml_config import INPUT_COLS, CROSS_VALIDATION_K, ALGORITHMS
from src.restAPI.plotIMLResults import plotAndWriteResultsToFS

DATE_FORMAT = '%Y%m%d%H%M%S'

app = Flask(__name__)
CORS(app)
# SERVER_URL = app.config['SERVER_NAME']
# print "Server URL: " + SERVER_URL

@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route("/")
def hello():
    return "Welcome to the iML Anonymization Machine Learning API!"


@app.route("/getDBResults")
def retrieveDBResults():
    return dbConnection.getResultsFromDB()


@app.route("/anonML", methods=['POST'])
def sendResults():
    if request.method == 'POST':
        print( "Cient POST..." )
    else:
        raise InvalidUsage('This route can only be accessed via POST requests', status_code=500)

    # Store raw request into the database incl. timestamp (so we can trace computational time afterwards)
    start_time = datetime.datetime.now().strftime(DATE_FORMAT)
    dbConnection.storeRawRequest(request.json, start_time)

    # print request.json.get('csv').get('bias')
    target_col = request.json.get('target')

    overall_results = {
        "target"        : target_col,
        "grouptoken"    : request.json.get('grouptoken'),
        "usertoken"     : request.json.get('usertoken'),
        "results"       : {}
    }

    for anon_type in ['bias', 'iml']:
        overall_results['results'][anon_type] = computeResultsForData(request.json.get('csv').get(anon_type), target_col)

    # After computation add another timestamp to the result object
    overall_results["timestamp"] = datetime.datetime.now().strftime(DATE_FORMAT)
    plotAndWriteResultsToFS(overall_results)

    # Now store all the information in the result table
    dbConnection.storeResult(request, overall_results)

    return json.dumps(overall_results)



def computeResultsForData(csv_string, target_col):
    encoded_data = input_preproc.readFromString(
        csv_string,
        INPUT_COLS[target_col],
        target_col
    )

    algo_results = {}

    for algo_str in ALGORITHMS:
        print( "Running on algorithm: " + algo_str )
        algorithm = importlib.import_module("src.multi_class." + algo_str)
        algo_results[algo_str] = computeResultsFromAlgo(encoded_data, algorithm, target_col)

    return algo_results



def computeResultsFromAlgo(encoded_data, algorithm, target_col):
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

    print( "\n================================" )
    print( "Precision | Recall | F1 Score | Accuracy" )
    print( "%.6f %.6f %.6f %.6f" % (final_precision, final_recall, final_f1, final_accuracy) )
    print( "================================\n" )

    algo_results = {
        'precision': final_precision,
        'recall': final_recall,
        'f1': final_f1,
        'accuracy': final_accuracy
    }
    return algo_results



if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )

'''
    On a python commandline, start with something like:
    PYTHONPATH="." IML_SERVER="berndmalle.com" python2 src/restAPI/mainAPI.py
    PYTHONPATH="." IML_SERVER="berndmalle.com" pm2 start src/restAPI/mainAPI.py --name iMLRestAPI
'''
