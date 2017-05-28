import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import sklearn.preprocessing as preprocessing
from InvalidUsage import InvalidUsage
import importlib
import datetime
import threading

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, send, emit

from src.multi_class import random_forest
from src.multi_class import input_preproc
from flask_cors import CORS, cross_origin

from iml_config import INPUT_COLS, CROSS_VALIDATION_K, ALGORITHMS
from plotIMLResults import plotAndWriteResultsToFS

DATE_FORMAT = '%Y%m%d%H%M%S'

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretsocketpassword'
cors = CORS(app) # ,resources={r"/*":{"origins":"*"}}

from gevent import monkey
monkey.patch_all()

# socketio = SocketIO(app, async_mode='threading')
# socketio = SocketIO(app, async_mode='eventlet')
socketio = SocketIO(app, async_mode='gevent')
# socketio = SocketIO(app)

# Right now we're setting AMOUNTS_RESULT to a fixed 8 (bias / iml x 4 classifiers
AMOUNT_RESULTS = 8

# SERVER_URL = app.config['SERVER_NAME']
# print "Server URL: " + SERVER_URL


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route("/")
def hello():
    return "Hello World!"


@socketio.on("computeMLResults")
def runClassifiersAndSendResults(request_data):

    # ONLY FOR TRADITIONAL FLASK (NOT SOCKET IO)
    # if request.method == 'POST':
    #     print "Cient POST..."
    # else:
    #     raise InvalidUsage('This route can only be accessed via POST requests', status_code=500)

    data = request_data['request']

    target_col = data['target']

    overall_results = {
        "timestamp"     : datetime.datetime.now().strftime(DATE_FORMAT),
        "target"        : target_col,
        "grouptoken"    : data['grouptoken'],
        "usertoken"     : data['usertoken'],
        "results"       : {}
    }

    # Inform client via Sockets about starting the compute cycle & how many intermediary results to expect
    nr_intermediary_results = CROSS_VALIDATION_K * AMOUNT_RESULTS
    emitViaSocket('computationStarted', {'nr_intermediary_results': nr_intermediary_results})

    for anon_type in ['bias', 'iml']:
        overall_results['results'][anon_type] = computeResultsForData(data['csv'][anon_type], target_col)

    plotAndWriteResultsToFS(overall_results)

    emitViaSocket('computationCompleted', {'overall_results': overall_results})
    # return json.dumps(overall_results)



def computeResultsForData(csv_string, target_col):
    encoded_data = input_preproc.readFromString(
        csv_string,
        INPUT_COLS[target_col],
        target_col
    )

    algo_results = {}

    for algo_str in ALGORITHMS:
        print "Running on algorithm: " + algo_str
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

        # Inform client via Sockets that an intermediary result is ready
        intermediary_results = {
            'algorithm': algorithm.NAME,
            'precision': precision,
            'recall': recall,
            'f1': f1_score,
            'accuracy': accuracy
        }
        emitViaSocket('intermediaryComputed', {'result': intermediary_results})

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
        'accuracy': final_accuracy
    }

    return algo_results



def emitViaSocket(event, json):
    print "Socket send event"
    emit(event, json, json=True)


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=5000)
    socketio.run(app, port=5000, host='0.0.0.0')

'''
    On a python commandline, start with something like:
    PYTHONPATH="." IML_SERVER="berndmalle.com" python src/restAPI/mainAPI.py
    PYTHONPATH="." IML_SERVER="berndmalle.com" pm2 start src/restAPI/mainAPI.py --name 'iML REST API'
'''
