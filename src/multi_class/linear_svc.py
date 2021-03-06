from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from src.multi_class import input_preproc
from src.multi_class import calculate_metrics


def runClassifier(X_train, X_test, y_train, y_test):
    # print y_train

    predictions = OneVsRestClassifier(LinearSVC(), n_jobs=-1)\
        .fit(X_train, y_train).predict(X_test)


    # Metrics...
    precision, recall, f1, accuracy = calculate_metrics.calculateMetrics(predictions, y_test)
    print( "intermediary results (precision | recall | F1 Score | Accuracy):" )
    print( "%.6f %.6f %.6f %.6f" % (precision, recall, f1, accuracy) )
    return precision, recall, f1, accuracy


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = input_preproc.readIris()
    precision, recall, f1, accuracy = runClassifier(X_train, X_test, y_train, y_test)
    print( "\n================================" )
    print( "Precision | Recall | F1 Score | Accuracy: " )
    print( "%.6f %.6f %.6f %.6f" % (precision, recall, f1, accuracy) )
