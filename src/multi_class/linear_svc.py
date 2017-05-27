from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import input_preproc
import calculate_metrics

NAME = "Linear SVC"

def runClassifier(X_train, X_test, y_train, y_test):
    # print y_train

    predictions = OneVsRestClassifier(LinearSVC(), n_jobs=-1)\
        .fit(X_train, y_train).predict(X_test)


    # Metrics...
    precision, recall, f1, accuracy = calculate_metrics.calculateMetrics(predictions, y_test)
    print "intermediary results (precision | recall | F1 Score | Accuracy):"
    print("%.6f %.6f %.6f %.6f" % (precision, recall, f1, accuracy))
    return precision, recall, f1, accuracy


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = input_preproc.readIris()
    precision, recall, f1 = runClassifier(X_train, X_test, y_train, y_test)
    print "\n================================"
    print "Precision / Recall / F1 Score: "
    print("%.6f %.6f %.6f" % (precision, recall, f1))
