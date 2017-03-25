from sklearn.multiclass import OneVsRestClassifier
from sklearn import ensemble
import input_preproc
import calculate_metrics


def runClassifier(X_train, X_test, y_train, y_test):
    # print y_train
    cls = ensemble.RandomForestClassifier(n_estimators=10,
                                          criterion="gini",
                                          max_features=None,
                                          verbose=0)

    predictions = cls.fit(X_train, y_train).predict(X_test)
    precision, recall, f1 = calculate_metrics.calculateMetrics(predictions, y_test)

    print "intermediary results (precision / recall / F1 Score):"
    print("%.6f %.6f %.6f" % (precision, recall, f1))

    return precision, recall, f1


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = input_preproc.readIris()
    precision, recall, f1 = runClassifier(X_train, X_test, y_train, y_test)
    print "\n================================"
    print "Precision / Recall / F1 Score: "
    print("%.6f %.6f %.6f" % (precision, recall, f1))
