from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
import input_preproc
import calculate_metrics


def runClassifier(X_train, X_test, y_train, y_test):
    # print y_train

    # ENSEMBLE SVM
    n_estimators = 10
    cls = OneVsRestClassifier(
      BaggingClassifier(SVC(kernel='linear',
                            probability=True,
                            class_weight='balanced'
                            ),
                        max_samples=1.0 / n_estimators,
                        n_estimators=n_estimators
                        )
    )

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
