import input_preproc
import calculate_metrics
import sklearn.linear_model as linear_model


def runClassifier(X_train, X_test, y_train, y_test):

  # LOGISTIC REGRESSION Model
  cls = linear_model.LogisticRegression(
    class_weight="balanced", # default = None
    max_iter=1000, # default = 100
    solver="lbfgs", # default = liblinear (can only handle on-vs-rest)
    multi_class="ovr",
    n_jobs=-1
  )

  predictions = cls.fit(X_train, y_train).predict(X_test)

  # Metrics...
  precision, recall, f1, accuracy = calculate_metrics.calculateMetrics(predictions, y_test)
  print "intermediary results (precision | recall | F1 Score | Accuracy):"
  print("%.6f %.6f %.6f %.6f" % (precision, recall, f1, accuracy))
  return precision, recall, f1, accuracy


# def showMostCriticalColumns():
#   coefs = pd.Series(cls.coef_[0], index=X_train.columns)
#   coefs.sort_values(inplace=True)
#   ax = plt.subplot(2, 1, 1)
#   coefs.plot(kind="bar", rot=90)
#   plt.show()


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = input_preproc.readIris()
    precision, recall, f1, accuracy = runClassifier(X_train, X_test, y_train, y_test)
    print "\n================================"
    print "Precision | Recall | F1 Score | Accuracy: "
    print("%.6f %.6f %.6f %.6f" % (precision, recall, f1, accuracy))
