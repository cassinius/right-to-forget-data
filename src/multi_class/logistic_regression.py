from sklearn import ensemble
import input_preproc
import calculate_metrics
import sklearn.linear_model as linear_model


def runClassifier(X_train, X_test, y_train, y_test):
  # LOGISTIC REGRESSION
  cls = linear_model.LogisticRegression(
    class_weight="balanced", # default = None
    max_iter=1000, # default = 100
    solver="liblinear", # default = liblinear (can only handle on-vs-rest)
    multi_class="ovr",
    n_jobs=-1
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
