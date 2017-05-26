from sklearn import ensemble
import input_preproc
import calculate_metrics


def runClassifier(X_train, X_test, y_train, y_test):
  # GRADIENT BOOSTING
  cls = ensemble.GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    verbose=0
  )

  predictions = cls.fit(X_train, y_train).predict(X_test)

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
