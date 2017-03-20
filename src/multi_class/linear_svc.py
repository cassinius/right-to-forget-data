from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import sklearn.cross_validation as cross_validation
import sklearn.metrics as skmetrics

iris = datasets.load_iris()
X, y = iris.data, iris.target

# Train / Test split
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, train_size=0.80)

predictions = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train).predict(X_test)
print predictions

precision = skmetrics.precision_score(predictions, y_test, average="macro")
recall = skmetrics.recall_score(predictions, y_test, average="macro")
f1 = (2.0 * precision * recall) / (precision + recall)

print "\n================================"
print "Precision / Recall / F1 Score: "
print( "%.6f %.6f %.6f" % (precision , recall, f1) )