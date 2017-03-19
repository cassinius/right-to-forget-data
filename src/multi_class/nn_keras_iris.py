import numpy
import pandas
import sklearn.metrics as skmetrics
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn import datasets


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# Load dataset via downloaded CSV
# dataframe = pandas.read_csv("./demo_data/iris.csv", header=None)
# dataset = dataframe.values
# X = dataset[:,0:4].astype(float)
# Y = dataset[:,4]

# Load dataset via scikit learn
iris = datasets.load_iris()
X = iris.data
Y = iris.target

# Debug Output Iris dataset
# print X
# print Y


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)


# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=4, kernel_initializer='normal', activation='relu'))
	model.add(Dense(3, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy", metrics.categorical_accuracy, metrics.MSE])
	return model


# Build the estimator
estimator = KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)


# K-fold cross validation
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)


# Result computation
# results = cross_val_score(estimator, X, dummy_y, cv=kfold)
# print results
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))



# Fitting without Keras
model = Sequential()
model.add(Dense(4, input_dim=4, kernel_initializer='normal', activation='relu'))
model.add(Dense(3, kernel_initializer='normal', activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy", metrics.MSE])
model.fit(X, dummy_y, epochs=100, batch_size=5, verbose=0)
scores = model.evaluate(X, dummy_y)
# print scores
# print model.metrics_names
# print model.metrics
# print("%s: %.2f%%" % (model.metrics[2], scores[1]*100))


# Predicting
predictions = model.predict_classes(X)
print predictions

precision = skmetrics.precision_score(predictions, Y, average="macro")
recall = skmetrics.recall_score(predictions, Y, average="macro")
f1 = skmetrics.f1_score(predictions, Y, average="macro")
# f1 = (2.0 * precision * recall) / (precision + recall)

print "Precision / Recall / F1 Score: "

print( "%.2f%% %.2f%% %.2f%%", precision , recall, f1 )
