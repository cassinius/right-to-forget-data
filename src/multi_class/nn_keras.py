import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import input_preproc
import calculate_metrics



def runClassifier(X_train, X_test, y_train, y_test):
    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)

    print "Input Shape: " + str( X_train.shape )
    print "Output Shape: " + str( dummy_y.shape )
    input_dim = X_train.shape[1]
    output_dim = dummy_y.shape[1]

    # Building model
    model = Sequential()
    model.add(Dense(input_dim, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(output_dim, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy", metrics.MSE])
    model.fit(X_train, dummy_y, epochs=100, batch_size=5, verbose=0)
    # scores = model.evaluate(X_train, dummy_y)
    # print scores

    # Predicting
    predictions = model.predict_classes(X_test)
    precision, recall, f1 = calculate_metrics.calculateMetrics(predictions, y_test)
    return precision, recall, f1



if __name__ == "__main__":
    X_train, X_test, y_train, y_test = input_preproc.readIris()
    precision, recall, f1 = runClassifier(X_train, X_test, y_train, y_test)
    print "\n================================"
    print "Precision / Recall / F1 Score: "
    print("%.6f %.6f %.6f" % (precision, recall, f1))