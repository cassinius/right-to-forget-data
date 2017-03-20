from sklearn import datasets
import sklearn.cross_validation as cross_validation
import pandas


def readIris():
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    # Return Train / Test split
    return cross_validation.train_test_split(X, y, train_size=0.80)


def readFromDataset():
    return None


