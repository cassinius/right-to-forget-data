print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from src.multi_class import input_preproc


INPUT_FILE = "../../../data/anonymization/adults_target_income/adults_original_dataset.csv"
INPUT_COLS = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "workclass",
        "native-country",
        "sex",
        "race",
        "marital-status",
        "relationship",
        "occupation",
        "income"
     ]
TARGET_COL = "income"

#TODO refactor to array [0.1, 0.2, 0.3, 0.4, 0.5] and adapt code below
CONTAMINATION_LEVEL = 0.2


rng = np.random.RandomState(42)

# # Generate train data
# X = 0.3 * rng.randn(100, 20)
# X_train = np.r_[X + 2, X - 2]
# # Generate some regular novel observations
# X = 0.3 * rng.randn(20, 20)
# X_test = np.r_[X + 2, X - 2]
# # Generate some abnormal novel observations
# X_outliers = rng.uniform(low=-4, high=4, size=(20, 20))
#
# # fit the model
# clf = IsolationForest(max_samples=100, random_state=rng)
# clf.fit(X_train)
# y_pred_train = clf.predict(X_train)
# y_pred_test = clf.predict(X_test)
# y_pred_outliers = clf.predict(X_outliers)

# print y_pred_train
# print y_pred_test
# print y_pred_outliers


clf = IsolationForest(n_estimators=100,
                      max_samples='auto',
                      contamination=0.05,
                      # max_features=1,
                      bootstrap=False,
                      n_jobs=-1,
                      random_state=rng)
encoded_data = input_preproc.readFromDataset(
    INPUT_FILE,
    INPUT_COLS,
    TARGET_COL
)

# Split into predictors and target
X_adults = np.array( encoded_data[encoded_data.columns.difference([TARGET_COL])] )
y_adults = np.array( encoded_data[TARGET_COL] )

clf.fit(X_adults)
y_pred_adults = clf.predict(X_adults)

# print type(y_pred_adults)
# print len(y_pred_adults)
# print y_pred_adults

print "\n========================\n"

indices = np.where(y_pred_adults == -1)[0]
print("%d Outliers found, %s" %(len(indices), "at indices:"))
print indices

print "\n========================\n"


# plot the line, the samples, and the nearest vectors to the plane
# xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
# Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
#
# plt.title("IsolationForest")
# plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
#
# b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white')
# b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green')
# c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red')
# plt.axis('tight')
# plt.xlim((-5, 5))
# plt.ylim((-5, 5))
# plt.legend([b1, b2, c],
#            ["training observations",
#             "new regular observations", "new abnormal observations"],
#            loc="upper left")
# plt.show()