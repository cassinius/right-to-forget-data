import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn import svm


digits = datasets.load_digits()

# Just to explore
# print(digits.data)
# print(digits.target)
# print(digits.images[0])

# converting everything to numbers

# converting everything to binary (or a -1..1 interval)


clf = svm.SVC(gamma=0.0001, C=100)

print "Size of data set: " + str(len(digits.data)) + " samples."

X,y = digits.data[:-10], digits.target[:-10]

# X = np.array(X)
# X = X.reshape(len(X), -1)
#
# print len(X)
# print X


clf.fit(X,y)

print 'Prediction: ' + str( clf.predict(digits.data[-6])[0] )

plt.imshow(digits.images[-6], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()

