import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn import svm

def max_margin_classifier(ax, X, Y):
    clf = svm.SVC(kernel='linear')
    clf.fit(X, Y)

    #get the separating hyperplane
    w = clf.coef_[0]
    m = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = m * xx - (clf.intercept_[0] / w[1])

    #plot the parallel lines to the separating hyperplane that pass through the support vectors
    t = clf.support_vectors_[0]
    yy_down = m * xx + (t[1] - m * t[0])
    t = clf.support_vectors_[-1]
    yy_up = m * xx + (t[1] - m * t[0])

    ax.plot(xx, yy, 'k-')
    ax.plot(xx, yy_down, 'k--')
    ax.plot(xx, yy_up, 'k--')

    ax.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:, 1], s=80, facecolors='none')
    ax.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
    ax.axis('tight')

    return w

import matplotlib.pyplot as plt

np.random.seed()
#X = np.r_[np.random.randn(30,2) - [3,3], np.random.rand(30,2) + [3,3]]
#Y = [0] * 30 + [1] * 30

#X = [0,0], [0, -1], [-2,0]
X = np.array([[0,0], [0, -1], [-2,0]])
Y = [-1, -1, 1]

print(X)
print(Y)

fig = plt.figure()
ax = fig.add_subplot(111)
w = max_margin_classifier(ax, X, Y)
print(w)
plt.quiver(0, 0, 2*w[0], 2 * w[0]/w[1], angles='xy', scale_units='xy', scale=2)
plt.show()