from sklearn.datasets import load_iris 
iris = load_iris()
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
X = iris.data
y = iris.target

def part1(X_):
    #PCA with k=1 component
    pca = PCA(n_components=1)
    pca.fit(X_)
    print("PCA with k=1")
    print("Explained Variance: {0}".format(pca.explained_variance_))
    print("Explained Variance Ratios: {0}".format(pca.explained_variance_ratio_))
    print("Total Explained Variance Ratio: {0}\n".format(sum(pca.explained_variance_ratio_)))
    #PCA with k=2 component
    pca = PCA(n_components=2)
    pca.fit(X_)
    print("PCA with k=2")
    print("Explained Variance: {0}".format(pca.explained_variance_))
    print("Explained Variance Ratios: {0}".format(pca.explained_variance_ratio_))
    print("Total Explained Variance Ratio: {0}\n".format(sum(pca.explained_variance_ratio_)))
    #PCA with k=3 component
    pca = PCA(n_components=3)
    pca.fit(X_)
    print("PCA with k=3")
    print("Explained Variance: {0}".format(pca.explained_variance_))
    print("Explained Variance Ratios: {0}".format(pca.explained_variance_ratio_))
    print("Total Explained Variance Ratio: {0}\n".format(sum(pca.explained_variance_ratio_)))
    #PCA with k=4 component
    pca = PCA(n_components=4)
    pca.fit(X_)
    print("PCA with k=4")
    print("Explained Variance: {0}".format(pca.explained_variance_))
    print("Explained Variance Ratios: {0}".format(pca.explained_variance_ratio_))
    print("Total Explained Variance Ratio: {0}\n".format(sum(pca.explained_variance_ratio_)))

#Problem 2 part 1
print('######### Problem 2.1 #########')
part1(X)

#Problem 2 Part 2
print('######### Problem 2.2 #########')
#Centered Data
centered_X = np.array(X)
for i in range(4):
    centered_X[:,i] = centered_X[:,i] - centered_X[:,i].mean()
print("Centered Data")
part1(centered_X)
#Scaled/Normalized Data
norm_X = np.array(X)
for i in range(4):
    norm_X[:,i] = (norm_X[:,i] - min(norm_X[:,i])) / (max(norm_X[:,i]) - min(norm_X[:,i]))
print("Scaled/Normalized Data")
part1(norm_X)
#Standardized Data
stand_X = np.array(X)
for i in range(4):
    stand_X[:,i] = (stand_X[:,i] - stand_X[:,i].mean()) / stand_X[:,i].std()
print("Standardized Data")
part1(stand_X)

#Problem 2 part 3
print('######### Problem 2.3 #########')
pca = PCA(n_components=2)
pca.fit(X)
new_X = pca.transform(X)
plt.scatter(new_X[:, 0], new_X[:, 1], c=y)
plt.show()
plt.clf()

#Problem 2 part 4
print('######### Problem 2.4 #########')
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=12).fit(new_X)
plt.scatter(new_X[:, 0], new_X[:, 1], c=kmeans.labels_)
plt.show()
plt.clf()