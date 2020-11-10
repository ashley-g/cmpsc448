import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import math

def k_init(X, k):
    """ k-means++: initialization algorithm

    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    k: int
        The number of clusters

    Returns
    -------
    init_centers: array (k, d)
        The initialize centers for kmeans++
    """
    centers = []
    ind = np.random.choice(len(X))
    centers.append(X.iloc[ind,:])
    for i in range(k-1):
        x_dist = []
        for x in range(len(X)):
            min_dist = np.inf
            for c in centers:
                #calculate distance from x to c using euclidean distance
                dist = 0
                for j in range(len(c)):
                    dist += (c[j] - X.iloc[x,j])**2
                min_dist = min(dist, min_dist)
            x_dist.append(dist)
        x_dist = x_dist / sum(x_dist) #create the probabilities proportional to d squared
        ind = np.random.choice(np.arange(0, len(X)), p=x_dist) #choose next centroid randomly with those probabilities
        centers.append(X.iloc[ind,:])
    return centers


def k_means_pp(X, k, max_iter):
    """ k-means++ clustering algorithm

    step 1: call k_init() to initialize the centers
    step 2: iteratively refine the assignments

    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    k: int
        The number of clusters

    max_iter: int
        Maximum number of iteration

    Returns
    -------
    final_centers: array, shape (k, d)
        The final cluster centers
    """
    centers = k_init(X, k)

    for i in range(max_iter):
        data_map = assign_data2clusters(X, centers)
        new_centers = []

        #calculate means for each cluster
        for c in centers:
            count = 0
            sum_x = 0
            sum_y = 0
            for item in data_map:
                if item[1][0] == c[0] and item[1][1] == c[1]:
                    sum_x += item[0][0]
                    sum_y += item[0][1]
                    count += 1
            new_centers.append([sum_x/count, sum_y/count])
        centers = new_centers

    return centers



def assign_data2clusters(X, C):
    """ Assignments of data to the clusters
    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    C: array, shape(k ,d)
        The final cluster centers

    Returns
    -------
    data_map: array, shape(n, k)
        The binary matrix A which shows the assignments of data points (X) to
        the input centers (C).
    """
    data_map = []
    for x in range(len(X)):
        cluster = None
        min_dist = np.inf
        for c in C:
            #calculate distance from x to c using euclidean distance
            dist = 0
            for j in range(len(c)):
                dist += (c[j] - X.iloc[x,j])**2
            if min_dist > dist: #keep track of smallest distance to center and which center it is
                min_dist = dist
                cluster = c
        data_map.append((X.iloc[x,:], cluster)) #assign data to closest centers cluster
    return data_map



def compute_objective(X, C):
    """ Compute the clustering objective for X and C
    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    C: array, shape(k ,d)
        The final cluster centers

    Returns
    -------
    accuracy: float
        The objective for the given assigments
    """
    data_map = assign_data2clusters(X, C)
    #accuracy = 0

    total_dist = 0
    for c in centers:
        for item in data_map:
            dist = 0
            if item[1][0] == c[0] and item[1][1] == c[1]:
                for j in range(len(c)):
                    dist += (c[j] - item[0][j])**2
        total_dist += dist

    #for x in range(len(X)):
    #    for c in C:
     #       #calculate the objective
     #       for j in range(len(c)):
     #           accuracy += (c[j] - X.iloc[x,j])**2
    #return accuracy
    return total_dist

#read in data from file and add column headers
data = pd.read_csv('iris.data')
data.columns = ['sepal_length','sepal_width','petal_length','petal_width','class']
print(data.head())

#create new data set with 2 features
#feature 1 = (sepal length/sepal width)
x1 = data['sepal_length'] / data['sepal_width']

#feature 2 = (petal length/petal width)
x2 = data['petal_length'] / data['petal_width']

y = data['class']
new_set = pd.DataFrame()
new_set['x1'] = x1
new_set['x2'] = x2
plt.scatter(x1, x2)
plt.show()
plt.clf()

#run with k=1,2,3,4,5
acc = []
centers = k_means_pp(new_set, 1, 50)
acc.append(compute_objective(new_set, centers))

centers = k_means_pp(new_set, 2, 50)
acc.append(compute_objective(new_set, centers))

centers = k_means_pp(new_set, 3, 50)
acc.append(compute_objective(new_set, centers))

centers = k_means_pp(new_set, 4, 50)
acc.append(compute_objective(new_set, centers))

centers = k_means_pp(new_set, 5, 50)
acc.append(compute_objective(new_set, centers))

plt.plot([1,2,3,4,5], acc)
plt.show()
plt.clf()

#my best was k=5 so now ill change the number fo iterations with k=5
acc = []
centers = k_means_pp(new_set, 5, 1)
acc.append(compute_objective(new_set, centers))

centers = k_means_pp(new_set, 5, 20)
acc.append(compute_objective(new_set, centers))

centers = k_means_pp(new_set, 5, 50)
acc.append(compute_objective(new_set, centers))

centers = k_means_pp(new_set, 5, 100)
acc.append(compute_objective(new_set, centers))

centers = k_means_pp(new_set, 5, 200)
acc.append(compute_objective(new_set, centers))

plt.plot([1,20,50,100,200], acc)
plt.show()
plt.clf()

#plot with data colored by cluster
centers = k_means_pp(new_set, 5, 200)
data_map = assign_data2clusters(new_set, centers)
label = []
for i in range(len(data_map)):
    if data_map[i][1] == centers[0]:
        label.append(0)
    elif data_map[i][1] == centers[1]:
        label.append(1)
    elif data_map[i][1] == centers[2]:
        label.append(2)
    elif data_map[i][1] == centers[3]:
        label.append(3)
    elif data_map[i][1] == centers[4]:
        label.append(4)
plt.scatter(x1, x2, c=label)
plt.show()
