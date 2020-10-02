import math
import random
import numpy as np


def bgd_l2(data, y, w, eta, delta, lam, num_iter):
    ones = np.full((100, 1), 1)
    x = np.concatenate((ones, data), axis=1)
    new_w = w
    history_fw = []

    for i in range(num_iter):
        wt = np.transpose(new_w)
        
        #calulate gradient
        grad = 0
        for t in range(len(x)):
            if (y[t] >= (np.dot(wt, x[t]) + delta)):
                grad += -2*(y[t] - np.dot(wt, x[t]) - delta) * x[t]
            elif (abs(y[t] - np.dot(wt, x[t])) < delta):
                grad += 0
            elif (y[t] <= (np.dot(wt, x[t]) - delta)):
                grad += -2*(y[t] - np.dot(wt, x[t]) + delta) * x[t]
        grad = grad/len(x)
        grad += 2*lam*sum(wt)

        new_w = new_w - (eta * grad)
        wt = np.transpose(new_w)

        #calulate new f(w)
        fw = 0
        for t in range(len(x)):
            if (y[t] >= (np.dot(wt, x[t]) + delta)):
                fw += ((y[t] - np.dot(wt, x[t]) - delta) ** 2)
            elif (abs(y[t] - np.dot(wt, x[t])) < delta):
                fw += 0
            elif (y[t] <= (np.dot(wt, x[t]) - delta)):
                fw += ((y[t] - np.dot(wt, x[t]) + delta) ** 2)
        fw = fw/len(x)
        fw += lam*sum(wt**2)
        history_fw.append(fw)

    return new_w, history_fw


def sgd_l2(data, y, w, eta, delta, lam, num_iter, i=-1):

    ones = np.full((100, 1), 1)
    x = np.concatenate((ones, data), axis=1)
    new_w = w
    history_fw = []

    if (i != -1):
        numer_iter = 1
    else:
        i = random.randrange(0, len(x))

    for j in range(1, num_iter+1):
        wt = np.transpose(new_w)
        
        #calulate gradient
        grad = 0
        for t in range(len(x)):
            if (y[i] >= (np.dot(wt, x[i]) + delta)):
                grad += -2*(y[i] - np.dot(wt, x[i]) - delta) * x[i]
            elif (abs(y[t] - np.dot(wt, x[i])) < delta):
                grad += 0
            elif (y[i] <= (np.dot(wt, x[i]) - delta)):
                grad += -2*(y[i] - np.dot(wt, x[i]) + delta) * x[i]
        grad = grad/len(x)
        grad += 2*lam*sum(wt)

        new_w = new_w - ((eta / math.sqrt(j))  * grad)
        wt = np.transpose(new_w)

        #calulate new f(w)
        fw = 0
        for t in range(len(x)):
            if (y[t] >= (np.dot(wt, x[t]) + delta)):
                fw += ((y[t] - np.dot(wt, x[t]) - delta) ** 2)
            elif (abs(y[t] - np.dot(wt, x[t])) < delta):
                fw += 0
            elif (y[t] <= (np.dot(wt, x[t]) - delta)):
                fw += ((y[t] - np.dot(wt, x[t]) + delta) ** 2)
        fw = fw/len(x)
        fw += lam*sum(wt**2)
        history_fw.append(fw)

        i = random.randrange(0, len(x))

    return new_w, history_fw

