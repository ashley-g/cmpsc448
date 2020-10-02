############### test 5a ###############

import numpy as np
from Problem5 import bgd_l2, sgd_l2
import matplotlib.pyplot as plt

w = np.random.random(2)
data = np.load('data.npy')
x = np.hsplit(data, 2)
y = x[1]
x = x[0]

########### test 1 ################
new_w, history_fw = bgd_l2(x, y, w, .05, .1, .001, 50)
#print(history_fw)
plt.plot(history_fw)
plt.show()

########### test 2 ################
new_w, history_fw = bgd_l2(x, y, w, .1, .01, .001, 50)
plt.plot(history_fw)
plt.show()

########### test 3 ################
new_w, history_fw = bgd_l2(x, y, w, .1, 0, .001, 100)
plt.plot(history_fw)
plt.show()

########### test 4 ################
new_w, history_fw = bgd_l2(x, y, w, .1, 0, 0, 100)
plt.plot(history_fw)
plt.show()

#######################################

############### test 5b ###############

########### test 1 ################
new_w, history_fw = sgd_l2(x, y, w, 1, .1, .5, 800)
plt.plot(history_fw)
plt.show()

########### test 2 ################
new_w, history_fw = sgd_l2(x, y, w, 1, .01, .1, 800)
plt.plot(history_fw)
plt.show()

########### test 3 ################
new_w, history_fw = sgd_l2(x, y, w, 1, 0, 0, 40)
plt.plot(history_fw)
plt.show()

########### test 4 ################
new_w, history_fw = sgd_l2(x, y, w, 1, 0, 0, 800)
plt.plot(history_fw)
plt.show()

#######################################