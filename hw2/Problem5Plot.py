import random
import numpy as np
import matplotlib.pyplot as plt
from Problem5 import bgd_l2, sgd_l2

if __name__ == '__main__':
    # Put the code for the plots here, you can use different functions for each
    # part

    ############### test 5a ###############

    w = np.random.random(2)
    data = np.load('data.npy')
    x = np.hsplit(data, 2)
    y = x[1]
    x = x[0]

    ########### test 1 ################
    new_w, history_fw = bgd_l2(x, y, w, .05, .1, .001, 50)
    #print(history_fw)
    plt.plot(history_fw)
    plt.title("Hsitory of Objective Function with GD - test 1")
    plt.xlabel("Iterations")
    plt.ylabel("Objective Function")
    plt.show()

    ########### test 2 ################
    new_w, history_fw = bgd_l2(x, y, w, .1, .01, .001, 50)
    plt.plot(history_fw)
    plt.title("Hsitory of Objective Function with GD - test 2")
    plt.xlabel("Iterations")
    plt.ylabel("Objective Function")
    plt.show()

    ########### test 3 ################
    new_w, history_fw = bgd_l2(x, y, w, .1, 0, .001, 100)
    plt.plot(history_fw)
    plt.title("Hsitory of Objective Function with GD - test 3")
    plt.xlabel("Iterations")
    plt.ylabel("Objective Function")
    plt.show()

    ########### test 4 ################
    new_w, history_fw = bgd_l2(x, y, w, .1, 0, 0, 100)
    plt.plot(history_fw)
    plt.title("Hsitory of Objective Function with GD - test 4")
    plt.xlabel("Iterations")
    plt.ylabel("Objective Function")
    plt.show()

    #######################################

    ############### test 5b ###############

    ########### test 1 ################
    new_w, history_fw = sgd_l2(x, y, w, 1, .1, .5, 800)
    plt.plot(history_fw)
    plt.title("Hsitory of Objective Function with SGD - test 1")
    plt.xlabel("Iterations")
    plt.ylabel("Objective Function")
    plt.show()

    ########### test 2 ################
    new_w, history_fw = sgd_l2(x, y, w, 1, .01, .1, 800)
    plt.plot(history_fw)
    plt.title("Hsitory of Objective Function with SGD - test 2")
    plt.xlabel("Iterations")
    plt.ylabel("Objective Function")
    plt.show()

    ########### test 3 ################
    new_w, history_fw = sgd_l2(x, y, w, 1, 0, 0, 40)
    plt.plot(history_fw)
    plt.title("Hsitory of Objective Function with SGD - test 3")
    plt.xlabel("Iterations")
    plt.ylabel("Objective Function")
    plt.show()

    ########### test 4 ################
    new_w, history_fw = sgd_l2(x, y, w, 1, 0, 0, 800)
    plt.plot(history_fw)
    plt.title("Hsitory of Objective Function with SGD - test 4")
    plt.xlabel("Iterations")
    plt.ylabel("Objective Function")
    plt.show()

    #######################################
