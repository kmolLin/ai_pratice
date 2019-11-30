import numpy as np
import matplotlib.pyplot as plt

import random


x_init = np.array([[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]])
x_label = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
y_inmit = np.array([24, 42, 63, 87, 101, 126, 135, 158, 183, 205])
b_paremeter = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])


def least_square():
    A_matrix = np.c_[x_init.T, b_paremeter.T]
    B = np.linalg.inv(A_matrix.T @ A_matrix) @ A_matrix.T @ y_inmit.T
    return B


def recusive_least_square():
    theta_hat_save = []
    theta_hat = np.array([0, 0])
    delta = 1
    p = delta * np.eye(2)
    for i in range(10):
        phi = np.array([[x_label[i]], [1]])
        p = p - p @ phi @ np.linalg.inv(1 + phi.T @ p @ phi) @ phi.T @ p
        k = p @ phi
        theta_hat = theta_hat + k @ (y_inmit[i] - phi.T @ theta_hat)
        theta_hat_save.append(theta_hat)

    return theta_hat_save, theta_hat_save[-1]


def lms_method():

    error_grp = []
    iter_grp = []
    # initialise the training patterns
    # y = np.array([24, 42, 63, 87, 101, 126, 135, 158, 183, 205])
    training = [[np.array([10, 1]).transpose(), 24],
                [np.array([20, 1]).transpose(), 42],
                [np.array([30, 1]).transpose(), 63],
                [np.array([40, 0]).transpose(), 87],
                [np.array([50, 1]).transpose(), 101],
                [np.array([60, 1]).transpose(), 126],
                [np.array([70, 1]).transpose(), 135],
                [np.array([80, 1]).transpose(), 158],
                [np.array([90, 1]).transpose(), 183],
                [np.array([100, 1]).transpose(), 205],
                ]
    def lms(w):
        i = 0
        alpha = [0.1, 0.01, 0.0001]
        Emax = 10000000000000000000000000
        maxIter = 1000
        E = 0
        while (i < maxIter) and (E < Emax):
            E = 0
            for pair in training:
                y = np.dot(w.transpose(), pair[0])
                w = w + np.dot((alpha[2] * (pair[1] - y)), pair[0])
                E = E + np.power((pair[1] - y), 2)
            # Put the error in the array after going thru the whole training pattern
            error_grp.append(E)
            iter_grp.append(i)
            i = i + 1
        return w

    w = []
    final_value = None
    for x_pp in range(50):
        for y in range(2):
            val = round(random.random(), 1)
            # print(val)
            w.append(val)
        weight_vector = np.array(w)
        tmp = lms(weight_vector)
        w = []
        if x_pp == 49:
            final_value = tmp

    # least square method
    # ls_method = least_square()
    lms_newY = []
    le_newY = []
    for i in range(10):
        print(final_value[0] * x_init[0][i])
        lms_newY.append(final_value[0] * x_label[i] + final_value[1])
        # le_newY.append(ls_method[0] * x_label[i] + ls_method[1])
    plt.plot(x_label, y_inmit, "o", label="data")
    plt.plot(x_label, lms_newY, "red", label="LMS")
    # plt.plot(x_label, le_newY, "blue", label="LSQ")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # least_square
    answer = least_square()
    print(answer)
    newY = []
    error = []
    for i in range(len(x_label)):
        newY.append(answer[0] * x_label[i] + answer[1])
        error.append(answer[0] * x_label[i] + answer[1] - y_inmit[i])

    t = []
    for i in range(len(error) - 1):
        print((error[i] * error[i + 1]) / len(error))
    plt.title("Least Square")
    plt.plot(x_label, y_inmit, "o")
    plt.plot(x_label, newY)
    plt.show()

    # recusive_least_square
    # all, newist = recusive_least_square()
    # t = []
    # for i in range(10):
    #     t.append(i)
    # newY = []
    # for i in range(10):
    #     newY.append(newist[0] * x_label[i] + newist[1])
    # plt.title("Recusive Least Square")
    # plt.plot(x_label, y_inmit, "o")
    # plt.plot(x_label, newY)
    # plt.show()

    # Least Mean square Method()
    # lms_method()
