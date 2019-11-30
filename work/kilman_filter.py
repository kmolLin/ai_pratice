import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = []
    y = []
    x.append(0)
    y.append(x[0] + np.random.normal(0, 1))

    # Generate data
    for i in range(1, 200):
        x.append(0.9 * x[i - 1] + np.random.normal(0, 1))
        y.append(x[i - 1] + np.random.normal(0, 1))

    xi = np.array(x).T
    yi = np.array(y).T
    theta_hat = np.array([[0], [0]])
    sigma = 1
    p = sigma * np.eye(2)
    A = np.eye(2)

    # calc all data in 1
    yy_tmp = []
    theta_hat_save = []
    for i in range(len(xi)):
        phi = np.array([[xi[i]], [1]])
        Lk = p @ phi * np.linalg.inv(1 + phi.T @ p @ phi)
        p = p + 1 - p @ phi @ np.linalg.inv(1 + phi.T @ p @ phi) @ phi.T @ p
        theta_hat = theta_hat + Lk @ (yi[i] - phi.T @ theta_hat)
        yy_tmp.append(xi[i] * theta_hat[0] + theta_hat[1])
        theta_hat_save.append(theta_hat)

    print(theta_hat)
    plt.title("Kalman")
    plt.plot(yi, "r", label="original")
    plt.plot(yy_tmp, "b", label="Kalman Filter Estimation")
    plt.legend()
    plt.show()
