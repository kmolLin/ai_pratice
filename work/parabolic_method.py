import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from math import pi
from mpl_toolkits import mplot3d as mp

fig = plt.figure()
ax = mp.Axes3D(fig)

# x = u * v * cos(phi)
# y = u * v * sin(phi)
# z = 1/2 * (u^2 - v^2)

if __name__ == '__main__':

    # x = np.linspace(-5, 5, 100)
    # y = np.linspace(-5, 5, 100)
    a = 0.5
    b = 0.10

    # u = 10
    # v = 10
    # x, y = np.meshgrid(x, y)
    # phi = np.linspace(0, 2 * pi, 10)

    # x = u * v * np.cos(phi)
    # y = u * v * np.sin(phi)
    # z = ((x ** 2) / a + (y ** 2) / b)
    theta = np.linspace(0, 2 * pi, 100)
    radin = 100
    points = ((0, 0), (10, 10), (20, 20))
    degree = 0
    pathpoints = []
    degree_tmp = []
    x_tmp = []
    r = np.linspace(-5, 5, 100)

    p = 4
    for i in range(0, p):
        pathpoints.append((i, i))
        degree_tmp.append(np.deg2rad(i * 45))
        # x_tmp.append(i)

    x_tmp = np.linspace(-5, 5, p)

    for i in range(p):
        tt = np.linspace(-10 - pathpoints[i][0], 10 + pathpoints[i][1], 100)
        t1x = tt * np.cos(degree_tmp[i])
        t1y = tt * np.sin(degree_tmp[i])
        zz = (t1x - pathpoints[i][0]) ** 2 + (t1y - pathpoints[i][1]) ** 2
        ax.plot(t1x, t1y, zz, "b")

        for j in range(p):
            r = x_tmp[i] / np.cos(degree_tmp[j])
            y = r * np.sin(degree_tmp[j])
            z = (x_tmp[i] - pathpoints[j][0]) ** 2 + (y - pathpoints[j][1]) ** 2

            if z > 250:
                continue
            ax.scatter(x_tmp[i], y, z, c="r")

    plt.show()
    exit()
    # let x = 1
    # for i in range(0, 20):
    #     # y = x_tmp[i] * (1 / np.tan(degree_tmp[i]))
    #     # x, y is limit to show the value
    #     x = np.linspace(-radin * np.cos(degree_tmp[i]), radin * np.cos(degree_tmp[i]), 100)
    #     y = np.linspace(-radin * np.sin(degree_tmp[i]), radin * np.sin(degree_tmp[i]), 100)
    #     z = (((x - pathpoints[i][0]) ** 2) / a ** 2 + ((y - pathpoints[i][1]) ** 2) / b ** 2)
    #     # if z > 20000:
    #     #     continue
    #     print(i)
    #     ax.plot(x, y, z)

    # for t in test:
    #     z = (((t[0]) ** 2) / a ** 2 + ((t[1]) ** 2) / b ** 2)
    #     ax.scatter(t[0], t[1], z)

    # for point in points:
    #     # let step rotate 30 degree
    #     # x, y is limit to show the value
    #     x = np.linspace(-radin * np.cos(np.deg2rad(degree)), radin * np.cos(np.deg2rad(degree)), 100)
    #     y = np.linspace(-radin * np.sin(np.deg2rad(degree)), radin * np.sin(np.deg2rad(degree)), 100)
    #     z = (((x - point[0]) ** 2) / a ** 2 + ((y - point[1]) ** 2) / b ** 2)
    #     degree += 30
    #     ax.scatter(x, y, z)

    # surf = ax.plot_surface(x, y, z)
    # ax.scatter(x, y, z)

    plt.show()

