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
    a = b = 100

    # u = 10
    # v = 10
    # x, y = np.meshgrid(x, y)
    # phi = np.linspace(0, 2 * pi, 10)

    # x = u * v * np.cos(phi)
    # y = u * v * np.sin(phi)
    # z = ((x ** 2) / a + (y ** 2) / b)
    theta = np.linspace(0, 2 * pi, 100)
    radin = 25
    points = ((0, 0), (10, 10), (20, 20))
    degree = 0
    test = []
    for i in range(10):
        for j in range(10):
            test.append((i, j))
    for t in test:
        z = (((t[0]) ** 2) / a ** 2 + ((t[1]) ** 2) / b ** 2)
        ax.scatter(t[0], t[1], z)
    for point in points:
        # let step rotate 30 degree
        # x, y is limit to show the value
        x = np.linspace(-radin * np.cos(np.deg2rad(degree)), radin * np.cos(np.deg2rad(degree)), 100)
        y = np.linspace(-radin * np.sin(np.deg2rad(degree)), radin * np.sin(np.deg2rad(degree)), 100)
        z = (((x - point[0]) ** 2) / a ** 2 + ((y - point[1]) ** 2) / b ** 2)
        degree += 30
        ax.scatter(x, y, z)

    # surf = ax.plot_surface(x, y, z)
    # ax.scatter(x, y, z)

    plt.show()

