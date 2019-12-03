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
    # radin = 100
    # points = ((0, 0), (10, 10), (20, 20))
    degree = 0
    pathpoints = []
    degree_tmp = []
    x_tmp = []
    r = np.linspace(-5, 5, 100)

    p = 4
    # for i in range(0, p):
    #     pathpoints.append((i, i))
    #     degree_tmp.append(np.deg2rad(i * 45))
    #     # x_tmp.append(i)

    points_path = []
    with open("ttt.txt", "r") as w:
        files = w.readlines()
        i = 0
        for file in files:
            # print(float(file.split(",")[0]), float(file.split(",")[1]))
            points_path.append((float(file.split(",")[0]), float(file.split(",")[1])))
            degree_tmp.append(np.deg2rad(i * 24))
            i = i + 1

    x_tmp = np.linspace(-0, 21, 2000)
    # for i in range(len(degree_tmp)):
        # tt = np.linspace((-10 - points_path[i][0]) * 2, (10 + points_path[i][1]) * 2, 100)
        # t1x = tt * np.cos(degree_tmp[i]) - points_path[i][0]
        # t1y = tt * np.sin(degree_tmp[i]) - points_path[i][1]
        # zz = t1x ** 2 + t1y ** 2
        # ax.plot(t1x, t1y, zz, "b")
        # pass
    y_tmp = []
    z_tmp = []
    i = 0
    for x_t in x_tmp:
        i += 1
        for j in range(len(degree_tmp)):
            # if degree_tmp[j] == 0:
            #     continue
            y = (x_t - points_path[j][0]) * np.tan(degree_tmp[j]) + points_path[j][1]
            z = np.sqrt((x_t - points_path[j][0]) ** 2 + (y - points_path[j][1]) ** 2)
            if points_path[j][1] > 0:
                cc = "r"
            else:
                cc = "b"
            if y > 13:
                continue
            if y < -4:
                continue
            if z > 5:
                continue
            if z < 0.5:
                if -0.5 < y < 0.5:
                    ax.scatter(x_t, y, z, s=4, c=cc)

            # r = (x_t - points_path[j][0]) / np.cos(degree_tmp[j])
            # y = r * np.sin(degree_tmp[j]) + points_path[j][1]
            # y = (np.sin(degree_tmp[j]) / np.sin((np.pi / 2) - degree_tmp[j])) * (x_t - points_path[j][0])\
            #     + points_path[j][1]
            # print(x_t, y)
            # xx, yy = np.meshgrid(x_t, y)
            # z = x_t ** 2 + y ** 2
            # print(z)
            if z > 200:
                continue
            # y_tmp.append(y)
            # z_tmp.append(z)
            # # ax.scatter(x_t, y, z, c="r", s=1)
            # ax.scatter(x_t, y, z, s=0.7, c=cc)
            # ax.scatter(x_t, y, 1)

            # second method of Z dimesion
            # x = np.sqrt(x_t) * np.cos(degree_tmp[j]) + points_path[j][0]
            # y = np.sqrt(x_t) * np.sin(degree_tmp[j]) + points_path[j][1]
            #
            # x2 = np.sqrt(x_t) * np.cos(degree_tmp[j] + np.pi) + points_path[j][0]
            # y2 = np.sqrt(x_t) * np.cos(degree_tmp[j] + np.pi) + points_path[j][1]
            #
            # ax.scatter(x, y, x_t, c="r", s=0.7)
            # ax.scatter(x2, y2, x_t, c="b", s=0.7)

    # aa, bb = np.meshgrid(x_tmp, y_tmp)
    # ax.plot_wireframe(aa, bb, z_tmp)
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

