import random

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def show():
    plt.show()


def draw_MOEAD_Pareto(Pareto_F_Data, Pop_F_Data, name):
    Len = len(Pareto_F_Data[0])
    if Len == 2:
        plt.xlabel('Function 1', fontsize=15)
        plt.ylabel('Function 2', fontsize=15)
        plt.title(name)
        # plt.xlim(0, 2)
        # plt.ylim(0, 2)
        for pp in Pop_F_Data:
            plt.scatter(pp[0], pp[1], c='black', s=5)
        for p in Pareto_F_Data:
            plt.scatter(p[0], p[1], c='red', s=10)
    if Len == 3:
        pass


def draw_W():
    data = np.loadtxt('test.csv')
    if data.shape[1] == 3:
        fig = plt.figure()
        ax = Axes3D(fig)
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        VecStart_x = np.zeros(data.shape[0])
        VecStart_y = np.zeros(data.shape[0])
        VecStart_z = np.zeros(data.shape[0])
        VecEnd_x = data[:, 0]
        VecEnd_y = data[:, 1]
        VecEnd_z = data[:, 2]
        ax.scatter(x, y, z, marker='.', s=50, label='', color='r')
        for i in range(VecStart_x.shape[0]):
            ax.plot([VecStart_x[i], VecEnd_x[i]], [VecStart_y[i], VecEnd_y[i]], zs=[VecStart_z[i], VecEnd_z[i]])
    if data.shape[1] == 2:
        x, y = data[:, 0], data[:, 1]
        plt.xlabel('X')
        plt.xlabel('Y')
        VecStart_x = np.zeros(data.shape[0])
        VecStart_y = np.zeros(data.shape[0])
        VecEnd_x = data[:, 0]
        VecEnd_y = data[:, 1]
        plt.scatter(x, y, marker='.', s=50, label='', color='r')
        for i in range(VecStart_x.shape[0]):
            plt.plot([VecStart_x[i], VecEnd_x[i]], [VecStart_y[i], VecEnd_y[i]])

