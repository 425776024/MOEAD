import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
绘图工具包
'''

fig = plt.figure()
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
ax = 0


def show():
    plt.show()


def draw_MOEAD_Pareto(moead, name):
    Pareto_F_ID = moead.EP_X_ID
    Pop_F_Data = moead.Pop_FV

    Len = len(Pop_F_Data[0])
    if Len == 2:
        r_x = Pop_F_Data[0][:]
        r_y = Pop_F_Data[0][:]
        for pi, pp in enumerate(Pop_F_Data):
            plt.scatter(pp[0], pp[1], c='black', s=5)
        for pid in Pareto_F_ID:
            p = Pop_F_Data[pid]
            if p[0] < r_x[0]:
                r_x[0] = p[0]
            if p[0] > r_x[1]:
                r_x[1] = p[0]
            if p[1] < r_y[0]:
                r_y[0] = p[0]
            if p[1] > r_y[1]:
                r_y[1] = p[1]
            # cc=(np.random.rand(),np.random.rand(),np.random.rand())
            plt.scatter(p[0], p[1], c='r', s=20)
            # plt.text(p[0], p[1]+0.2*np.random.rand(), str(pid), fontsize=10)

        plt.xlabel('Function 1', fontsize=15)
        plt.ylabel('Function 2', fontsize=15)
        plt.title(name)
        # plt.xlim(r_x[0] - 0.1, r_x[1] + 0.1)
        # plt.ylim(r_y[0] - 0.1, r_y[1] + 0.1)
    # if Len == 3:
        # global ax, fig
        # if ax == 0:
        #     ax = Axes3D(fig)
        # ax.set_xlabel('Function 1')
        # ax.set_ylabel('Function 2')
        # ax.set_zlabel('Function 3')
        # for pp in Pop_F_Data:
        #     ax.scatter(pp[0], pp[1], pp[2], c='black', s=5)
        # for pid in Pareto_F_ID:
        #     p = Pop_F_Data[pid]
        #     ax.scatter(p[0], p[1], p[2], c='red', s=10)
        # ax.set_xlim([0,20])
        # ax.set_ylim([0,20])
        # ax.set_zlim([0,20])


def draw_W(moead):
    Start_Pts = moead.Z
    path = moead.csv_file_path + '/' + moead.name + '.csv'
    data = np.loadtxt(path)
    Pareto_F_ID = moead.EP_X_ID
    Pop_F_Data = moead.Pop_FV

    if data.shape[1] == 3:
        global ax, fig
        if ax == 0:
            ax = Axes3D(fig)
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        VecStart_x = Start_Pts[0]
        VecStart_y = Start_Pts[1]
        VecStart_z = Start_Pts[2]
        VecEnd_x = data[:, 0]
        VecEnd_y = data[:, 1]
        VecEnd_z = data[:, 2]
        ax.scatter(x, y, z, marker='.', s=50, label='', color='r')
        for i in range(VecEnd_x.shape[0]):
            ax.plot([VecStart_x, VecEnd_x[i]], [VecStart_y, VecEnd_y[i]], zs=[VecStart_z, VecEnd_z[i]])
    if data.shape[1] == 2:
        # x, y = data[:, 0], data[:, 1]
        # plt.xlabel('X')
        # plt.xlabel('Y')
        # plt.scatter(x, y, marker='.', s=50, label='', color='r')

        VecStart_x = Start_Pts[0]
        VecStart_y = Start_Pts[1]
        VecEnd_x = data[:, 0]
        VecEnd_y = data[:, 1]
        for i in range(VecEnd_y.shape[0]):
            if i == moead.now_y:
                plt.plot([VecEnd_x[i], Pop_F_Data[i][0]], [VecEnd_y[i], Pop_F_Data[i][1]])
            plt.plot([VecStart_x, VecEnd_x[i]], [VecStart_y, VecEnd_y[i]])
