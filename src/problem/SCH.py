'''
求解问题部分
'''

import numpy as np
# 函数的维度（目标维度不一致的自行编写目标函数）
Dimention = 30
# 目标函数个数
Func_num = 2
Bound = [-1000, 1000]


def Func(X):
    f1 = F1(X)
    f2 = F2(X)
    return [f1, f2]


def F1(X):
    return X[0] ** 2


def F2(X):
    return (X[0] - 2) ** 2
