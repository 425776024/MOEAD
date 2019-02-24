'''
求解问题部分
'''

import numpy as np

Dimention = 30
Func_num = 2
Bound = [0, 1]


def Func(X):
    f1 = F1(X)
    gx = g(X)
    f2 = F2(gx, X)
    return [f1, f2]


def F1(X):
    return X[0]


def F2(gx, X):
    x1 = X[0]
    f2 = gx * (1 - np.sqrt(x1 / gx) - (x1 / gx) * np.sin(10 * np.pi * x1))
    return f2


def g(X):
    g = 1 + 9 * (np.sum(X[1:], axis=0) / (X.shape[0] - 1))
    return g
