'''
求解问题部分
'''

import numpy as np

Dimention = 3
Func_num = 2
Bound = [-5, 5]


def Func(X):
    if X.shape[0] < 2:
        return -1
    f1 = F1(X)
    f2 = F2(X)
    return [f1, f2]


def F1(X):
    xx=X[: -1] ** 2 + X[1:] ** 2
    ep=np.exp(-0.2 * np.sqrt(xx))
    f = np.sum(-10 * ep, axis=0)
    return f


def F2(X):
    f = np.sum(np.abs(X) ** 0.8 + 5 * np.sin(X ** 3), axis=0)
    return f


def g(X):
    g = 1 + 9 * (np.sum(X[1:], axis=0) / (X.shape[0] - 1))
    return g
