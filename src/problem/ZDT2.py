'''
求解问题部分
'''


import numpy as np

Dimention = 30
Func_num = 2
Bound = [0, 1]


def Func(X):
    if X.shape[0] < 2:
        return -1
    f1 = F1(X)
    ag = g(X)
    f2 = F2(ag, X)
    return [f1, f2]


def F1(X):
    return X[0]


def F2(gx, X):
    x=X[0]
    f2 = gx * (1 - (x / gx)**2)
    return f2


def g(X):
    g = 1 + 9 * (np.sum(X[1:], axis=0) / (X.shape[0] - 1))
    return g
