'''
求解问题部分
'''

import numpy as np

Dimention = 10
Func_num = 3
Bound = [0, 1]
gx = -1


def Func(X):
    f1 = F1(X)
    f2 = F2(X)
    f3 = F3(X)
    return [f1, f2, f3]


def F1(X):
    global gx
    if gx == -1:
        gx = G(X)
    f = (1 + gx) * X[0] * X[1]
    return f


def F2(X):
    global gx
    if gx == -1:
        gx = G(X)
    f = (1 + gx) * X[0] * (1 - X[1])
    return f


def F3(X):
    global gx
    if gx == -1:
        gx = G(X)
    f = (1 + gx) * (1 - X[0])
    return f


def G(X):
    xx = (X[2:] - 0.5) ** 2
    cc = np.cos(20 * np.pi * (X[2:] - 0.5))
    g = 100 * (Dimention - 2) + 100 * np.sum(xx - cc, axis=0)
    return g
