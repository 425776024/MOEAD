'''
求解问题部分
'''

import numpy as np

Dimention = 10
Func_num = 3
Bound = [-1000, 1000]


def Func(X):
    f1 = F1(X)
    f2 = F2(X)
    return [f1, f2]


def F1(X):
    return X[0] ** 2


def F2(X):
    return (X[0] - 2) ** 2


def F3(X):
    pass


def G(X):
    xx=(X-0.5)**2
    cc=np.cos(20*np.pi*(X-0.5))
    g=100*(Dimention-2)+100*np.sum(xx-cc,axis=0)
    return g
