import os
from math import sqrt

import numpy as np
import src.problem.SCH as sch
import src.problem.KUR as kur
import src.problem.ZDT1 as zdt1
import src.problem.ZDT2 as zdt2
import src.problem.ZDT4 as zdt4
import matplotlib.pyplot as plt
import src.Utils as utils
from src.Mean_Vector_Util import Mean_vector


class MOEAD:
    # 0最小化，1最大化
    problem_type = 0
    # problem_type=1
    Test_fun = kur
    name = 'kur'
    Pop_size = -1
    max_gen = 300
    T_size = 30
    h = 100
    m = 2
    c_rate = 1
    m_rate = 0.1
    EP_X = []
    # 种群
    Pop = []
    W = []
    W_Bi_T = []
    Z = []

    def __init__(self):
        pass

    def Creat_child(self):
        child = self.Test_fun.Bound[0] + (self.Test_fun.Bound[1] - self.Test_fun.Bound[0]) * np.random.rand(
            self.Test_fun.Dimention)
        return child

    def Creat_Pop(self):
        Pop = []
        if self.Pop_size < 1:
            print('error in creat_Pop')
            return -1
        while len(Pop) != self.Pop_size:
            Pop.append(self.Creat_child())
        return Pop

    def Load_W(self, name='test.csv'):
        if os.path.exists(name) == False:
            print('not exists')
            mv = Mean_vector(self.h, self.m, name)
            mv.generate()
            print('created')
        W = np.loadtxt(fname=name)
        self.Pop_size = W.shape[0]
        return W

    def cpt_W_Bi_T(self):
        # 计算权重的T个邻居
        if self.T_size < 1:
            return -1
        for bi in range(self.W.shape[0]):
            Bi = self.W[bi]
            DIS = np.sum((self.W - Bi) ** 2, axis=1)
            B_T = np.argsort(DIS)
            B_T = B_T[1:self.T_size + 1]
            self.W_Bi_T.append(B_T)

    def cpt_Z(self):
        # 更新Z集
        Z = []
        for fi in range(self.Test_fun.Func_num):
            z_i = self.Test_fun.Func(self.Pop[0])[fi]
            for pi in range(1, self.Pop_size):
                zpi = self.Test_fun.Func(self.Pop[pi])[fi]
                if self.problem_type == 0:
                    if zpi < z_i:
                        z_i = zpi
                if self.problem_type == 1:
                    if zpi > z_i:
                        z_i = zpi
            Z.append(z_i)
        return Z

    # def cpt_FV(self):
    #     FV = []
    #     for xi in self.Pop:
    #         fi = self.Test_fun.Func(xi)
    #         FV.append(fi)
    #     return FV

    def cpt_tchbycheff(self, idx, X):
        # idx X在种群中的位置
        max = self.Z[0]
        ri = self.W[idx]
        F_X = self.Test_fun.Func(X)
        for i in range(self.Test_fun.Func_num):
            fi = self.Tchebycheff_dist(ri[i], F_X[i], self.Z[i])
            if fi > max:
                max = fi
        return max

    def cross_mutation(self, p1, p2):
        y1 = np.copy(p1)
        y2 = np.copy(p2)
        if np.random.rand() < self.c_rate:
            yj = 0
            uj = np.random.rand()
            if uj < 0.5:
                yj = (2 * uj) ** (1 / 3)
            else:
                yj = (1 / (2 * (1 - uj))) ** (1 / 3)
            y1 = 0.5 * (1 + yj) * y1 + (1 - yj) * y2
            y2 = 0.5 * (1 - yj) * y1 + (1 + yj) * y2
            y1[y1 > self.Test_fun.Bound[1]] = self.Test_fun.Bound[1]
            y1[y1 < self.Test_fun.Bound[0]] = self.Test_fun.Bound[0]
            y2[y2 > self.Test_fun.Bound[1]] = self.Test_fun.Bound[1]
            y2[y2 < self.Test_fun.Bound[0]] = self.Test_fun.Bound[0]
        if np.random.rand() < self.m_rate:
            dj = 0
            uj = np.random.rand()
            if uj < 0.5:
                dj = (2 * uj) ** (1 / 3) - 1
            else:
                dj = 1 - 2 * (1 - uj) ** (1 / 6)
            y1 = y1 + dj
            y2 = y2 + dj
            y1[y1 > self.Test_fun.Bound[1]] = self.Test_fun.Bound[1]
            y1[y1 < self.Test_fun.Bound[0]] = self.Test_fun.Bound[0]
            y2[y2 > self.Test_fun.Bound[1]] = self.Test_fun.Bound[1]
            y2[y2 < self.Test_fun.Bound[0]] = self.Test_fun.Bound[0]
        return y1, y2

    def EO(self, p1):
        m = p1.shape[0]
        tp_best = np.copy(p1)
        for i in range(m):
            temp_best = np.copy(tp_best)
            temp_best[i] = temp_best[i] + np.random.normal(0, 0.3, 1)
            temp_best[temp_best > self.Test_fun.Bound[1]] = self.Test_fun.Bound[1]
            temp_best[temp_best < self.Test_fun.Bound[0]] = self.Test_fun.Bound[0]
            f = self.is_dominate(temp_best, tp_best)
            if f:
                tp_best[:] = temp_best
        return tp_best

    def cpt_to_Z_dist(self, X):
        #  X点到参考点距离
        F_X = self.Test_fun.Func(X)
        d = 0
        for i, fm in enumerate(F_X):
            d = d + (fm - self.Z[i]) ** 2
        d = sqrt(d)
        return d

    def update_Z(self, Y):
        F_y = self.Test_fun.Func(Y)
        for j in range(self.Test_fun.Func_num):
            if self.problem_type == 0:  # minimize
                if self.Z[j] > F_y[j]:
                    self.Z[j] = F_y[j]
            if self.problem_type == 1:  # maximize
                if self.Z[j] < F_y[j]:
                    self.Z[j] = F_y[j]

    def update_BTX(self, P_B, Y):
        for j in P_B:
            Xj = self.Pop[j]
            d_x = self.cpt_tchbycheff(j, Xj)
            d_y = self.cpt_tchbycheff(j, Y)
            if d_y < d_x:
                self.Pop[j], _ = self.cross_mutation(Xj, np.copy(Y))

    def update_EP(self, Y):
        i = 0
        for pi, P in enumerate(self.EP_X):
            if i != 0:
                break
            if self.is_dominate(Y, P):
                del self.EP_X[pi]
            if self.is_dominate(P, Y):
                i += 1
        if i == 0:
            self.EP_X.append(Y)

    def cpt_EP(self):
        if len(self.EP_X) == 0:
            for pi, p in enumerate(self.Pop):
                np = 0
                for ppi, pp in enumerate(self.Pop):
                    if pi != ppi:
                        if self.is_dominate(pp, p):
                            np += 1
                if np == 0:
                    self.EP_X.append(p)
        else:
            for pi, p in enumerate(self.Pop):
                np = 0
                for epi, ep in enumerate(self.EP_X):
                    if self.is_dominate(ep, p):
                        np += 1
                        break
                    if self.is_dominate(p, ep):
                        del self.EP_X[epi]
                if np == 0:
                    self.EP_X.append(p)

    def is_dominate(self, X, Y):
        # x if dominate y
        F_X = self.Test_fun.Func(X)
        F_Y = self.Test_fun.Func(Y)
        i = 0
        if self.problem_type == 0:  # minimize
            for xv, yv in zip(F_X, F_Y):
                if xv < yv:
                    i = i + 1
                if xv > yv:
                    return False
        if self.problem_type == 1:  # maximize
            for xv, yv in zip(F_X, F_Y):
                if xv > yv:
                    i = i + 1
                if xv < yv:
                    return False
        if i != 0:
            return True
        return False

    def Init_data(self):
        self.W = self.Load_W(name=self.name + '.csv')
        self.cpt_W_Bi_T()
        self.Pop = self.Creat_Pop()
        self.cpt_EP()
        # self.FV = self.cpt_FV()
        self.Z = self.cpt_Z()

    def generate_next(self, p1, p2):
        p1, p2 = self.cross_mutation(p1, p2)
        p1 = self.EO(p1)
        p2 = self.EO(p2)
        Y = np.copy(p2)
        if self.is_dominate(p1, p2):
            Y = np.copy(p1)
        return Y

    def envolution(self):
        for gen in range(self.max_gen):
            print('gena %s,EP len :%s' % (gen, len(self.EP_X)))
            for pi, p in enumerate(self.Pop):
                # 第pi个个体的邻居集
                Bi = self.W_Bi_T[pi]
                k = np.random.randint(self.T_size)
                l = np.random.randint(self.T_size)
                #     随机从邻居内选2个个体，产生新解
                if k == l:
                    break
                ik = Bi[k]
                il = Bi[l]
                Xk = self.Pop[ik]
                Xl = self.Pop[il]
                Y = self.generate_next(Xk, Xl)
                self.Pop[ik][:] = Y[:]
                self.update_Z(Y)
                self.update_BTX(Bi, Y)
                self.update_EP(Y)
        # self.cpt_EP()
        return self.EP_X

    def Tchebycheff_dist(self, w, f, z):
        return w * abs(f - z)

    def run(self):
        self.Init_data()
        self.envolution()
        print(len(self.EP_X))
        # utils.draw_W()
        Pareto_F_Data = []
        Pop_F_Data = []
        for P in self.EP_X:
            Pareto_F_Data.append(self.Test_fun.Func(P))
        for p in self.Pop:
            Pop_F_Data.append(self.Test_fun.Func(p))
        utils.draw_MOEAD_Pareto(Pareto_F_Data, Pop_F_Data, self.name)
        utils.show()


if __name__ == '__main__':
    moead = MOEAD()
    moead.run()
