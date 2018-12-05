import os
import time
from math import sqrt

import numpy as np
import src.problem.SCH as sch
import src.problem.KUR as kur
import src.problem.ZDT1 as zdt1
import src.problem.ZDT2 as zdt2
import src.problem.ZDT4 as zdt4
import src.problem.DTLZ1 as dtlz1
import matplotlib.pyplot as plt
import src.Utils as utils


class MOEAD:
    # 0最小化，1最大化
    problem_type = 0
    # problem_type=1
    Test_fun = zdt2
    name = 'zdt2'
    Pop_size = -1
    max_gen = 100
    T_size = 20
    # popsize
    h = 200
    # m = 2
    EP_X_ID = []
    EP_X_FV = []

    # 种群
    Pop = []
    Pop_FV = []

    W = []
    W_Bi_T = []
    Z = []
    csv_file_path = 'vector_csv_file'
    # 当前迭代代数
    gen = 0
    # 是否动态展示
    need_dinamic = False

    # need_dinamic=True

    def __init__(self):
        self.Init_data()

    def Init_data(self):
        utils.Load_W(self)
        utils.cpt_W_Bi_T(self)
        utils.Creat_Pop(self)
        utils.cpt_EP(self)
        utils.cpt_Z(self)

    def show(self):
        # utils.draw_W(self)
        utils.draw_MOEAD_Pareto(self, moead.name + "第：" + str(self.max_gen) + "")
        utils.show()

    def run(self):
        t = time.time()
        EP_X_ID = utils.envolution(self)
        dt = time.time() - t
        print("PE size:%s,used time:%s s" % (len(EP_X_ID), dt))
        self.show()


if __name__ == '__main__':
    moead = MOEAD()
    moead.run()
