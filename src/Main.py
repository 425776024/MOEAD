import time

# import src.problem.DTLZ1 as DTLZ1
# import src.problem.KUR as KUR
# import src.problem.SCH as SCH
import src.problem.ZDT1 as ZDT1
import src.problem.ZDT2 as ZDT2
import src.problem.ZDT3 as ZDT3
import src.problem.ZDT4 as ZDT4
from src.utils import Utils


class MOEAD:
    # 0最小化，1最大化
    problem_type = 0
    # problem_type=1


    Test_fun = ZDT1
    name = 'ZDT1'
    # GA_DE_Utils = Utils.DE_Utils
    GA_DE_Utils = Utils.GA_Utils

    Pop_size = -1
    max_gen = 50
    T_size = 5
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
    # need_dynamic = False
    need_dynamic = True
    draw_w = False
    # draw_w=True
    now_y = 0

    def __init__(self):
        self.Init_data()

    def Init_data(self):
        Utils.Load_W(self)
        Utils.cpt_W_Bi_T(self)
        self.GA_DE_Utils.Creat_Pop(self)
        Utils.cpt_Z(self)
        # utils.cpt_Z2(self)

    def show(self):
        if self.draw_w:
            Utils.draw_W(self)
        Utils.draw_MOEAD_Pareto(self, moead.name + "第：" + str(self.max_gen) + "")
        Utils.show()

    def run(self):
        t = time.time()
        # print('Z:', self.Z)
        EP_X_ID = self.GA_DE_Utils.envolution(self)
        dt = time.time() - t
        # print("PE size:%s,used time:%s s" % (len(EP_X_ID), dt))
        # print('Z:', self.Z)
        self.show()


if __name__ == '__main__':
    # np.random.seed(1)
    moead = MOEAD()
    moead.run()
