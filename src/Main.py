import time

# import src.problem.DTLZ1 as DTLZ1
import src.problem.KUR as KUR
# import src.problem.SCH as SCH
import src.problem.ZDT1 as ZDT1
import src.problem.ZDT2 as ZDT2
import src.problem.ZDT3 as ZDT3
import src.problem.ZDT4 as ZDT4
from src.utils import Utils


class MOEAD:
    # 0表示最小化目标求解，1最大化目标求解。（约定）
    problem_type = 0
    # problem_type=1
    # 测试函数
    Test_fun = ZDT1
    # 动态展示的时候的title名称
    name = 'ZDT1'
    # 使用那种方式、DE/GA 作为进化算法
    # GA_DE_Utils = Utils.DE_Utils
    GA_DE_Utils = Utils.GA_Utils

    # 种群大小，取决于vector_csv_file/下的xx.csv
    Pop_size = -1
    # 最大迭代次数
    max_gen = 50
    # 邻居设定（只会对邻居内的相互更新、交叉）
    T_size = 5
    # 支配前沿ID
    EP_X_ID = []
    # 支配前沿 的 函数值
    EP_X_FV = []

    # 种群
    Pop = []
    # 种群计算出的函数值
    Pop_FV = []
    # 权重
    W = []
    # 权重的T个邻居。比如：T=2，(0.1,0.9)的邻居：(0,1)、(0.2,0.8)。永远固定不变
    W_Bi_T = []
    # 理想点。（比如最小化，理想点是趋于0）
    # ps:实验结论：如果你知道你的目标，比如是极小化了，且理想极小值(假设2目标)是[0,0]，
    # 那你就一开始的时候就写死moead.Z=[0,0]吧
    Z = []
    # 权重向量存储目录
    csv_file_path = 'vector_csv_file'
    # 当前迭代代数
    gen = 0
    # 是否动态展示
    # need_dynamic = False
    need_dynamic = True
    # 是否画出权重图
    draw_w = True
    # 用于绘图：当前进化种群中，哪个，被，正在 进化。draw_w=true的时候可见
    now_y = []

    # draw_w=True

    def __init__(self):
        self.Init_data()

    def Init_data(self):
        # 加载权重
        Utils.Load_W(self)
        # 计算每个权重Wi的T个邻居
        Utils.cpt_W_Bi_T(self)
        # 创建种群
        self.GA_DE_Utils.Creat_Pop(self)
        # 初始化Z集，最小问题0,0
        Utils.cpt_Z(self)

    def show(self):
        if self.draw_w:
            Utils.draw_W(self)
        Utils.draw_MOEAD_Pareto(self, moead.name + "num:" + str(self.max_gen) + "")
        Utils.show()

    def run(self):
        t = time.time()
        # EP_X_ID：支配前沿个体解，的ID。在上面数组：Pop，中的序号
        # envolution开始进化
        EP_X_ID = self.GA_DE_Utils.envolution(self)
        print('你拿以下序号到上面数组：Pop中找到对应个体，就是多目标优化的函数的解集啦!')
        print("支配前沿个体解，的ID（在上面数组：Pop，中的序号）：", EP_X_ID)
        dt = time.time() - t
        self.show()


if __name__ == '__main__':
    # np.random.seed(1)
    moead = MOEAD()
    moead.run()
