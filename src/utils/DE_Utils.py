import src.utils.MOEAD_Utils as MOEAD_Utils
import src.utils.Draw_Utils as Draw_Utils
import numpy as np

'''
差分进化算法工具包
'''

# 交叉率
Cross_Rate = 0.5


def Creat_child(moead):
    # 创建一个个体
    # （就是一个向量，长度为Dimention，范围在moead.Test_fun.Bound中设定）
    child = moead.Test_fun.Bound[0] + (moead.Test_fun.Bound[1] - moead.Test_fun.Bound[0]) * np.random.rand(
        moead.Test_fun.Dimention)
    return child


def Creat_Pop(moead):
    Pop = []
    Pop_FV = []
    if moead.Pop_size < 1:
        print('error in creat_Pop')
        return -1
    while len(Pop) != moead.Pop_size:
        X = Creat_child(moead)
        Pop.append(X)
        Pop_FV.append(moead.Test_fun.Func(X))
    moead.Pop, moead.Pop_FV = Pop, Pop_FV
    return Pop, Pop_FV


def mutate(moead, best, p1, p2):
    f = 0.5 + 1.5 * np.random.rand()  # 缩放因子
    d = f * (p1 - p2)
    temp_p = best + d
    temp_p[temp_p > moead.Test_fun.Bound[1]] = moead.Test_fun.Bound[0] + (
            moead.Test_fun.Bound[1] - moead.Test_fun.Bound[0]) * np.random.rand()
    temp_p[temp_p < moead.Test_fun.Bound[0]] = moead.Test_fun.Bound[0] + (
            moead.Test_fun.Bound[1] - moead.Test_fun.Bound[0]) * np.random.rand()
    return temp_p


def crossover(moead, p1, vi):
    var_num = moead.Test_fun.Dimention
    ui = np.zeros(var_num)
    k = np.random.random_integers(0, var_num - 1)
    for j in range(0, var_num):
        if np.random.random() < Cross_Rate or j == k:
            ui[j] = vi[j]
        else:
            ui[j] = p1[j]
    return ui


def generate_next(moead, wi, p0, p1, p2):
    qbxf_p0 = MOEAD_Utils.cpt_tchbycheff(moead, wi, p0)
    qbxf_p1 = MOEAD_Utils.cpt_tchbycheff(moead, wi, p1)
    qbxf_p2 = MOEAD_Utils.cpt_tchbycheff(moead, wi, p2)
    arr = [p0, p1, p2]
    qbxf = np.array([qbxf_p0, qbxf_p1, qbxf_p2])
    index = np.argsort(qbxf)
    best = arr[index[0]]
    bw = arr[index[2]]
    bm = arr[index[1]]

    vi = mutate(moead, best, bm, bw)
    ui = crossover(moead, p0, vi)
    return ui


def envolution(moead):
    for gen in range(moead.max_gen):
        moead.gen = gen
        for pi, p in enumerate(moead.Pop):
            # 第pi个个体的邻居集
            Bi = moead.W_Bi_T[pi]
            k = np.random.randint(moead.T_size)
            l = np.random.randint(moead.T_size)
            #     随机从邻居内选2个个体，产生新解
            ik = Bi[k]
            il = Bi[l]
            Xi = moead.Pop[pi]
            Xk = moead.Pop[ik]
            Xl = moead.Pop[il]

            Y = generate_next(moead, pi, Xi, Xk, Xl)
            cbxf_i = MOEAD_Utils.cpt_tchbycheff(moead, pi, Xi)

            cbxf_y = MOEAD_Utils.cpt_tchbycheff(moead, pi, Y)

            d = 0.001
            if cbxf_y < cbxf_i:
                moead.now_y = pi
                moead.Pop[pi] = np.copy(Y)
                F_Y = moead.Test_fun.Func(Y)[:]
                MOEAD_Utils.update_EP_By_ID(moead, pi, F_Y)
                MOEAD_Utils.update_Z(moead, Y)
                if abs(cbxf_y - cbxf_i) > d:
                    MOEAD_Utils.update_EP_By_Y(moead, pi)
            MOEAD_Utils.update_BTX(moead, Bi, Y)

        if moead.need_dynamic:
            Draw_Utils.plt.cla()
            if moead.draw_w:
                Draw_Utils.draw_W(moead)
            Draw_Utils.draw_MOEAD_Pareto(moead, moead.name + "第：" + str(gen) + "")
            Draw_Utils.plt.pause(0.001)
        print('gen %s,EP size :%s,Z:%s' % (gen, len(moead.EP_X_ID), moead.Z))
    return moead.EP_X_ID
