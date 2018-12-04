import numpy as np
import src.MOEAD_Utils as moead_utils
import src.Draw_Utils as draw_utils


def Creat_child(moead):
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


def mutate(var_num, p1):
    for i in range(int(var_num * 0.1)):
        j = np.random.randint(0, var_num, size=1)[0]
        d = np.random.randint(0, var_num, size=1)[0]
        p1[j] = p1[d]
    return p1


def crossover_pop(var_num, pop1, pop2):
    r1 = int(var_num * np.random.rand())
    if np.random.rand() < 0.5:
        pop1[:r1], pop2[:r1] = pop2[:r1], pop1[:r1]
    else:
        pop1[r1:], pop2[r1:] = pop2[r1:], pop1[r1:]
    return pop1, pop2


def EO(moead, wi, p1):
    m = p1.shape[0]
    tp_best = np.copy(p1)
    qbxf_tp = moead_utils.cpt_tchbycheff(moead, wi, tp_best)
    Up = np.sqrt(moead.Test_fun.Bound[1] - moead.Test_fun.Bound[0]) / 2
    for i in range(m):
        temp_best = np.copy(p1)
        rd = np.random.normal(0, Up, 1)
        temp_best[i] = temp_best[i] + rd
        temp_best[temp_best > moead.Test_fun.Bound[1]] = moead.Test_fun.Bound[1]
        temp_best[temp_best < moead.Test_fun.Bound[0]] = moead.Test_fun.Bound[0]
        qbxf_te = moead_utils.cpt_tchbycheff(moead, wi, temp_best)
        if qbxf_te < qbxf_tp:
            qbxf_tp = qbxf_te
            tp_best[:] = temp_best[:]
    return tp_best


def cross_mutation(moead, p1, p2):
    y1 = np.copy(p1)
    y2 = np.copy(p2)
    c_rate = 1
    m_rate = 0.1

    if np.random.rand() < c_rate:
        yj = 0
        uj = np.random.rand()
        if uj < 0.5:
            yj = (2 * uj) ** (1 / 3)
        else:
            yj = (1 / (2 * (1 - uj))) ** (1 / 3)
        y1 = 0.5 * (1 + yj) * y1 + (1 - yj) * y2
        y2 = 0.5 * (1 - yj) * y1 + (1 + yj) * y2
        y1[y1 > moead.Test_fun.Bound[1]] = moead.Test_fun.Bound[1]
        y1[y1 < moead.Test_fun.Bound[0]] = moead.Test_fun.Bound[0]
        y2[y2 > moead.Test_fun.Bound[1]] = moead.Test_fun.Bound[1]
        y2[y2 < moead.Test_fun.Bound[0]] = moead.Test_fun.Bound[0]
    if np.random.rand() < m_rate:
        dj = 0
        uj = np.random.rand()
        if uj < 0.5:
            dj = (2 * uj) ** (1 / 6) - 1
        else:
            dj = 1 - 2 * (1 - uj) ** (1 / 6)
        y1 = y1 + dj
        y2 = y2 + dj
        y1[y1 > moead.Test_fun.Bound[1]] = moead.Test_fun.Bound[1]
        y1[y1 < moead.Test_fun.Bound[0]] = moead.Test_fun.Bound[0]
        y2[y2 > moead.Test_fun.Bound[1]] = moead.Test_fun.Bound[1]
        y2[y2 < moead.Test_fun.Bound[0]] = moead.Test_fun.Bound[0]
    return y1, y2


def generate_next(moead, gen, wi, p0, p1, p2):
    qbxf_p0 = moead_utils.cpt_tchbycheff(moead, wi, p0)
    qbxf_p1 = moead_utils.cpt_tchbycheff(moead, wi, p1)
    qbxf_p2 = moead_utils.cpt_tchbycheff(moead, wi, p2)
    n_p0, n_p1, n_p2 = np.copy(p0), np.copy(p1), np.copy(p2)
    if np.random.rand() < 1 / 20:
        n_p0 = EO(moead, wi, n_p0)
        n_p1 = EO(moead, wi, n_p1)
        n_p2 = EO(moead, wi, n_p2)

    n_p0, n_p1 = cross_mutation(moead, n_p0, n_p1)
    n_p1, n_p2 = cross_mutation(moead, n_p1, n_p2)

    qbxf_np0 = moead_utils.cpt_tchbycheff(moead, wi, n_p0)
    qbxf_np1 = moead_utils.cpt_tchbycheff(moead, wi, n_p1)
    qbxf_np2 = moead_utils.cpt_tchbycheff(moead, wi, n_p2)

    qbxf = np.array([qbxf_p0, qbxf_p1, qbxf_p2, qbxf_np0, qbxf_np1, qbxf_np2])
    mins = np.argmin(qbxf)
    Y = [p0, p1, p2, n_p0, n_p1, n_p2][mins]
    return Y


def envolution(moead):
    for gen in range(moead.max_gen):
        moead.gen = gen
        # if gen % 100 == 0:
        #     moead.need_dinamic = True
        # else:
        #     moead.need_dinamic = False
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
            Y = generate_next(moead, gen, pi, Xi, Xk, Xl)
            cbxf_i = moead_utils.cpt_tchbycheff(moead, pi, Xi)
            cbxf_y = moead_utils.cpt_tchbycheff(moead, pi, Y)
            d = 0.000
            if cbxf_y < cbxf_i:
                moead.Pop[pi][:] = Y[:]
                moead.Pop_FV[pi][:] = moead.Test_fun.Func(Y)[:]
                moead_utils.update_Z(moead, Y)
                moead_utils.update_BTX(moead, Bi, Y)
                if abs(cbxf_y - cbxf_i) > d:
                    moead_utils.update_EP_By_Y(moead, pi)
        if moead.need_dinamic:
            draw_utils.plt.cla()
            draw_utils.draw_W(moead)
            draw_utils.draw_MOEAD_Pareto(moead, moead.name + "第：" + str(gen) + "")
            draw_utils.plt.pause(0.0001)
        print('gen %s,EP size :%s' % (gen, len(moead.EP_X_ID)))
    return moead.EP_X_ID
