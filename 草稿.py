import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

'''
 1.函数修改为返回一个列表，直接作为 plot 函数的参数输入
 2.寻找不同 r 值情况下，对曲线有什么影响
 3.A 值是与非线性度相关的，探寻一下，本程序内曲线在不同非线性度的情况下是否与文章的曲线的不同非线性度的情况下的结果相同
 4.
'''

def get_B(Gmax, Gmin, A, r):
    B = (Gmax - Gmin)/(1-math.exp(-r/A))
    return B


def potentiation(P, Gmin, B, A):
    G = B * (1 - math.exp(- P / A)) + Gmin
    return G


def depression(P, Pmax, Gmax, B, A):
    G = -B * (1-math.exp((P - Pmax)/A)) + Gmax
    return G


def potentiation_r(P, Gmin, B, A):
    G = B * (1-math.exp(-r*P/A)) + Gmin
    return G


def depression_r(P, Pmax, Gmax, B, A):
    G = -B * (1-math.exp((P - r * Pmax)/A)) + Gmax
    return G


def Normalization(list_, Max, Min):
    list_Normalized = []
    for each_ in list_:
        list_Normalized.append((each_ - Min) / (Max - Min))
    return list_Normalized


def change_r(x):
    r_list = []
    for each_ in range(r-x, r+x+1):
        r_list.append(each_)
    return r_list


if __name__ == '__main__':

    r = 1.2  # r 是拟合函数的修正系数？
    P = range(1, 16)  # 施加的脉冲个数
    P_max = 15  # 最大脉冲数
    G_max = 80.9  # 电导的最大值
    G_min = 7.5  # 电导的最小值

    A_P = 18.0377  # NL_P = 0.07
    A_D = 0.4945  # NL_D = -2.42

    B_P, B_D = get_B(G_max, G_min, A_P, r), get_B(G_max, G_min, A_P, r)

    NeuroSim_P = []
    NeuroSim_D = []

    P_r = []
    D_r = []

    for each in P:
        NeuroSim_P.append(potentiation(each, G_min, get_B(G_max, G_min, A_P, 1), A_P))
        NeuroSim_D.append(depression(each, P_max, G_max, get_B(G_max, G_min, A_P, 1), A_P))

        P_r.append(potentiation(each, G_min, B_P, A_P))
        D_r.append(depression_r(each, P_max, G_max, B_D, A_P))

    NeuroSim_P_Normalization = Normalization(NeuroSim_P, 700, 0)
    NeuroSim_D_Normalization = Normalization(NeuroSim_D, 0, -700)

    P_r_Normalization = Normalization(P_r, 700, 0)
    D_r_Normalization = Normalization(D_r, 0, -700)

    plt.subplot(2, 2, 1)
    plt.plot(P, NeuroSim_P, linestyle='-', color='green', marker='o', label='NeuroSim_P')
    plt.plot(P, NeuroSim_D, linestyle='-', color='red', marker='^', label='Neurosim_D')
    plt.legend(ncol=1, loc=2)
    plt.xlim([-5, 20])
    my_x_ticks = np.arange(0, 20, 5)
    plt.xticks(my_x_ticks)
    plt.subplot(2, 2, 2)
    plt.plot(P, P_r, linestyle=':', color='blue', marker='o', label='P_r, r=%0.2f' % r)
    plt.plot(P, D_r, linestyle=':', color='orange', marker='o', label='D_r, r=%0.2f' % r)
    plt.legend(ncol=1, loc=2)
    plt.xlim([-5, 20])
    my_x_ticks = np.arange(0, 20, 5)
    plt.xticks(my_x_ticks)

    plt.subplot(2, 2, 3)
    plt.plot(P, NeuroSim_P_Normalization, linestyle='-', color='green', marker='o', label='NeuroSim_P_Normalization')
    plt.plot(P, NeuroSim_D_Normalization, linestyle='-', color='red', marker='^', label='Neurosim_D_Normalization')
    plt.legend(ncol=1, loc=2)
    plt.xlim([-5, 20])
    my_x_ticks = np.arange(0, 20, 5)
    plt.xticks(my_x_ticks)
    plt.subplot(2, 2, 4)
    plt.plot(P, P_r_Normalization, linestyle=':', color='blue', marker='o', label='P_r_Normalization, r=%0.2f' % r)
    plt.plot(P, D_r_Normalization, linestyle=':', color='orange', marker='o', label='D_r_Normalization, r=%0.2f' % r)
    plt.legend(ncol=1, loc=2)
    plt.xlim([-5, 20])
    my_x_ticks = np.arange(0, 20, 5)
    plt.xticks(my_x_ticks)

    plt.show()

    print('\n程序运行结束。')
