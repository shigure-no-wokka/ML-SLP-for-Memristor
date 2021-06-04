import matplotlib.pyplot as plt
import math


# parameters = [
#     {'a+':1e-2, 'b+':3.0},
#     {'a-':5e-3, 'b-':3.0},
#     {'wmax': 1, 'wmin':1e-4}
# ]
parameters = [
    {'a+':1e-2, 'b+':3},
    {'a-':1e-2, 'b-':3},
    {'wmax': 1, 'wmin':1e-4}
]
weight_plus_list = [parameters[2]['wmin']]
weight_minus_list = [parameters[2]['wmax']]


def weight_plus(w):
    delta_w = parameters[0]['a+'] * math.exp(-parameters[0]['b+']*(w-parameters[2]['wmin'])/(parameters[2][
                                                                                                 'wmax']-parameters[2][
        'wmin']))
    return w+delta_w
def weight_minus(w):
    delta_w = parameters[1]['a-'] * math.exp(-parameters[1]['b-']*(parameters[2]['wmax']-w)/(parameters[2][
                                                                                                 'wmax']-parameters[2]['wmin']))
    return w-delta_w


def draw_graph_plus(weight):
    plt.plot(weight, color='red', label='weight_minus_list', marker='>')
    return 0
def draw_graph_minus(weight):
    plt.plot(weight, color='black', label='weight_plus_list', marker='o')
    return 0
def draw_graph_both(weight_plus, weight_minus):
    weight = weight_plus + weight_minus
    plt.plot(weight, color='blue', label='weight', marker='o')
    return 0

def text_save(filename, data):  # filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename, 'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace('{', '').replace('}', '')
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存文件成功")

def equation():

    count_num = 0
    while True:

        weight_plus_list.append(weight_plus(weight_plus_list[-1]))
        weight_minus_list.append(weight_minus(weight_minus_list[-1]))
        count_num += 1
        if weight_plus_list[-1] >= 1:
            break
        elif count_num == 100:
            break
        else:
            continue

    # draw_graph_both(weight_plus_list, weight_minus_list)


    wpl = [(each - weight_plus_list[0]) / (weight_plus_list[-1]-weight_plus_list[0]) for each in weight_plus_list]
    wml = [(each - weight_minus_list[-1]) / (weight_minus_list[0]-weight_minus_list[-1]) for each in weight_minus_list]

    # draw_graph_both(wpl, wml)
    # plt.show()
    # plt.savefig(f"Nonlinear={parameters[0]['b+']}.png", dpi=720)

    # text_save(f"Nonlinear={parameters[0]['b+']}.txt", wpl+wml)

    return wpl, wml


