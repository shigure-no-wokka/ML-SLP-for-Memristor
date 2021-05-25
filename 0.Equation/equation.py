import matplotlib.pyplot as plt
import math


# parameters = [
#     {'a+':1e-2, 'b+':3.0},
#     {'a-':5e-3, 'b-':3.0},
#     {'wmax': 1, 'wmin':1e-4}
# ]
parameters = [
    {'a+':2.61, 'b+':0.07},
    {'a-':6.35, 'b-':1.68},
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


if __name__ == '__main__':


    count_cycle = 0
    count_set = input('请输入需要的电导数据个数：')


    while True:

        weight_plus_list.append(weight_plus(weight_plus_list[-1]))
        weight_minus_list.append(weight_minus(weight_minus_list[-1]))

        diff = weight_minus_list[-1] - weight_plus_list[-1]
        count_cycle += 1

        if count_cycle == int(count_set):
            break


    print(weight_plus_list)
    print(weight_minus_list)

    draw_graph_minus(weight_minus_list)
    draw_graph_plus(weight_plus_list)
    plt.show()