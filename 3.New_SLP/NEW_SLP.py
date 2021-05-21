import numpy as np
from pathlib import Path
import struct
import matplotlib.pyplot as plt
import copy


# 电导拟合公式
GP = np.array([0.99,
3.7,
6.1,
8.6,
11.2,
13.6,
16.1,
18.7,
21.1,
23.8,
26.3,
28.7,
31.2,
33.7,
36.1,
38.6,
41.2,
43.6,
46.1,
48.5,
51.2,
53.5,
55.9,
58.4,
61.1,
63.9,
66.5,
69.1,
71.7,
74.5,
77.3,
79.8,
82.6,
85.3,
88.1,
90.9,
93.7,
96.3,
99.1,
101.8,
104.6,
107.1,
109.8,
112.3,
114.9,
117.5,
120.2,
122.7,
125.3,
127.8,
130.4,
133.1,
135.9,
138.5,
141.1,
143.8,
146.6,
149.3,
152.1,
154.8,
157.4,
160.1,
162.7,
165.4,
168.1,
170.1,
172.6,
175.2,
177.9,
180.7,
183.3,
186.1,
188.8,
191.4,
194.1,
196.9,
199.6,
202.4,
205.1,
207.7,
210.3,
212.9,
215.6,
218.3,
220.8,
223.2,
225.6,
227.9,
230.2,
232.4,
234.6,
236.7,
238.8,
240.7,
242.7,
244.5,
246.3,
248.1,
249.7,
251.4
])
GD = np.array([0.4,
1.3,
2.1,
3.1,
4.1,
5.1,
6.1,
6.3,
7.4,
8.6,
9.8,
11.1,
12.3,
13.6,
15,
16.3,
17.8,
19.2,
20.6,
22.2,
23.7,
25.3,
26.9,
28.5,
30.2,
31.9,
33.6,
35.3,
37.1,
38.8,
40.6,
42.4,
44.2,
46.1,
48,
49.9,
51.8,
53.7,
55.6,
57.6,
59.6,
61.7,
63.8,
65.9,
68.1,
70.3,
72.5,
74.8,
77.1,
79.5,
81.9,
84.2,
86.5,
88.8,
91.2,
93.7,
96.2,
98.8,
101.3,
104,
106.5,
109.3,
112,
114.9,
117.8,
120.6,
123.4,
126.2,
129,
131.9,
134.8,
137.6,
140.5,
143.5,
146.3,
149.1,
152,
155,
157.8,
160.7,
163.6,
166.5,
169.5,
172.4,
175.3,
178.2,
181.1,
183.9,
186.8,
189.8,
192.8,
195.9,
199,
202.4,
206.3,
210.8,
216.3,
223.3,
232,
242.9])
GPmax = max(GP)
GPmin = min(GP)
GDmax = max(GD)
GDmin = min(GD)
xP = GP[:-1]
yP = GP[1:]
xD = GD[:-1]
yD = GD[1:]
def delt_GP(xP, aP, bP):
    delt_GP = aP * np.exp(-bP * ((xP - GPmin) / (GPmax - GPmin)))
    return delt_GP
def delt_GD(xD, aD, bD):
    delt_GD = - aD * np.exp(-bD * ((GDmax - xD) / (GDmax - GDmin)))
    return delt_GD


# 激活函数
def tanh(x):
    return np.tanh(x)
def softmax(x):
    exp = np.exp(x-x.max())
    return exp/exp.sum()

# 设置网络基本结构
dimensions = [28*28, 10] # 输入层 28*28， 输出层为 10
activation = [tanh, softmax]
distribution = [
    {'b':[0, 1]},
    {'g+':[0, GPmax], 'g-':[0,GPmax]},
]

# 初始化参数
def init_parameters_g_plus(layer):
    dist = distribution[layer]['g+']
    return np.random.rand(dimensions[layer-1], dimensions[layer]) * (dist[1] - dist[0]) + dist[0]
def init_parameters_g_minus(layer):
    dist = distribution[layer]['g-']
    return np.random.rand(dimensions[layer-1], dimensions[layer]) * (dist[1] - dist[0]) + dist[0]
def init_parameters():
    parameter = []
    for i in range(len(distribution)):
        layer_parameter = {}
        for j in distribution[i].keys():
            if j == 'g+':
                layer_parameter['g+'] = init_parameters_g_plus(i)
                continue
            if j == 'g-':
                layer_parameter['g-'] = init_parameters_g_minus(i)
                continue
        parameter.append(layer_parameter)
    return parameter


# 网络预测公式
def predict(img, parameters):
    l0_in = img
    l0_out = activation[0](l0_in)
    l1_in = np.dot(l0_out, parameters[1]['g+']-parameters[1]['g-'])
    l1_out = activation[1](l1_in)
    return l1_out
# parameters = init_parameters()
# print(predict(np.random.rand(784), parameters).argmax())


# 导入MNIST数据集
train_num = 60000   # 训练集
test_num = 10000    # 测试集
dataset_path = Path('./MNIST')
train_img_path = dataset_path/'train-images-idx3-ubyte'
train_lab_path = dataset_path/'train-labels-idx1-ubyte'
test_img_path = dataset_path/'t10k-images-idx3-ubyte'
test_lab_path = dataset_path/'t10k-labels-idx1-ubyte'
with open(train_img_path, 'rb') as f:
    struct.unpack('>4i', f.read(16))
    tmp_img = np.fromfile(f, dtype=np.uint8).reshape(-1, 28*28)/255
    train_img = tmp_img[:]
with open(test_img_path, 'rb') as f:
    struct.unpack('>4i', f.read(16))
    test_img = np.fromfile(f, dtype=np.uint8).reshape(-1, 28*28)/255
with open(train_lab_path, 'rb') as f:
    struct.unpack('>2i', f.read(8))
    tmp_lab = np.fromfile(f, dtype=np.uint8)
    train_lab = tmp_lab[:]
with open(test_lab_path, 'rb') as f:
    struct.unpack('>2i', f.read(8))
    test_lab = np.fromfile(f, dtype=np.uint8)


def show_train(index):
    plt.imshow(train_img[index].reshape(28, 28), cmap='gray')
    print('label: {}'.format(train_lab[index]))
    plt.pause(1)
def show_test(index):
    plt.imshow(test_img[index].reshape(28, 28), cmap='gray')
    print('label: {}'.format(test_lab[index]))
    plt.pause(1)


def  d_softmax(data):
    sm = softmax(data)
    return np.diag(sm) - np.outer(sm, sm)
# def d_tanh(data):
#     return np.diag(1/(np.cosh(data))**2)
def d_tanh(data):
    return 1/(np.cosh(data))**2


onehot = np.identity(dimensions[-1])


def sqr_loss(img, lab, parameters):
    y_pred = predict(img, parameters)
    y = onehot[lab]
    diff = y - y_pred
    return np.dot(diff, diff)


differential = {softmax:d_softmax, tanh:d_tanh}


def grad_parameters(img, lab, parameters):
    l0_in = img
    l0_out = activation[0](l0_in)
    l1_in = np.dot(l0_out, parameters[1]['g+'] - parameters[1]['g-'])
    l1_out = activation[1](l1_in)

    diff = onehot[lab] - l1_out
    grad_g_plus = np.random.rand(dimensions[0], dimensions[1]) * 0
    grad_g_minus = np.random.rand(dimensions[0], dimensions[1]) * 0

    i = 0
    for each in diff:
        if each > 0:
            grad_g_plus[:, i] = delt_GP(parameters[1]['g+'][:, i], 2.58, 0.06)
            grad_g_minus[:, i] = delt_GD(parameters[1]['g-'][:, i], -7.31, 1.91)

        elif each < 0:
            grad_g_plus[:, i] = delt_GD(parameters[1]['g+'][:, i], -7.31, 1.91)
            grad_g_minus[:, i] = delt_GP(parameters[1]['g-'][:, i], 2.58, 0.06)

        i += 1

    return {'g+': grad_g_plus, 'g-': grad_g_minus}


def count_num(parameters):
    dist_of_num = [
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0},
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0},
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0},
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0},
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0},
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0},
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0},
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0},
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0},
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    ]
    for img_i in range(train_num):
        dist_of_num[train_lab[img_i]][predict(train_img[img_i], parameters).argmax()] += 1
    return dist_of_num


def train_loss(parameters):
    loss_accu = 0
    for img_i in range(train_num):
        loss_accu += sqr_loss(train_img[img_i], train_lab[img_i], parameters)
    return loss_accu / (train_num / 10000)


def train_accuracy(parameters):
    correct = [predict(train_img[img_i], parameters).argmax() == train_lab[img_i] for img_i in range(train_num)]
    return correct.count(True) / len(correct)


def test_loss(parameters):
    loss_accu = 0
    for img_i in range(test_num):
        loss_accu += sqr_loss(test_img[img_i], test_lab[img_i], parameters)
    return loss_accu / (test_num / 10000)


def test_accuracy(parameters):
    correct = [predict(test_img[img_i], parameters).argmax() == test_lab[img_i] for img_i in range(test_num)]
    return correct.count(True) / len(correct)


# print(valid_loss(parameters))
# print(valid_accuracy(init_parameters()))

batch_size = 100

def train_batch(current_batch, parameters):
    grad_accu = grad_parameters(train_img[current_batch * batch_size + 0], train_lab[current_batch * batch_size + 0],
                                parameters)
    for img_i in range(1, batch_size):
        grad_tmp = grad_parameters(train_img[current_batch * batch_size + img_i],
                                   train_lab[current_batch * batch_size + img_i], parameters)
        for key in grad_accu.keys():
            grad_accu[key] += grad_tmp[key]
    for key in grad_accu.keys():
        grad_accu[key] /= batch_size
    return grad_accu

def check_G(parameters):
    parameter_tmp = copy.deepcopy(parameters)

    G_minux_max = np.where(parameter_tmp[1]['g-'] > 1)
    G_plus_max = np.where(parameter_tmp[1]['g-'] > 1)
    parameter_tmp[1]['g-'][G_minux_max] = 0
    parameter_tmp[1]['g+'][G_plus_max] = 0

    return parameter_tmp


def combine_parameters(parameters, grad):
    parameter_tmp = copy.deepcopy(parameters)
    parameter_tmp[1]['g-'] += grad['g-']
    parameter_tmp[1]['g+'] += grad['g+']
    parameter_tmp = check_G(parameter_tmp)
    return parameter_tmp


parameters = init_parameters()
current_epoch = 0
train_loss_list = []
test_loss_list = []
train_accu_list = []
test_accu_list = []

save_path = Path('./TEST/返修/')
learn_rate = 0.01
epoch_num = 10
for epoch in range(epoch_num):
    current_epoch += 1
    print('Now running epoch %d/%d' % (current_epoch, epoch_num))
    for i in range(int(train_num / batch_size)):
        # print(f'running batch {i + 1}/{int(train_num / batch_size)}')
        if i % 100 == 99:
            print('running batch {}/{}'.format(i + 1, train_num / batch_size))
        grad_tmp = train_batch(i, parameters)
        parameters = combine_parameters(parameters, grad_tmp)


    train_loss_list.append(train_loss(parameters))
    train_accu_list.append(train_accuracy(parameters))
    print(train_accu_list[-1])


lower = 0
plt.plot(train_loss_list[lower:], color='red', label='train loss', marker='>')
plt.show()

plt.plot(train_accu_list[lower:], color='red', label='train accuracy', marker='>')
plt.show()



