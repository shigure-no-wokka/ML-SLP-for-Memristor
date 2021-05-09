import numpy as np
from pathlib import Path
import struct
import matplotlib.pyplot as plt
import copy
import math
import datetime

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

GP_new = np.array([(2*each - GPmin - GPmax) / (GPmax - GPmin) for each in GP])
GD_new = np.array([(2*each - GDmin - GDmax) / (GDmax - GDmin) for each in GD])

xP = GP[:-1]
yP = GP[1:]

xD = GD[:-1]
yD = GD[1:]

# 定义激活函数
def tanh(x):
    return np.tanh(x)
def softmax(x):
    exp = np.exp(x-x.max())
    return exp/exp.sum()

dimensions = [28*28, 20*20,10] # 输入层 784, 隐藏层 400, 输出层 10
activation = [tanh, tanh, softmax]
distribution = [
    {'b':[0,0]},
    {'b':[0,0], 'w':[-1, +1]},
    {'b':[0,0], 'w':[-1, +1]},
]

# 初始化参数
def init_parameters_b(layer):
    dist = distribution[layer]['b']
    return np.random.rand(dimensions[layer])*(dist[1]-dist[0])+dist[0]
def init_parameters_w(layer):
    dist = distribution[layer]['w']
    return np.random.rand(dimensions[layer-1], dimensions[layer]) * (dist[1] - dist[0]) + dist[0]
def init_parameters():
    parameter = []
    for i in range(len(distribution)):
        layer_parameter = {}
        for j in distribution[i].keys():
            if j == 'b':
                layer_parameter['b'] = init_parameters_b(i)
                continue
            if j == 'w':
                layer_parameter['w'] = init_parameters_w(i)
                continue
        parameter.append(layer_parameter)
    return parameter

# parameters = init_parameters()

def predict(img, parameters):
    l0_in = img + parameters[0]['b']
    l0_out = activation[0](l0_in)
    l1_in = np.dot(l0_out, parameters[1]['w'])+parameters[1]['b']
    l1_out = activation[1](l1_in)
    l2_in = np.dot(l1_out, parameters[2]['w'])+parameters[2]['b']
    l2_out = activation[2](l2_in)
    return l2_out
# print(predict(np.random.rand(784), parameters).argmax())
dataset_path = Path('./MNIST')
train_img_path = dataset_path/'train-images-idx3-ubyte'
train_lab_path = dataset_path/'train-labels-idx1-ubyte'
test_img_path = dataset_path/'t10k-images-idx3-ubyte'
test_lab_path = dataset_path/'t10k-labels-idx1-ubyte'

# train_f = open(train_img_path, 'rb')
# struct.unpack('>4i', train_f.read(16))
# print(np.fromfile(train_f, dtype=np.uint8).reshape(-1, 28*28))
train_num = 60000   # 训练集
valid_num = 0   # 验证集
test_num = 10000    # 测试集

with open(train_img_path, 'rb') as f:
    struct.unpack('>4i', f.read(16))
    tmp_img = np.fromfile(f, dtype=np.uint8).reshape(-1, 28*28)/255
    train_img = tmp_img[:]
    # valid_img = tmp_img[train_num:]
with open(test_img_path, 'rb') as f:
    struct.unpack('>4i', f.read(16))
    test_img = np.fromfile(f, dtype=np.uint8).reshape(-1, 28*28)/255
with open(train_lab_path, 'rb') as f:
    struct.unpack('>2i', f.read(8))
    tmp_lab = np.fromfile(f, dtype=np.uint8)
    train_lab = tmp_lab[:]
    # valid_lab = tmp_lab[train_num:]
with open(test_lab_path, 'rb') as f:
    struct.unpack('>2i', f.read(8))
    test_lab = np.fromfile(f, dtype=np.uint8)

def show_train(index):
    plt.imshow(train_img[index].reshape(28, 28), cmap='gray')
    print('label: {}'.format(train_lab[index]))
    plt.pause(1)
# def show_valid(index):
#     plt.imshow(valid_img[index].reshape(28, 28), cmap='gray')
#     print('label: {}'.format(valid_lab[index]))
#     plt.pause(1)
def show_test(index):
    plt.imshow(test_img[index].reshape(28, 28), cmap='gray')
    print('label: {}'.format(test_lab[index]))
    plt.pause(1)
# print(train_img[0].reshape(28, 28))
# img = train_img[0].reshape(28, 28)
# plt.imshow(img, cmap='gray')
# plt.pause(0.5)

# show_train(np.random.randint(train_num))
# show_valid(np.random.randint(valid_num))
# show_test(np.random.randint(test_num))

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
    l0_in = img + parameters[0]['b']
    l0_out = activation[0](l0_in)
    l1_in = np.dot(l0_out, parameters[1]['w']) + parameters[1]['b']
    l1_out = activation[1](l1_in)
    l2_in = np.dot(l1_out, parameters[2]['w']) + parameters[2]['b']
    l2_out = activation[2](l2_in)

    diff = onehot[lab] - l2_out
    act1 = np.dot(differential[activation[1]](l1_in), diff)
    act2 = np.dot(differential[activation[2]](l2_in), diff)

    act3 = np.outer(l0_out, differential[activation[2]](l2_in))
    act4 = np.dot(parameters[2]['w'], act3)

    grad_b2 = -2 * act2
    grad_w2 = -2 * np.outer(l1_out, act2)
    grad_b1 = -2 * differential[activation[1]](l1_in) * np.dot(parameters[2]['w'], act2)
    grad_w1 = -2 * np.outer(act4, act2)
    grad_b0 = -2 * differential[activation[0]](l0_in) * np.dot(parameters[1]['w'], differential[activation[1]](l1_in)) * np.dot(parameters[2]['w'], act2)

    return {'w2': grad_w2, 'b2': grad_b2, 'w1': grad_w1, 'b1':grad_b1, 'b0':grad_b0}


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


# print(train_batch(0, init_parameters()))

def findGP_num(data, find_array=GP_new):
    data_array = np.array([data] * len(find_array))
    diff = abs(data_array - find_array)
    return find_array[np.where(diff == np.min(diff))]


def findGD_num(data, find_array=GD_new):
    data_array = np.array([data] * len(find_array))
    diff = abs(data_array - find_array)
    return find_array[np.where(diff == np.min(diff))]


def each_change(matrix_old, gradw1):
    matrix_new = np.zeros(matrix_old.shape)
    for i in range(matrix_old.shape[0]):
        for j in range(matrix_old.shape[1]):
            if gradw1[i][j] < 0:
                matrix_new[i][j] = findGP_num(matrix_old[i][j])
            else:
                matrix_new[i][j] = findGD_num(matrix_old[i][j])
    return matrix_new


def combine_parameters(parameters, grad, learn_rate):
    parameter_tmp = copy.deepcopy(parameters)
    parameter_tmp[0]['b'] -= learn_rate * grad['b0']
    parameter_tmp[1]['b'] -= learn_rate * grad['b1']
    parameter_tmp[1]['w'] -= learn_rate * grad['w1']
    parameter_tmp[1]['w'] = each_change(parameter_tmp[1]['w'], grad['w1'])
    parameter_tmp[2]['b'] -= learn_rate * grad['b2']
    parameter_tmp[2]['w'] -= learn_rate * grad['w2']
    parameter_tmp[2]['w'] = each_change(parameter_tmp[2]['w'], grad['w1'])
    return parameter_tmp


# print(combine_parameters(parameters, train_batch(0, parameters), 1))


def text_save(filename, data):  # filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename, 'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace('{', '').replace('}', '')
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存文件成功")


parameters = init_parameters()
current_epoch = 0
# train_loss_list = []
# test_loss_list = []
train_accu_list = []
# test_accu_list = []
# print(valid_accuracy(parameters))

# 进度条
# from tqdm import tqdm_notebook

learn_rate = 0.01
epoch_num = 10
for epoch in range(epoch_num):
    current_epoch += 1
    print('Now running epoch %d/%d' % (current_epoch, epoch_num))
    for i in range(int(train_num / batch_size)):
        if i % 100 == 99:
            print('running batch {}/{}'.format(i + 1, train_num / batch_size))
        grad_tmp = train_batch(i, parameters)
        parameters = combine_parameters(parameters, grad_tmp, learn_rate)
    if current_epoch == 10:
        dist_of_num = count_num(parameters)
        text_save('./TEST/100/epoch=%d' % current_epoch, dist_of_num)

    # train_loss_list.append(train_loss(parameters))
    train_accu_list.append(train_accuracy(parameters))
    # test_loss_list.append(test_loss(parameters))
    # test_accu_list.append(test_accuracy(parameters))

lower = 0
# plt.plot(test_loss_list[lower:], color='black', label='test loss', marker='o')
# plt.plot(train_loss_list[lower:], color='red', label='train loss', marker='>')
# plt.show()
# plt.plot(test_accu_list[lower:], color='black', label='test accuracy', marker='o')
plt.plot(train_accu_list[lower:], color='red', label='train accuracy', marker='>')
plt.show()

train_filename = './TEST/100/train_accu epoch=%d batch=%d learn_rate=%0.2e.txt' % (epoch_num, batch_size, learn_rate)
test_filename = './TEST/100/test_accu epoch=%d batch=%d learn_rate=%0.2e.txt' % (epoch_num, batch_size, learn_rate)
text_save(train_filename, train_accu_list)
# text_save(test_filename, test_accu_list)
