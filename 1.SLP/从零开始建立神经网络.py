import numpy as np
from pathlib import Path
import struct
import matplotlib.pyplot as plt
import copy

# 电导数据
GP = np.array([0.99,3.7,6.1,8.6,11.2,13.6,16.1,18.7,21.1,23.8,26.3,28.7,31.2,33.7,36.1,38.6,41.2,43.6,46.1,48.5,
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
114.6,
116.8,
118.9,
120.8,122.6,124.3]) # 激活部分的电导
GD = np.array([0.5,
0.9,
1.3,
1.7,
2.2,
2.7,
3.2,
3.8,
4.4,
5.1,
5.8,
6.6,
7.4,
8.3,
9.2,
10.3,
11.4,
12.6,
13.8,
15.2,
16.8,
18.5,
20.3,
22.2,
24.3,
26.4,
28.6,
30.9,
33.3,
35.8,
38.4,
41.1,
43.9,
46.8,
49.9,
53.1,
56.2,
59.4,
62.7,
66.1,
69.6,
73.2,
76.9,
80.7,
84.6,
88.7,
92.8,
97.1,
103.5,
110.2]) # 抑制部分的电导
GPmax = max(GP)
GPmin = min(GP)
GDmax = max(GD)
GDmin = min(GD)
GP_new = np.array([(2*each - GPmin - GPmax) / (GPmax - GPmin) for each in GP])  # 归一化电导数据
GD_new = np.array([(2*each - GDmin - GDmax) / (GDmax - GDmin) for each in GD])  # 归一化电导数据

# 定义激活函数
def tanh(x):
    return np.tanh(x)
def softmax(x):
    exp = np.exp(x-x.max())
    return exp/exp.sum()

dimensions = [28*28, 10] # 输入层 28*28， 输出层为 10
activation = [tanh, softmax] # 激活函数
distribution = [
    {'b':[0,0]},
    {'b':[0,0], 'w':[-1, +1]},
]   # 定义网络参数范围

# 初始化参数，即初始化权重与偏执
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

# 输入图片并预测结果
def predict(img, parameters):
    l0_in = img + parameters[0]['b']    # 对输入图片加上偏执
    l0_out = activation[0](l0_in)   # 输入层的激活函数 tanh
    l1_in = np.dot(l0_out, parameters[1]['w']+parameters[1]['b'])   # 输出层的输入，与权重点乘再加上输出层偏执
    l1_out = activation[1](l1_in)   # 输出层的激活函数 softmax
    return l1_out

# 导入MNIST数据集，同一目录下
dataset_path = Path('./MNIST')
train_img_path = dataset_path/'train-images-idx3-ubyte'
train_lab_path = dataset_path/'train-labels-idx1-ubyte'
test_img_path = dataset_path/'t10k-images-idx3-ubyte'
test_lab_path = dataset_path/'t10k-labels-idx1-ubyte'

train_num = 60000   # 训练集
test_num = 10000    # 测试集

# 打开训练集与测试集，并reshape为1*784的一维向量
with open(train_img_path, 'rb') as f:
    struct.unpack('>4i', f.read(16))
    train_img = np.fromfile(f, dtype=np.uint8).reshape(-1, 28*28)/255
with open(test_img_path, 'rb') as f:
    struct.unpack('>4i', f.read(16))
    test_img = np.fromfile(f, dtype=np.uint8).reshape(-1, 28*28)/255
with open(train_lab_path, 'rb') as f:
    struct.unpack('>2i', f.read(8))
    train_lab = np.fromfile(f, dtype=np.uint8)
with open(test_lab_path, 'rb') as f:
    struct.unpack('>2i', f.read(8))
    test_lab = np.fromfile(f, dtype=np.uint8)

# 查看MNIST的图片
def show_train(index):
    plt.imshow(train_img[index].reshape(28, 28), cmap='gray')
    print('label: {}'.format(train_lab[index]))
    plt.pause(1)
def show_test(index):
    plt.imshow(test_img[index].reshape(28, 28), cmap='gray')
    print('label: {}'.format(test_lab[index]))
    plt.pause(1)

# 激活函数的梯度
def  d_softmax(data):
    sm = softmax(data)
    return np.diag(sm) - np.outer(sm, sm)
def d_tanh(data):
    return 1/(np.cosh(data))**2
onehot = np.identity(dimensions[-1])

# 损失函数
def sqr_loss(img, lab, parameters):
    y_pred = predict(img, parameters) # 预测值
    y = onehot[lab] # 实际值
    diff = y - y_pred
    return np.dot(diff, diff) # 预测值与实际值的差的平方和
differential = {softmax:d_softmax, tanh:d_tanh}

# 梯度下降，损失函数对权重与偏执求导，并返回三个参数的梯度
def grad_parameters(img, lab, parameters):
    l0_in = img + parameters[0]['b']
    l0_out = activation[0](l0_in)
    l1_in = np.dot(l0_out, parameters[1]['w']+parameters[1]['b'])
    l1_out = activation[1](l1_in)

    diff = onehot[lab] - l1_out
    act1 = np.dot(differential[activation[1]](l1_in), diff)

    grad_b1 = -2 * act1
    grad_w1 = -2 * np.outer(l0_out, act1)
    grad_b0 = -2 * differential[activation[0]](l0_in) * np.dot(parameters[1]['w'] ,act1)

    return {'w1': grad_w1, 'b1':grad_b1, 'b0':grad_b0}

# 训练集与测试集的loss与accuracy
def train_loss(parameters):
    loss_accu = 0
    for img_i in range(train_num):
        loss_accu += sqr_loss(train_img[img_i], train_lab[img_i], parameters)
    return loss_accu/(train_num/10000)
def train_accuracy(parameters):
    correct = [predict(train_img[img_i], parameters).argmax()==train_lab[img_i] for img_i in range(train_num)]
    return correct.count(True)/len(correct)
def test_loss(parameters):
    loss_accu = 0
    for img_i in range(test_num):
        loss_accu += sqr_loss(test_img[img_i], test_lab[img_i], parameters)
    return loss_accu/(test_num/10000)
def test_accuracy(parameters):
    correct = [predict(test_img[img_i], parameters).argmax()==test_lab[img_i] for img_i in range(test_num)]
    return correct.count(True)/len(correct)

# 每回合batch的训练，这里有点忘了
batch_size = 100
def train_batch(current_batch, parameters):
    grad_accu = grad_parameters(train_img[current_batch*batch_size+0], train_lab[current_batch*batch_size+0], parameters)
    for img_i in range(1, batch_size):
        grad_tmp = grad_parameters(train_img[current_batch*batch_size+img_i], train_lab[current_batch*batch_size+img_i], parameters)
        for key in grad_accu.keys():
            grad_accu[key] += grad_tmp[key]
    for key in grad_accu.keys():
        grad_accu[key] /= batch_size
    return grad_accu

# 从电导数据中寻找最接近的数据
def findGP_num(data, find_array=GP_new):
    data_array = np.array([data]*len(find_array))
    diff = abs(data_array - find_array)
    return find_array[np.where(diff == np.min(diff))]
def findGD_num(data, find_array=GD_new):
    data_array = np.array([data] * len(find_array))
    diff = abs(data_array - find_array)
    return find_array[np.where(diff == np.min(diff))]

# 替换权重矩阵（比较笨，只会一个一个的换），最后返回一个替换好的矩阵
def each_change(matrix_old, gradw1):
    matrix_new = np.zeros(matrix_old.shape)
    for i in range(matrix_old.shape[0]):
        for j in range(matrix_old.shape[1]):
            if gradw1[i][j] < 0:
                matrix_new[i][j] = findGP_num(matrix_old[i][j])
            else:
                matrix_new[i][j] = findGD_num(matrix_old[i][j])
    return matrix_new

# 更新参数
def combine_parameters(parameters, grad, learn_rate):
    parameter_tmp = copy.deepcopy(parameters)
    parameter_tmp[0]['b'] -= learn_rate * grad['b0']    # 更新输入层的偏执
    parameter_tmp[1]['b'] -= learn_rate * grad['b1']    # 更新输出层的偏执
    parameter_tmp[1]['w'] -= learn_rate * grad['w1']    # 更新权重
    # parameter_tmp[1]['w'] = each_change(parameter_tmp[1]['w'], grad['w1'])  # 用电导数据中最为接近的数字替换
    return parameter_tmp

# 保存准确率文件
def text_save(filename, data):#filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存文件成功")

parameters = init_parameters()
current_epoch = 0
train_loss_list = []
test_loss_list = []
train_accu_list = []
test_accu_list = []

# 开始训练网络
learn_rate = 0.01
epoch_num = 20
for epoch in range(epoch_num):
    current_epoch += 1
    print('Now running epoch %d/%d' %(current_epoch, epoch_num))
    for i in range(int(train_num/batch_size)):
        if i % 100 == 99:
            print('running batch {}/{}'.format(i+1, train_num/batch_size))
        grad_tmp = train_batch(i, parameters)
        parameters = combine_parameters(parameters, grad_tmp, learn_rate)

    # 储存准确率与损失
    train_loss_list.append(train_loss(parameters))
    train_accu_list.append(train_accuracy(parameters))
    test_loss_list.append(test_loss(parameters))
    test_accu_list.append(test_accuracy(parameters))

    while epoch == epoch_num-1:
        w = parameters[1]['w']
        for i in range(10):
            plt.imshow(w[:, i].reshape(28, 28)*255, cmap='gray')
            plt.show()
            text_save('./权重的十个特征矩阵/权重特征矩阵 %d epoch%d lr%d' % (i, epoch_num, learn_rate), w[:, i])

# 输出权重的10个列向量，再将每个列向量reshape为28*28的矩阵。
# 按理说这个矩阵应该就各自代表了一个数字的特征矩阵。但是我输出后都是很乱的结果。
# w = parameters[1]['w']


# 画出准确率与损失图像，并保存为文件
lower = 0
plt.plot(test_loss_list[lower:], color='black', label='test loss', marker='o')
plt.plot(train_loss_list[lower:], color='red', label='train loss', marker='>')
plt.show()

plt.plot(test_accu_list[lower:], color='black', label='test accuracy', marker='o')
plt.plot(train_accu_list[lower:], color='red', label='train accuracy', marker='>')
plt.show()

# train_filename = 'S:\WorkPlace\Pycharm2020\机器学习\From0/50准确率保存\添加GD部分数据后/train_accu epoch=%d batch=%d learn_rate=%0.2e.txt'%(epoch_num, batch_size, learn_rate)
# test_filename = 'S:\WorkPlace\Pycharm2020\机器学习\From0/50准确率保存\添加GD部分数据后/test_accu epoch=%d batch=%d learn_rate=%0.2e.txt'%(epoch_num, batch_size, learn_rate)
# text_save(train_filename, train_accu_list)
# text_save(test_filename, test_accu_list)
