from keras.utils import to_categorical
from keras import models, layers, regularizers, initializers
from keras.optimizers import RMSprop
from keras.datasets import mnist
from keras import backend as K
import math
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Dense, Activation, Dropout

# # 使用 cpu 训练，关闭 gpu 的使用
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 加载数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 将数据铺开成一维向量，长度为 28*28
train_images = train_images.reshape((60000, 28*28)).astype('float')
test_images = test_images.reshape((10000, 28*28)).astype('float')
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 过拟合：一个假设在训练数据上能够获得比其他假设更好的拟合， 但是在训练数据外的数据集上却不能很好地拟合数据，
# 此时认为这个假设出现了过拟合的现象。出现这种现象的主要原因是训练数据中存在噪音或者训练数据太少。

# 自定义初始化器
GP = np.array([7.5, 12.7, 18, 23.3, 28.6, 33.8, 39, 44.3, 49.6, 54.8, 60, 65.2, 70.4, 75.7, 80.9])


# 神经网络模型
network = models.Sequential()

network.add(layers.Dense(units=784, activation='relu', input_shape=(28*28, ),
                         kernel_initializer='RandomNormal', kernel_regularizer=regularizers.l1(0.0001)))  # 输入层 units 个神经元，这里换层书写的部分为正则化防止过拟合
network.add(layers.Dropout(0.01))   # 每次让神经元有 0.01% 的概率丧失功能 避免过拟合
network.add(layers.Dense(units=10, activation='softmax'))   # 输出层 10个神经元，对应 10个图片特征

# 神经网络训练
# 编译：确定优化器和损失函数
network.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# 训练网络：确定训练的数据集，训练的轮数和每次训练的样本数等
# 训练网络用fit函数，epochs 表示训练多少个回合， batch_size 表示每次训练给多大的数据
history = network.fit(train_images, train_labels, validation_split=0.25, epochs=40, batch_size=128, verbose=2)

# keras 的summary 来可视化结构
# print(network.summary())

# my_models_weights = network.get_weights()
# print(my_models_weights)
# # 保存权重
# network.save_weights('S:/Test/my_model_weights.h5')

# 用训练好的模型进行预测，并在测试集上作出评价
y_pre = network.predict(test_images[:])    # 测试集
print(y_pre, test_labels[:])   # 与实际结果进行对比
test_loss, test_accuracy = network.evaluate(test_images, test_labels)
print('test_loss:', test_loss, '    test_accuracy:', test_accuracy)

# plt.imshow(test_images[0])
# plt.savefig('S:/WorkPlace/Pycharm2020/机器学习/Test/' + 'figure epoch20' + '.png', dpi=720)
# plt.show()
# plt.close()

# 绘制训练 & 验证的准确率值
plt.plot(history.history['accuracy']) # Train
plt.plot(history.history['val_accuracy']) # Test
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
my_x_ticks = np.arange(0, 45, 5)
plt.xticks(my_x_ticks)
plt.ylim([0, 1])
my_y_ticks = np.arange(0, 1.2, 0.2)
plt.yticks(my_y_ticks)
plt.legend(['Train', 'Test'], loc='center right')
# plt.savefig('S:/WorkPlace/Pycharm2020/机器学习/Test/' + 'acc epoch20' + '.png', dpi=720)
plt.show()
plt.close()

# 绘制训练 & 验证的损失值
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.ylim([0, 1])
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

print('\n程序运行结束。')
