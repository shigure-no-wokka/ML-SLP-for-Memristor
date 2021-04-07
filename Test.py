import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pimg
import seaborn as sb  # 一个构建在matplotlib上的绘画模块，支持numpy,pandas等数据结构

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix  # 混淆矩阵

import itertools
#  keras
from keras.utils import to_categorical  # 数字标签转化成one-hot编码
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.datasets import mnist


# 设置绘画风格
sb.set(style='white', context='notebook', palette='deep')

# 加载数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# train_x = train_data.drop(labels=['label'],axis=1)  # 去掉标签列
train_x = train_images.iloc[:, 1:]
train_y = train_images.iloc[:, 0]
del train_images  # 释放一下内存

# 观察一下训练数据的分布情况
g = sb.countplot(train_y)
train_y.value_counts()

train_x.isnull().describe()  # 检查是否存在确实值

train_x.isnull().any().describe()

# 归一化
train_x = train_x / 255.0
test_x = test_images / 255.0

del test_images

# reshape trian_x, test_x
# train_x = train_x.values.reshape(-1, 28, 28, 1)
# test_x = test_x.values.reshape(-1, 28, 28, 1)
train_x = train_x.as_matrix().reshape(-1, 28, 28, 1)
test_x = test_x.as_matrix().reshape(-1, 28, 28, 1)

# 吧标签列转化为one-hot 编码格式
train_y = to_categorical(train_y, num_classes=10)


#从训练数据中分出十分之一的数据作为验证数据
random_seed = 3
train_x , val_x , train_y, val_y = train_test_split(train_x, train_y, test_size=0.1, random_state=random_seed)


plt.imshow(train_x[0][:,:,0])


model = Sequential()
# 第一个卷积层，32个卷积核，大小５x5，卷积模式SAME,激活函数relu,输入张量的大小
model.add(Conv2D(filters= 32, kernel_size=(5,5), padding='Same', activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(filters= 32, kernel_size=(5,5), padding='Same', activation='relu'))
# 池化层,池化核大小２x2
model.add(MaxPool2D(pool_size=(2,2)))
# 随机丢弃四分之一的网络连接，防止过拟合
model.add(Dropout(0.25))
model.add(Conv2D(filters= 64, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(Conv2D(filters= 64, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
# 全连接层,展开操作，
model.add(Flatten())
# 添加隐藏层神经元的数量和激活函数
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
# 输出层
model.add(Dense(10, activation='softmax'))

# 设置优化器
# lr :学习效率，　decay :lr的衰减值
optimizer = RMSprop(lr=0.001, decay=0.0)

# 编译模型
# loss:损失函数，metrics：对应性能评估函数
model.compile(optimizer=optimizer, loss='categorical_crossentropy',
              metrics=['accuracy'])

# keras的callback类提供了可以跟踪目标值，和动态调整学习效率
# moitor : 要监测的量，这里是验证准确率
# matience: 当经过３轮的迭代，监测的目标量，仍没有变化，就会调整学习效率
# verbose : 信息展示模式，去０或１
# factor :　每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
# mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值触发学习率减少。在max模式下，当检测值不再上升则触发学习率减少。
# epsilon：阈值，用来确定是否进入检测值的“平原区”
# cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作
# min_lr：学习率的下限
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3,
                                            verbose=1, factor=0.5, min_lr=0.00001)

epochs = 40
batch_size = 100


# 数据增强处理，提升模型的泛化能力，也可以有效的避免模型的过拟合
# rotation_range : 旋转的角度
# zoom_range : 随机缩放图像
# width_shift_range : 水平移动占图像宽度的比例
# height_shift_range
# horizontal_filp : 水平反转
# vertical_filp : 纵轴方向上反转
data_augment = ImageDataGenerator(rotation_range= 10,zoom_range= 0.1,
                                  width_shift_range = 0.1,height_shift_range = 0.1,
                                  horizontal_flip = False, vertical_flip = False)


history = model.fit_generator(data_augment.flow(train_x, train_y, batch_size=batch_size),
                             epochs= epochs, validation_data = (val_x,val_y),
                             verbose =2, steps_per_epoch=train_x.shape[0]//batch_size,
                             callbacks=[learning_rate_reduction])