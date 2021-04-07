from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):

        # 随机数发生器种子，以保证每次获得相同结果
        random.seed(1)

        # 对单个神经元建模，含有 3 个输入连接和一个输出连接
        # 对一个 3 X 1 的矩阵赋予随机权重值，范围 -1~1，平均值为0
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    # Sigmoid 函数， S形曲线
    # 用这个函数对输入的加权总和做正规化，使其范围在 0~1
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # 神经网络-思考
    def predict(self, inputs):
        # 把输入传递给神经网络
        return self.__sigmoid(dot(inputs, self.synaptic_weights))

    # Sigmoid 函数的导数
    # Sigmoid 函数的梯度
    # 表示我们对当前权重的置信程度
    def __sigmoid_derivative(self, x):
        return x * (x - 1)

    # 通过试错过程训练神经网络
    # 每次都调整突出权重
    def train(self, trainingSetInputs, trainingSetOutputs, numberOfIterations):

        for iteration in range(numberOfIterations):
            # 将训练集导入神经网络
            output = self.predict(trainingSetInputs)

            # 计算误差（实际值与期望值之差）
            error = trainingSetOutputs - output

            # 将误差、输入和 S 曲线梯度相乘
            # 对于置信程度低的权重，调整程度也大
            # 为 0 的输入值不会影响权重
            adjustment = dot(trainingSetInputs.T, error *
                             self.__sigmoid_derivative(output))

            # 调整权重
            self.synaptic_weights -= adjustment

if __name__ == '__main__':

    # 初始化神经网络
    neuralNetwork = NeuralNetwork()

    print('random starting synaptic weights')
    print(neuralNetwork.synaptic_weights)

    # 训练集，每个有3个输入和一个输出
    trainingSetInputs = array([[0, 0, 0],
                               [0, 0, 1], [0, 1, 0], [1, 0, 0],
                               [0, 1, 1], [1, 0, 1], [1, 1, 0],
                               [1, 1, 1]
                               ])
    trainingSetOutputs = array([[0,
                                 0, 1, 0,
                                 1, 0, 1,
                                 1]
                                 ]).T

    # 用训练集训练神经网络
    # 重复一万次，每次做微小的调整
    neuralNetwork.train(trainingSetInputs, trainingSetOutputs, 10000)

    print('new wheights')
    print(neuralNetwork.synaptic_weights)

    # 用新数据测试神经网络
    print("testing")
    print(neuralNetwork.predict(array([0, 0, 1])))
