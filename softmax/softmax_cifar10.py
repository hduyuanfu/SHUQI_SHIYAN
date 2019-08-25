import math
import numpy as np
import random
import gzip                              # 解压数据集
import pickle                            # 将解压后的数据集划分为三部分
import time
from data_utils import load_CIFAR10

class Softmax(object):  

    def __init__(self):
        self.learning_step = 0.001    # 学习速率
        self.max_iteration = 50000      # 最大迭代次数
        # self.weight_lambda = 0.01       # 衰退权重


    def compute_wx(self, x, l):
        '''计算单个样本x与第l组权重(参数)的乘积'''

        theta_l = self.w[l]
        product = np.dot(theta_l, x)
        return math.exp(product)
        '''try:
            return math.exp(product)
        except OverflowError:
            ans = float('inf')'''


    def compute_probability(self, x, j):
        '''计算样本x属于类别j的概率'''

        # molecule分子；denominator分母
        molecule = self.compute_wx(x, j)
        denominator = sum([self.compute_wx(x, i) for i in range(self.k)])
        return 1.0 * molecule / denominator


    def compute_partial_derivative(self, x, y, j):
        '''计算代价函数关于第j组权重(参数)的偏导和梯度'''

        # 计算示性函数。int(y == j)中，若y = j,则返回1，否则返回0
        first = int(y == j)
        # 计算后面那个概率
        second = self.compute_probability(x, j)
        return -x * (first - second)  # + self.weight_lambda * self.w[j]


    def train(self, features, labels):
        '''对训练样本进行训练，求最优参数'''

        # len(labels)很大，而len(set(labels))可以求不同标签个数，即k=10
        self.k = len(set(labels))
        # 初始化参数（权重）矩阵，每一行代表一个类别的参数
        self.w = np.zeros((self.k, len(features[0]) + 1))
        times = 0
        while times < self.max_iteration:
            times += 1
            # 采用随机梯度下降SGD，每次不是用所有样本去计算梯度更新参数
            # 而是随机采样一个去计算梯度并更新参数，同样迭代次数下计算量小，且结果几乎一样
            index = random.randint(0, len(labels) - 1)
            x = features[index]
            y = labels[index]
            # x先转化为列表才能应用x.append(1.0),之后再换会数组形式
            x = list(x)
            x.append(1.0)
            x = np.array(x)
            # 在代价函数中，循环计算样本x关于k组权重的梯度(是个向量），并组合成梯度矩阵
            derivatives = [self.compute_partial_derivative(x, y, j) for j in range(self.k)]
            # 该次迭代后更新参数矩阵
            for j in range(self.k):
                self.w[j] -= self.learning_step * derivatives[j]


    def predict_single_sample(self, x):
        '''对单个测试样本x进行分类预测'''

        # x转置后才能进行 (k,n+1) * (n+1,1)的矩阵乘法
        x = np.transpose(x)
        # result中存储x属于每个类的概率
        result = np.dot(self.w, x)
        # 找最大值所在的列
        positon = np.argmax(result, axis = 0)
        return positon


    def predict(self, features):
        '''对测试样本进行标签预测'''

        labels = []
        for feature in features:
            # AttributeError: 'numpy.ndarray' object has no 
            # attribute 'append'。所以才有x = list(feature)
            x = list(feature)
            x.append(1)
            # 列表转化为矩阵
            x = np.matrix(x)
            labels.append(self.predict_single_sample(x))
        return labels
    

    def compute_accuracy_rate(self, test_labels_true, test_labels_predict):
        '''计算分类正确率'''

        count = 0
        for a, b in zip(test_labels_true, test_labels_predict):
            if a == b:
                count += 1
        return 1.0 * count / len(test_labels_predict)


if __name__ == '__main__':
    '''f = gzip.open('F:/jupyter/mnist.pkl.gz', 'rb')'''
    cifar10_dir = 'F:/jupyter/assignment1/cs231n/datasets/CIFAR10'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    # pickle模块可以序列化对象并保存到磁盘中，并在需要的时候读取出来，
    # 任何对象都可以执行序列化操作
    '''training_data, validation_data, test_data = pickle.load(f, encoding='bytes')'''
    # 经过pickle.load()后，三个类别里已经不是原始的7000个0,7000个1,7000个2
    # 这种顺序，而是打乱了。training_data = ([[图片1],[图片2],[]],[标签])
    # 图片不是以矩阵形式存储，想打印图片需要将归一化的数据*256，并int，再还原为28*28矩阵形式
    '''f.close()'''
    X_train = np.reshape(X_train,(X_train.shape[0], -1))
    X_test = np.reshape(X_test,(X_test.shape[0], -1))
    X_train = X_train / 256
    X_test = X_test / 256
    '''
    mini_batch_size = 1000

    training_data_data = X_train
    training_data_label = y_train
    # 对训练集随机切片取部分进行训练
    k = random.randint(0, 40000)
    # 这个地方不能少了[0,0]
    mini_train_batch = [0, 0]
    mini_train_batch[0] = training_data_data[k:k + mini_batch_size * 10]
    mini_train_batch[1] = training_data_label[k:k + mini_batch_size * 10]

    train_features = X_train
    train_labels = y_train
    # 这行注释可以观察数据结构print(mini_train_batch[0],mini_train_batch[1],mini_train_batch[0][0])
    # 创建多分类实例
    '''
    S = Softmax()
    time1 = time.time()
    S.train(X_train, y_train)
    time2 = time.time()
    print('训练用时: %.2f seconds'%(time2 - time1))
    print('正在预测中...')
    '''
    validation_labels_predict = S.predict(validation_data[0])'''
    test_labels_predict = S.predict(X_test)
    '''accuracy_validation = S.compute_accuracy_rate(validation_data[1], validation_labels_predict)'''
    accuracy_test = S.compute_accuracy_rate(y_test, test_labels_predict)
    '''print("在验证集上分类的正确率为：%f"%accuracy_validation)'''
    print("在测试集上分类的正确率为：%f"%accuracy_test)