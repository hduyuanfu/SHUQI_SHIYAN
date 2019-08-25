from data_utils import load_CIFAR10
from full_connect import *
import random
import gzip                              # 解压数据集
import pickle 
f = gzip.open('F:/jupyter/mnist.pkl.gz', 'rb')
# pickle模块可以序列化对象并保存到磁盘中，并在需要的时候读取出来，
# 任何对象都可以执行序列化操作
training_data, validation_data, test_data = pickle.load(f, encoding='bytes')
f.close()
mini_batch_size = 1000
training_data_data = training_data[0]
training_data_label = training_data[1]
# 对训练集随机切片取部分进行训练
k = random.randint(0, 40000)
# 这个地方不能少了[0,0]
mini_train_batch = [0, 0]
mini_train_batch[0] = training_data_data[k:k + mini_batch_size * 10]
mini_train_batch[1] = training_data_label[k:k + mini_batch_size * 10]
X_train = mini_train_batch[0]
y_train = mini_train_batch[1]
X_val = training_data_data[k:k + mini_batch_size]
y_val = training_data_label[k:k + mini_batch_size]
'''
cifar10_dir = 'F:/jupyter/assignment1/cs231n/datasets/CIFAR10'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
X_val = X_train[49000:]
y_val = y_train[49000:]
X_train = X_train[:49000]
y_train = y_train[:49000]
'''
data = {
        'X_train': X_train*256,
        'y_train': y_train,
        'X_val': X_val*256,
        'y_val': y_val
    }  # 以字典形式存入训练集和验证集的数据和标签
model = FullyConnectedNet(hidden_dims=[500,100], reg=10)  # 我们的神经网络模型
solver = Solver(model, data,  # 模型，数据
                    update_rule='sgd_momentum',  # 优化算法
                    optim_config={      # 该优化算法的参数
                        'learning_rate': 1e-3,  # 学习率
                    },
                    lr_decay=0.95,     # 学习率的衰减速率
                    num_epochs=10,     # 训练模型的次数
                    batch_size=100,    # 每次丢入模型训练的图片数目
                    print_every=100,
                    verbose=True)
solver.train()

