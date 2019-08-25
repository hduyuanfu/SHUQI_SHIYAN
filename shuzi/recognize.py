import network
import mnist_loader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

training_data, validation_data, test_data =mnist_loader.load_data_wrapper()
net = network.Network([784, 30, 10])
# 并不是在实例化的时候就把所有参数都定义好，二是在引用类里函数时在给参数赋值
net.SGD(training_data, 30, 3, 3.0, test_data)
'''
img = Image.open('F:/PYTHON/shuzi/2.png').convert('L')
test_inputs = np.reshape(img, (784, 1)) 
test_data = [(test_inputs, -1)]
net.SGD(training_data, 30, 3, 3.0, test_data)
'''