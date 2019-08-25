vscode问系统python.exe在哪里；然后系统去环境变量和用户变量里找，找到了python.exe后告诉vscode，代码就可以执行了。
系统优先是优先搜寻系统变量的，找不到才去另一个，所以一个路径只需要在添加在一个地方就好，没必要都添加。
'''
不能在根目录下创建文件夹，先关闭个已经打开的文件，趁着这个时候就可以了，或者直接去F盘中创建文件夹
exit(),clear,ctrl+z是撤销
一个电脑可以装好几个python环境，想用哪个可以直接点击左下角选环境，点中哪个则.vscode中的.settings中的路径会自己变化，
或者自己直接去修改settings
'''
'''
import numpy as np
import math
print(np.zeros(5).shape)
print(np.zeros((1, 5)).shape)
print(math.log(2.81))
a = np.array([[1],[2],[4]])
print(a.shape)
b = np.array([[2],
     [2],
     [3]])
print(b.shape)
z = -np.sum(np.dot((4-a).T, b))
print(z)
'''
'''
import numpy as np
# 创建数组需要两个（），np.array()函数自带一个,再把数组(1, 2, 3, 2)放进去
a = np.array((1, 2, 3, 2)) 
print(len(a))
print(len(set(a)))
'''
import numpy as np
import math
import torchvision
w = np.array([[1, 2],
              [3, 5]])
# w * w后还是个矩阵，只是对应位置元素相乘
print(w*w)

d = list([0.1, -0.2, 0.3])
d= np.array(d)/10
print(d)
print(d)
count = 0
for x in d:
    count += math.fabs(x)
print(count / 3)