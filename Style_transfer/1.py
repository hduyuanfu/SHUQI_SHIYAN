
import torch.nn as nn
class A(nn.Module):
    def __init__(self, target, c):
        super(A, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target
        self.c = c
    def forward(self, input):
        self.loss = self.target * self.c * input
        return input
        '''
        (1)在类函数里参与计算的变量，必须是__init__函数中的self.变量名,(实例化时传进来的变量必须全部在init中转化为
        self变量后才能在forward等函数中应用)或者forward等函数中被赋值的变量;
        (2)为什么在forward函数中给loss变量加self呢？因为不加self的话，不能用 实例.变量名 调用，提取不出来了，而我们的
        目的就是用self.loss暂存计算结果。
        (3)什么是模型类，(nn.Module),并且super(A, self).__init__()的就是，也只有这种类才能有forward函数，才能model(参数)
        等价于model.forward(参数)
        '''
'''       
f = A(2, 4)
x = f(-1)
print(x)
print(f.loss)
'''
list_ = [1,2,3]
it = iter(list_)
for i in range(3):
    def x(k):
        for j in range(k+1):
            y = next(iter([1,2,3]), 0)
        print(y)
    x(i)
'''
it = iter([17,2,3])
for ii in range(3):
    y = next(it, 0)
    print(y)
    '''
'''
list_ = [1, 2, 3, 4, 5]
it = iter(list_)
for i in range(5):
    line = next(it)
    print("第%d 行， %s" %(i, line))
'''
