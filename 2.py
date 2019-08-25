'''
import numpy  as np
a = list([1,2,3,0, 0,4,5])
print([i  for i, x in enumerate(a) if x == 0])

b = list([5,8,6,7,5,4,5,8],
         [4,8,6,1,5,6,5,0])
b = mat(b)
a = a[::-1]
index =a[:2]
print(index)
x = b[:,np.array(index)]
print(x)

import numpy as np 
w = np.matrix([[1,2],
               [2,4]])
x = w*w
y = w.dot(w)
print(x)
print(y)

a = np.zeros(4)
b = np.zeros((4,))

print(a.shape)
print(b.shape)
'''
'''
grads={}
num_layers = 4
grads['W%d'%(num_layers)] = 'nihao'
print(grads['W4'])
'''

from torchvision import models, datasets, transforms
import torch.nn as nn
model = models.resnet50(pretrained=False, progress=True)
print(model)
'''
import torch
import torchvision
print(torch.__version__)
print(torchvision.__version__)
'''
'''
import torch.nn as nn
class Bottleneck(nn.Module):
    expansion = 4.
    def __init__(self):
        self.z = 2.
        super(Bottleneck, self).__init__()
        self.x = 2.
    k = 3
    def a(self):
        self.t = 9
        self.v = self.x * exp
    def b(self):
        self.e = 6

y = Bottleneck()
print(y.a().v)
'''
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),  transforms.Normalize([0.5], [0.5])])
testset = datasets.MNIST(root='F:/shujuji/', train=False,  transform=transform)
testloader = DataLoader(testset, batch_size=8, shuffle=False)
print(len(testloader))
print(len(testloader.dataset))
