import torch
import torch.nn as nn
from torchvision import models
x = models.vgg16_bn()
print(x)
#model = nn.Sequential()
#a = features( Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
y = nn.Sequential( nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
y.add_module('haha', nn.ReLU())
model.add_module('features', y)
z = nn.Sequential(nn.Dropout(p=0.5), nn.Dropout(p=0.5))
model.add_module('classifier', z)
print(model)
print(model.features)