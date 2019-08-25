
import torch.nn as nn
import torch
from torchvision import models
net = models.vgg16_bn()
print(net)
special_layers = nn.ModuleList([net.classifier[0], net.classifier[3]])
special_layers_params = list(map(id, special_layers.parameters()))
print(special_layers_params, id(net.classifier[0].parameters()))
print(id(net.classifier[0].parameters()))