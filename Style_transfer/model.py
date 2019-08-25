import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms
import copy

cnn = models.vgg19(pretrained=False).features.to(device).eval()