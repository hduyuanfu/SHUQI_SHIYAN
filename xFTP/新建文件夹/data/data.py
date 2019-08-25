import os
import torch
import numpy as np
from torchvision import transforms,utils
from torch.utils.data import Dataset,DataLoader
from PIL import Image

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    '''通过上面的形式可以定义我们需要的数据类'''
    def __init__(self, txt, transform, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:  # python按行读取数据
            line = line.strip('\n')  # 去掉每行首尾的换行符
            line = line.rstrip()  # 去掉字符串末尾空格
            words = line.split()  # split的默认参数是空格，所以不传递任何参数时分割空格，在英文中也就等同于分割单词
            imgs.append((words[0], int(words[1])))

        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader('/data/guijun/aircraft/images/' + fn)
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)