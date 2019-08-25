import torch 
import torch.nn as nn
from torchvision import models, transforms, datasets
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
'''动起手来就已经成功一半了'''
net = models.vgg16_bn(pretrained=True).eval()  # 用预训练好的vgg16可视化lenna图片各层features map

'''我要利用net这个模型实例重建一个模型，大改net的话问题太多了'''
class Map(nn.Module):
    def __init__(self):
        super(Map, self).__init__()
        self.model = nn.Sequential()
        self.names = []
        i = j_1 = j_2 = j_3 = j_4 = 1
        # 卷积-BN-激活-池化(遇到池化就划分一截)为一个部分，用i表示；j_1,2,3,4表示每个部分内的卷积，BN，激活，池化
        '''实验过后发现，a=b=1,对a变化不影响b，尽管id一样；a,b=1,1时也不会相互影响'''
        for layer in net.features.children():  # 我们不需要classifier中的全连接等层(这个for循环主要就是为了给每层重新起个名字)
            if isinstance(layer, nn.Conv2d):
                name = 'conv%d_%d'%(i, j_1)
                j_1 += 1
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn%d_%d'%(i, j_2)
                j_2 += 1
            elif isinstance(layer, nn.ReLU):
                name = 'relu%d_%d'%(i, j_3)
                j_3 += 1
            elif isinstance(layer, nn.MaxPool2d):
                name = 'maxpool%d_%d'%(i, j_4)
                i += 1
                j_1 = j_2 = j_3 = 1
            #else:
                #raise RuntimeError('Unrecognized layer: %s'%(layer.__class__.__name__)
            self.model.add_module(name, layer)  # 把layer添加到model并取名为name
            self.names.append(name)
        #print(self.model)  # 也可以打印出来模型看看哦
    def forward(self, input_img):
        writer = SummaryWriter(log_dir='F:/PYTHON/features_map/feature_map')  # 定义tensorboard文件
        for i, layer in enumerate(self.model.children()):
            print(self.names[i])  # 输出这层名字
            input_img = layer(input_img)
            input_img = input_img.permute(1, 0, 2 ,3)  # 第0和第1维转置，交换位置，是为了适应make_grid函数
            print(input_img.size())  # 打印维度看看，变化成功了
            result = make_grid(input_img, normalize=True)  # 一个channel代表一张图片，把每层的多个图片拼接成一张大图
            writer.add_image(self.names[i], result, i)  # 每次写入时step为i,不写这个step不报错，但你会发现新图可能会覆盖旧图(没试过)
            input_img = input_img.transpose(1, 0)  # Tensor维度还要变回去，否则下次循环，维度不是BCHW的话layer(input_img)会出错的
        writer.close()
        # permute(1,0,2,3)需要把所有维度都写出来；transpose(1,0)可以只把需要转置的几个维度写出来；功能是差不多的
model =  Map()  # 新模型实例
image_path = 'F:/lenna.jpg'
image = Image.open(image_path)
transform = transforms.Compose([
                                transforms.ToTensor(), #转为Tensor格式，并将值取在[0,1]中
                                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #标准化，得到在[-1,1]的值
                                ])
image = transform(image).unsqueeze(0)  # 图片有三个通道，而VGG等网络的输入都需要BCHW,增加一个虚拟维度

model(image)  # 让新模型跑起来(坐等结果吧)