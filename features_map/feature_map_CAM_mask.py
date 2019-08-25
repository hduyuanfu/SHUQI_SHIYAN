import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
'''动起手来就已经成功一半了'''

def draw_features_tensorboard(features):
    channel = features.size(1)
    L = []
    for i in range(channel):     
        img = features[0, i, :, :]  #一个通道一个通道的取出来，每一个通道都产生一张热力图
        pmin = torch.min(img)  # torch.min/max()和np.min/max()用的范围不一样
        pmax = torch.max(img)    
        img = ((img - pmin) / (pmax - pmin + 0.000001))*255  #float在[0，1]之间，转换成0-255        
        img = img.numpy()  # 因为cv2只能处理int类型的numpy,并且.astype(np.uint8)只能处理numpy类型；所以这一步
        img = img.astype(np.uint8)  #转成unit8        
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET) #生成heat map(仍是numpy数组)       
        img = img[:, :, ::-1]#注意cv2（BGR）和matplotlib(RGB)通道是相反的  
        # [：，：，：：-1],第一个冒号——取遍图像的所有行数,第二个冒号——取遍图像的所有列数,第三个和第四个冒号——取遍图像的所有通道数，-1是反向取值      
        img = Image.fromarray(img)  # 我所熟悉的就是transform肯定适合读入RGB类型的图片，而不是numpy数组，所以从numpy到Image
        img = transform_tensorboard(img)
        # 不需要用transform(img).unsqueeze(0)；transform从来不都是直接读入RGB的Image吗？什么时候见过四维度的；
        # 以前那些四维度的BCHW都是在dataloader那里指定的，和transform无关
        
        L.append(img)
    return make_grid(L, normalize=True)
 
'''属性不用(),比如img.shape,img.size;只有函数才加()'''
def draw_features_plot(features, savename):
    fig = plt.figure(figsize=(16, 16))  #窗口上创建一个指定大小新图形
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    channel = features.size(1)   
    for i in range(channel):     
        plt.subplot(np.ceil(channel/8.), 8, i + 1)      
        plt.axis('off')
        img = features[0, i, :, :]
        pmin = torch.min(img)
        pmax = torch.max(img)        
        img = ((img - pmin) / (pmax - pmin + 0.000001))*255  #float在[0，1]之间，转换成0-255 
        img = img.numpy()       
        img = img.astype(np.uint8)  #转成unit8        
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET) #生成heat map        
        img = img[:, :, ::-1] #注意cv2（BGR）和matplotlib(RGB)通道是相反的 
        #print(img.shape)
        img =  cv2.resize(img, (400, 400)) 
        #print(img.shape)    
        plt.imshow(img)  # 这句就相当于把小图按照位置添加进去(不是交互模式，需要plt.show()才会真正显示出来)      
    fig.savefig(savename, dpi=100)    
    fig.clf()  # # Clear figure
    plt.close()  # Close a figure window(plt是窗口，而fig则是1位于该窗口上的一个图形)
    # cla()   Clear axis


def mask_Generation(feature, alpha):
    batch_size = feature.size(0)
    kernel = feature.size(2)
    sum = torch.sum(feature.detach(), dim=1)
    # 不管怎么样，把计算过程中的features map拿出来做其他事了，还是.detach()安全啊
    # features是B * C * H * W，dim=1指的是C，沿着C求和，就把C维度搞没了，所以sum是 B * H * W
    avg = torch.sum(torch.sum(sum, dim=1), dim=1) / kernel ** 2
    # dim=1代表第二个维度，也就是把H搞没了，变成C * W；此时W就成了第二个维度，所以又是一个dim=1，变成了B
    mask = torch.where(sum > alpha * avg.view(batch_size, 1, 1), torch.ones(sum.size()), #.to(device),
                       (torch.zeros(sum.size()) + 0.2))
                       # 三个输入参数，第一个是判断条件，第二个是符合条件的设置值，第三个是不满足条件的设置值
    mask = mask.unsqueeze(1)
    # mask一会儿还要和其他features map交互，所以它不能比别人少个维度啊，得加一个虚拟的通道维度
    return mask

net = models.vgg16_bn(pretrained=True).eval()
#for i in net.features.children():
    #print(i)
'''我要重建，大改model问题太多'''
class Map(nn.Module):
    def __init__(self):
        super(Map, self).__init__()
        self.model = nn.Sequential()
        self.names = []
        i = j_1 = j_2 = j_3 = j_4 = 1  # increment every time we see a conv
        '''实验过后发现，a=b=1,对a变化不影响b，尽管id一样；a,b=1,1时也不会相互影响'''
        for layer in net.features.children():
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


            self.pool_kernel4 = nn.MaxPool2d(kernel_size=4)
            self.pool_kernel2 = nn.MaxPool2d(kernel_size=2)


        #print(self.model)

    def forward(self, input_img):
        #writer = SummaryWriter(log_dir='F:/PYTHON/features_map/feature_map')
        for i, layer in enumerate(self.model.children()):
            print(self.names[i])
            input_img = layer(input_img)
            

            if self.names[i] == 'conv3_3':
                feature1 = input_img
            elif self.names[i] == 'conv4_3':
                feature2 = input_img
            elif self.names[i] == 'conv5_3':
                feature3 = input_img


            #draw_features_plot(input_img.detach(), 'F:/PYTHON/features_map/cam_plot/%s'%self.names[i])

            #writer.add_image(self.names[i] + '-CAM', draw_features_tensorboard(input_img.detach()), i)

            input_img = input_img.permute(1, 0, 2 ,3)
            
            #writer.add_image(self.names[i], make_grid(input_img, normalize=True), i)
            #print(input_img.size())
            input_img = input_img.transpose(1, 0)
        #writer.close()

        slack_mask1 = mask_Generation(feature1, alpha=0.8)
        slack_mask2 = mask_Generation(feature2, alpha=0.7)
        slack_mask3 = mask_Generation(feature3, alpha=0.6)
        
        Aggregated_mask = self.pool_kernel4(slack_mask1) * self.pool_kernel2(slack_mask2) * slack_mask3
        Aggregated_mask = F.interpolate(Aggregated_mask, scale_factor=16, mode='bilinear', align_corners=True)
        masked_image = image * Aggregated_mask  # shape (1, 3, 400, 400)
        masked_image = masked_image.squeeze(0)
        # 还要把tensor转化为0-255的int8类型的numpy才能用plt去画

        masked_image =masked_image * 255  #float在[0，1]之间，转换成0-255 
        masked_image = masked_image.permute(1, 2, 0)  # 只有tensor才能用permute()
        # 没有上面这行则TypeError: Invalid shape (3, 400, 400) for image data；说明plt.imshow()需要的是HWC，而不是CHW
        masked_image = masked_image.numpy()       
        masked_image = masked_image.astype(np.uint8)  #转成unit8
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(masked_image)
        plt.show()

model =  Map()
image_path = 'F:/lenna.jpg'
image = Image.open(image_path)
transform = transforms.Compose([
                                transforms.ToTensor(), #转为Tensor格式，并将值取在[0,1]中
                                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #标准化，得到在[-1,1]的值
                                ])
transform_tensorboard = transforms.Compose([
                                transforms.Resize(400),  # 不这样做的话，越往后得到的feature map尺寸越小，显示效果不好
                                transforms.ToTensor()
                                ])
image = transform(image).unsqueeze(0)
'''因为网络模型需要batch这个维度，所以如果不是通过dataloader这种函数导入的多张或单张图片(这个函数会自动给你添加一个维度)，
想要一张图片输入网络模型去跑，需要添加一个虚拟维度'''

model(image)