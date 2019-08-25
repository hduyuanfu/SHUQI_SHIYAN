import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms
import copy
import time
from data_and_time import image_loader, timeSince, asMinutes
from utils import *

'''debug解决不了逻辑错误'''
'''通过在代码中添加不同输出，观察运行流程和变量变化过程'''
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
'''
普通类和模型类：
普通类，(object),实例化为A，A.变量名，A.函数名(参数)才会执行计算；
模型类，(nn.Module)，必定有forward函数，实例化为B,则B(参数)就会执行，此时B(参数) = B.forward(参数)
调用torchvision的transforms.ToTensor()，像素会被转换到0-1
'''
class Config(object):

    img_size = 512 if torch.cuda.is_available() else 128  # use small size if no gpu
    num_epochs= 300 # 300
    style_weight= 1000000
    content_weight= 1
    content_layers_default = ['conv_4']
    style_layers_default = ['conv_2', 'conv_4', 'conv_7', 'conv_11']
    content_path = "F:/PYTHON/Style_transfer/pictures/dancing.jpg"
    style_path = "F:/PYTHON/Style_transfer/pictures/picasso.jpg"

opt = Config()

cnn = models.vgg19(pretrained=True).features.to(device).eval()

normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

content_img = image_loader(opt.content_path, opt.img_size, device)
style_img = image_loader(opt.style_path, opt.img_size, device)
input_img = content_img.clone()

# if you want to use white noise instead uncomment the below line:
# input_img = torch.randn(content_img.data.size(), device=device)

# add the original input image to the figure:
#plt.figure()
#imshow(input_img, title='Input Image')
start = time.time()
output = run_style_transfer(cnn, normalization_mean, normalization_std, content_img, style_img, input_img,
                            opt.content_layers_default, opt.style_layers_default, device,
                            opt.num_epochs, opt.style_weight, opt.content_weight)

timeSince(start)
writer = SummaryWriter(log_dir='train_result')  # tensorboard直接就能保存tensor，只要图片符合C x H x W就行。
writer.add_image('img', output.squeeze(0), 1)   # 前面为什么要增一维，导致这里又要减一维
writer.close()

# tensor只有一个元素，tensor.item();有很多元素tensor.data
# transforms.ToTensor()与下面对应，相互转化吧(不说类型，数值的变化为0-255转化为0-1；下面则相反)
# unloader = transforms.ToPILImage()  # reconvert into PIL image
def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    '''
    实验证明，clone()是为了不改变原来的变量；
    x = torch.Tensor(2,2).fill_(2);y = x.clone();y = x.clone().view(4);
    clone()可以重新开辟一块内存，x,y互不影响，克隆时也可以改变形状。
    import copy
    x = torch.Tensor(2,2).fill_(2);y = copy.copy(x),此时x,y指向一个内存单元;
    y = copy.deepcopy(x)重新开辟内存，互不影响。
    注：(1)copy.copy(),clone(),copy.deepcopy()都可以复制时改变维度；(2)Tensor同时又属性view()和reshape();numpyq却只用reshape
    copy()与deepcopy()区别：
    L1 = [1, 2, 3, 4]
    L = [ L1, 5, 7]
    L2 = L  这种情况就是建立一个索引指针
    Import copy
    L3 = copy.copy(L2)  只是拷贝了L2这一层，没有进行深层次的拷贝，即在内存中开辟了一个空间，与变量L3进行绑定，
                        L3的第一个元素与L1绑定，L3的第2个元素和第3个元素分别与对象5和对象7绑定，所以改变L1中的元素，
                        L3中的元素会随之改变，但L2中的元素改变时，L3中的元素不会改变
    L4 = copy.deepcopy(L2)  当于将L2中的每一层都进行拷贝了一遍，所以无论怎么改变L1和L2中的元素，L4中的元素都不会改变。

    '''
    '''clamp_用法：input_img.data.clamp_(0, 1)把值拉回到0-1范围内'''
    
    image = image.squeeze(0)      # remove the fake batch dimension
    unloader = transforms.ToPILImage()
    image = unloader(image)
    '''
    事实证明，想要用下面这一句话代替上面两句话是不行的，报错TypeError: Image data of dtype object cannot be converted to float
    image = transforms.ToPILImage(image)
    所以关于transforms的使用还是需要先定义再用，不能直接用
    '''
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.show()

    # plt.pause(0.001) # pause a bit so that plots are updated
imshow(output, title='Output Image')
'''
matplotlib的显示模式默认为阻塞（block）模式。什么是阻塞模式那？我的理解就是在plt.show()之后，程序会暂停到那儿，并不会继续执行下去。
如果需要继续执行程序，就要关闭图片。那如何展示动态图或多个窗口呢？这就要使用plt.ion()这个函数，使matplotlib的显示模式转换为交互
（interactive）模式。即使在脚本中遇到plt.imshow()等需要画图或表的操作，把图画完之后，代码还是会继续执行。

在交互模式下：
1、plt.plot(x)或plt.imshow(x)是直接出图像，不需要plt.show()
（是说每个图不用单独plt.show()，但plt.ioff()，plt.show()在最后必须得有，是固定用法）
2、如果在脚本中使用ion()命令开启了交互模式，没有使用ioff()关闭的话，则图像会一闪而过，并不会常留。
要想防止这种情况，需要在plt.show()之前加上ioff()命令。
plt.ion()    # 打开交互模式    
# 同时打开两个窗口显示图片    
plt.figure()  #图片一    
plt.imshow(i1)    
plt.figure()    #图片二    
plt.imshow(i2)
...想画几个图画几个，等所有需要画的图都添加进代码行后，可以下面这两句啦。
plt.ioff()  # 显示前关掉交互模式
plt.show()

在阻塞模式下：
1、打开一个窗口以后必须关掉才能打开下一个新的窗口(同样的，这个图片窗口不关闭程序就卡这里了)。
这种情况下，默认是不能像Matlab一样同时开很多窗口进行对比的。
2、plt.plot(x)或plt.imshow(x)不是直接出图像，需要plt.show()后才能显示图像
'''