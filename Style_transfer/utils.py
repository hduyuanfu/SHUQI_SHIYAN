import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
import copy

from loss import ContentLoss, StyleLoss, gram_matrix
from data_and_time import Normalization
'''
这个文件有三个问题需要明白：
1.怎么揉捏一个已有模型，在其基础上改造除一个自己的，而不是纯自己造
2.它这个优化器和以前见到的不一样,因为优化器不同所以函数closure()就用上啦。还有就是循环里定义函数也是可以的，不用非得在类里或靠最左边写。
发现一个东西：
list_ = [1,2,3]                                 for i in range(3): 
it = iter(list_)                                    def x(k): def x(k):
for i in range(3):                                      for j in range(k+1):
    def x(k): def x(k):                                     y = next(iter([1,2,3]), 0)
        for j in range(k+1):                            print(y)
            y = next(it, 0)                         x(i)
        print(y)
    x(i)
区别在于，左边这个for j in range(k+1)循环过程中是针对同一个可迭代对象；而右边for j in range(k+1)，y = next(iter([1,2,3]), 0)这里
每次都会新建一个可迭代对象，所以它只会输出第一个元素1
搭便车？？？
'''
def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img,
                               content_layers, style_layers, device):
    cnn = copy.deepcopy(cnn)  
    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []
    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially

    '''
    (1)给自定义的model建立了一个Sequential,万一以后有其他用处就方便点，如果没有这个nn.Sequential()，
    都不知道下面能不能用那个model.add_module(name, layer) ,为了方便以后自己也得有nn.Sequential();有个问题就是我怎么自定义呢？
    像该文件代码那样不建立类情况下，自定义出来一个像nn.features()和nn.classifier()一样的东西。下面代码亲自试验有效，可以任意订制模型：
    model = nn.Sequential()
    y = nn.Sequential( nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                        nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
    y.add_module('haha', nn.ReLU())
    model.add_module('features', y)
    z = nn.Sequential(nn.Dropout(p=0.5), nn.Dropout(p=0.5))
    model.add_module('classifier', z)
    print(model)
    print(model.features)
    至于怎么运行，就像下面代码一样，定义好结构，然后不用定义forward，直接model(input)，就执行了，记得把model各个块（各层）
    的参数放进优化器即可，其他什么反向传播等等都和调用模型时语法流程一样的。
    (2)建立类时：
    class LeNet(nn.Module):
        def __init__(self):
            super(LeNet, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 6, 5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(6, 16, 5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            # 由于调整shape并不是一个class层，
            # 所以在涉及这种操作（非nn.Module操作）需要拆分为多个模型
            self.classifiter = nn.Sequential(
                nn.Linear(16*5*5, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, 10)
            )
        def forward(self, x):
            x = self.features(x)
            x = x.view(-1, 16*5*5)
            x = self.classifiter(x)
            return x
    '''

    model = nn.Sequential(normalization)  #normalization才是model[0]
    '''相当于我自己要从头开始一块块(一个积木，一层，一个零件)建立一个模型了'''
    '''
    思想是在已有模型基础上进行分拆，添加，组合；而到底怎么做，要根据你利用的已有模型的结构来分析；
    比如VGG模型就是Conv,Relu,BN,Pool交替循环，所以才有了下面为什么这么写。
    '''
    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_%d'%i
        elif isinstance(layer, nn.ReLU):
            name = 'relu_%d'%i
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_%d'%i
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_%d'%i
        else:
            raise RuntimeError('Unrecognized layer: %s'%(layer.__class__.__name__))

        model.add_module(name, layer)  # 把layer添加到model并取名为name

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()  # 跑一下模型，现在定义了多少多长跑多长，就可以得到中间的C x features map
            content_loss = ContentLoss(target)  # 对ContentLoss这个模型类进行实例化，但还没有赋值进行forward
            model.add_module("content_loss_%d"%i, content_loss)  # 将这个实例添加进模型
            content_losses.append(content_loss)  # 实例添加进列表

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_%d"%i, style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):  # 从最后一层遍历到第一层
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]  # 这个模型后面用不到的几层既可以截掉啦

    return model, style_losses, content_losses



def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std, content_img, style_img, input_img, 
                       content_layers_default, style_layers_default, device,
                       num_epochs=300, style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn, normalization_mean,\
                                                        normalization_std, style_img, content_img,
                                                        content_layers_default, style_layers_default, device)
    optimizer = get_input_optimizer(input_img) 
    
    '''朴素风格迁移算法，把要调整的图片矩阵Tensor当做参数放进优化器，这样在反向时就可以修改参数，也就是修改图片'''
    '''要调整什么参数，就把什么参数放进优化器；整个模型参数都调整则地方；一些固定一些调整怎么放？'''

    '''
    清空梯度-向前传播-计算loss-反向传播-更新参数
    网络模块参数订制：为不同子网络赋值不同的学习率，finetune常用，使分类器学习参数更高，理论上使学习速度更高
    1.根据构建网络时划分好的模组进行学习率设定：
    optimizer = optim.SGD([{'params': net.features.parameters()}, # 默认lr是1e-5
                       {'params': net.classifiter.parameters(), 'lr': 1e-2}], lr=1e-5)
    2.以网络层为对象进行分组，并设定学习率：
    # 以层为单位，为不同层指定不同的学习率
    # 提取指定层对象
    special_layers = nn.ModuleList([net.classifiter[0], net.classifiter[3]])  # nn.ModuleList()这个类，把模块组装成列表
    # 获取指定层参数id
    special_layers_params = list(map(id, special_layers.parameters()))
    # map(函数，可迭代对象如列表元组)；用函数处理每一个迭代对象中的元素，返回一个迭代器，list(map()),迭代器转为列表
    print(special_layers_params)
    # 获取非指定层的参数id
    base_params = filter(lambda p: id(p) not in special_layers_params, net.parameters())
    optimizer = t.optim.SGD([{'params': base_params},
                            {'params': special_layers.parameters(), 'lr': 0.01}], lr=0.001)
    注：1.lambda作为一个表达式，定义了一个匿名函数。下例中代码x为函数入口参数，x+2为函数体。用lambda写法，
    简化了函数定义的书写形式，使代码更为简洁。还有几个定义好的全局函数：filter()、map()、reduce()。这些全局函数可以和lambda配合使用。
    func=lambda x:x+2等价于def func(x): return(x+2)
    2.filter(函数，序列)函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表。
    该函数接收两个参数，第一个为函数，第二个为序列，序列的每个元素作为参数传递给函数进行判断，然后返回 True 或 False，最后将返回 
    True 的元素放到新列表中。
    3.由map(),filter()可以推测reduce()用法和他们两个差不多
    '''
    '''
    为什么这里用while而不是for,因为对于一些损失函数，不想Adam和SGD正反向一次就optimizer.step()一步，有些损失函数
    比如共轭梯度和LBFGS默认参数是调用一次就会正反向计算并评估目标函数多次，比如下面的程序里optimizer.step(closure)一次
    就相当于优化了20次。若num_epochs=10,即迭代优化10次；用for可以强制执行optimizer.step(closure)10次，则实际上
    10*20=200次优化，而我们只要10次啊！用while时，第一次optimizer.step(closure)后已经优化20次，不在小于num_epochs,
    则会停止。num_epochs=300,用while则运行15次optimizer.step(closure);用for则执行6000次前向+反向的优化过程。
    '''
    print('Optimizing..')
    run = [0]
    while run[0] <= num_epochs:
    #for j in range(num_epochs):
        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)  # 因为input_img是参数，BP过程中会被调整，可能超出0-1范围

            optimizer.zero_grad()
            model(input_img)  #等价于model.forward(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss  # 实例.函数/变量；这里是实例.变量
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
            # if j % 50 ==0:
                print("run {}:".format(run))
                # print("run {}:".format(j))
                print('Style Loss : {:4f}  Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                    # 输入是tensor，在模型里运算再多层，涉及到的结果也是tensor，所以这里的style_score等还是tensor
                print()

            return style_score + content_score
        
        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img

'''
(1.1)Adam(Adaptive Moment Estimation)本质上是带有动量项的RMSprop，它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数
的学习率。它的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳。
(1.2)Adam的特点有： 
    1、结合了Adagrad善于处理稀疏梯度和RMSprop善于处理非平稳目标的优点; 
    2、对内存需求较小; 
    3、为不同的参数计算不同的自适应学习率; 
    4、也适用于大多非凸优化-适用于大数据集和高维空间。
(2.1)所有的优化器Optimizer都实现了step()方法来对所有的参数进行更新，它有两种调用方法：
    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
(2.2)for input, target in dataset:  # 这里强调把所有数据的batch过一遍(虽然每个batch会优化多次)，和上面讲的用for还是while不是一个东西
         def closure():             # 上一句才是实际读取数据，以前只是建立索引
             optimizer.zero_grad()
             output = model(input)
             loss = loss_fn(output, target)
             loss.backward()
             return loss
         optimizer.step(closure)
(3.1)为了使用torch.optim，需先构造一个优化器对象Optimizer，用来保存当前的状态，并能够根据计算得到的梯度来更新参数。 
要构建一个优化器optimizer，你必须给它一个可进行迭代优化的包含了所有参数（所有的参数必须是变量s）的列表。
然后，您可以指定程序优化特定的选项，例如学习速率，权重衰减等。
例如：optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
    optimizer = optim.Adam([var1, var2], lr = 0.0001)
    self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
(3.2)Optimizer还支持指定每个参数选项。 只需传递一个可迭代的dict来替换先前可迭代的Variable。dict中的每一项都可以
定义为一个单独的参数组，参数组用一个params键来包含属于它的参数列表。其他键应该与优化器接受的关键字参数相匹配，才能
用作此组的优化选项。
optim.SGD([
                {'params': model.base.parameters()},
                {'params': model.classifier.parameters(), 'lr': 1e-3}
            ], lr=1e-2, momentum=0.9)
如上，model.base.parameters()将使用1e-2的学习率，model.classifier.parameters()将使用1e-3的学习率。
0.9的momentum作用于所有的parameters。
(4.1)1：用相同数量的超参数来调参，SGD和SGD +momentum 方法性能在测试集上的额误差好于所有的自适应优化算法，尽管有时自适应优化算法在训练集上的loss更小，但是他们在测试集上的loss却依然比SGD方法高， 
     2：自适应优化算法 在训练前期阶段在训练集上收敛的更快，但是在测试集上这种有点遇到了瓶颈。 
     3：所有方法需要的迭代次数相同，这就和约定俗成的默认自适应优化算法 需要更少的迭代次数的结论相悖！
(4.2)自适应优化算法通常都会得到比SGD算法性能更差（经常是差很多）的结果，尽管自适应优化算法在训练时会表现的比较好，
因此使用者在使用自适应优化算法时需要慎重考虑！（因此CVPR的paper全都用的SGD了，而不是用理论上最diao的Adam）
'''
