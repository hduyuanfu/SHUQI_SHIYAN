'''
from torchvision import models, datasets, transforms
import torch.nn as nn
model = models.vgg16_bn(pretrained=True, progress=True)
print(model)  # 打印模型结构，便于修改
'''
import torch as t
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms

# 超参数
batch_size = 8 
learning_rate = 1e-2
num_epoches = 5

'''处理数据集'''
# 定义对数据的预处理
# Compose,将好几个变换组合在一起(下面读取数据时，需要将图片数据转化为Tensor才能在框架跑，想可视化图片则又需要变回去)
transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),  transforms.Normalize([0.5], [0.5])])
# 图片为几通道，则有几个数[0.5],三通道则[0.5, 0.5, 0.5];中括号外再加个小括号可以([0.5, 0.5, 0.5]);
# 但只是(0.5, 0.5, 0.5)用小括号要报错的
# 获取训练集
trainset = datasets.MNIST(root='F:/shujuji/', train=True, download=True, transform=transform)
'''
print(type(trainset))
print(len(trainset))
trainset = trainset[:100]
print(type(trainset))
print(len(trainset))
'''

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)  # 不要用多线程
# 测试集
testset = datasets.MNIST(root='F:/shujuji/', train=False,  transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

class BuildVgg16_bn(nn.Module):
    def __init__(self, model_type, output_num):
        super(BuildVgg16_bn, self).__init__()

        #self.model_type = model_type

        if model_type == 'pre':
            model = models.vgg16_bn(pretrained=True)
            # 已经把model定义为vgg16_bn框架(和参数)了，可以作修改和微调了
            model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.features = model.features
            model.classifier[6] = nn.Linear(4096, output_num)
            self.classifier = model.classifier
        elif model_type == 'new':
            # VGG16_bn跑的太慢，可以用阉割版，去掉了一些卷积、归一化和激活，但是池化和全连接保留下来了。但结果特别差
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            )

            self.classifier = nn.Sequential(
                nn.Linear(in_features=25088, out_features=4096, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(in_features=4096, out_features=4096, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(in_features=4096, out_features=10, bias=True)
            )
    
    def forward(self, x):
        x  = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


'''实例化模型，定义损失函数和优化器'''

# 模型实例化
model = BuildVgg16_bn('pre', 10)
print(model)
# 定义该实例模型的loss函数和其优化方法
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

'''model.parameters()不是很理解'''
'''训练模型，测试模型'''

for epoch in range(num_epoches):
    # 一趟训练
    model.train()
    i = 0
    for data in trainloader:  # 每次取一个batch_size
        if i < 50:  # 通过循环，强制程序只训练50*batch_size个样本
            img, label = data  # img.size:8*1*224*224
            output = model(img)
            loss = loss_fn(output, label)
            # 反向传播
            optimizer.zero_grad()  # 这里清零指的是把上一次计算出来的delta(w)清零，也就是上一次的梯度清零，不是w-old清零
            loss.backward()
            optimizer.step()
            print('i:%d'%i)
            i += 1
        else:
            break
    # 测试集上检验效果
    print('开始预测：')
    model.eval()
    total_eval_loss = 0
    total_num_correct = 0
    j = 0
    for data in testloader:
        if j < 50:  # 只测试50*batch_size个样本
            img, label = data
            out = model(img)
            # 这一个batch_size中128个图片，求得每张图平均损失就是下面的loss
            loss = loss_fn(out, label)
            total_eval_loss += loss.item() * label.size(0)  # Tensor.item()取出张量中打的值
            # out = batch_size*各类得分。torch.max(out,0/1),0跨行，1跨列；返回值是一个元素为Tensor的二元组，(各最大值，各索引)
            _, pred = t.max(out, 1)
            num_correct = (pred == label).sum()
            total_num_correct += num_correct.item()
            print('j:%d'%j)
            j += 1
        else:
            break
    # 输出的其实相当于每张图片的平均loss
    print('epoch: %d; test_loss: %.3f; test_acc: %.3f'%(epoch, 
            total_eval_loss / 400, total_num_correct / 400))
'''400个训练样本，400个测试样本；第一轮训练过后，测试准确率为0.73'''


