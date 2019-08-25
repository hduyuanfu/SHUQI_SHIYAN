'''
from torchvision import models, datasets, transforms
import torch.nn as nn
model = models.vgg16_bn(pretrained=True, progress=True)
print(model)  # 查看模型结构，便于修改
'''
import time
import math
import torch as t
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

def asMinutes(s):
    '''用于计算每个epoch花费时间'''
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since  # 已经花费时间s
    es = s / (percent)  # 已花费时间/已计算的比例=预计共费时es
    rs = es - s  # 剩余用时
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# 超参数
batch_size = 8 
learning_rate = 1e-2
num_epoches = 2

'''处理数据集'''
# 定义对数据的预处理
# Compose,将好几个变换组合在一起(下面读取数据时，需要将图片数据转化为Tensor才能在框架跑，想可视化图片则又需要变回去)
transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),  transforms.Normalize([0.5], [0.5])])
# 图片为几通道，则有几个数[0.5],三通道则[0.5, 0.5, 0.5];中括号外再加个小括号可以([0.5, 0.5, 0.5]);
# 但只是(0.5, 0.5, 0.5)用小括号要报错的
# 获取训练集
trainset = datasets.MNIST(root='/data/yuanfu/VGG16_bn_mnist/', train=True, download=False, transform=transform)
'''
print(type(trainset))  trainset类型不是可以切片的类型
print(len(trainset))
trainset = trainset[:100]
print(type(trainset))
print(len(trainset))
'''

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)  # 不要用多线程
# 测试集
testset = datasets.MNIST(root='/data/yuanfu/VGG16_bn_mnist/', train=False, download=False, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

class BuildVgg16_bn(nn.Module):
    def __init__(self, model_type, output_num):
        super(BuildVgg16_bn, self).__init__()

        #self.model_type = model_type

        if model_type == 'pre':
            model = models.vgg16_bn(pretrained=False)
            # 已经把model定义为vgg16_bn框架(和参数)了，可以作修改和微调了
            model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.features = model.features
            model.classifier[6] = nn.Linear(4096, output_num)
            self.classifier = model.classifier
    
    def forward(self, x):
        x  = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


'''实例化模型，定义损失函数和优化器'''

# 模型实例化
model = BuildVgg16_bn('pre', 10)
model.cuda()  
# model.cuda() == model =model.cuda() ,因为是对model自身内存进行迁移，而变量只是一个从cpu到GPU的拷贝，所以须是后者
model = nn.DataParallel(model, device_ids=[0,1,2])
#print(model)
# 定义该实例模型的loss函数和其优化方法
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
'''model.parameters()不是很理解'''



'''训练模型，测试模型'''
train_accuracy_history = []
test_accuracy_history = []

loss_per_epoch = []  # 记录每次epoch的平均的损失

start = time.time()
for epoch in range(1, 1 + num_epoches):
    print('第%d轮'%epoch)
    print('开始训练：')
    model.train()
    # 进行num_epoches趟训练
    train_loss = 0  # 用来累计每个batch的loss
    total_train_num_correct = 0
    for data in trainloader:  # 每次取一个batch_size
        
        img, label = data  # img.size:8*1*224*224
        img, label = img.cuda(), label.cuda()
        output = model(img)
        loss = loss_fn(output, label)
        # 将每个batch的损失加起来(上面的loss为这个batch中样本的的平均损失)
        train_loss += loss.item() * batch_size
         # 将每个batch的正确个数加起来
        _, pred = t.max(output, 1)
        num_correct = (pred == label).sum()
        total_train_num_correct += num_correct.item()
        # 反向传播
        optimizer.zero_grad()  # 这里清零指的是把上一次计算出来的delta(w)清零，也就是上一次的梯度清零，不是w-old清零
        loss.backward()
        optimizer.step()
        
    train_accuracy_history.append(total_train_num_correct/(len(trainloader)*batch_size))
    loss_per_epoch.append(train_loss/(len(trainloader)*batch_size))
            

    # 测试集上检验效果
    print('开始预测：')
    model.eval()

    total_eval_loss = 0
    total_num_correct = 0
    for data in testloader:
        img, label = data
        img, label = img.cuda(), label.cuda()
        out = model(img)
        # 这一个batch_size中128个图片，求得每张图平均损失就是下面的loss
        loss = loss_fn(out, label)
        total_eval_loss += loss.item() * batch_size  # Tensor.item()取出张量中的值
        # out = batch_size*各类得分。torch.max(out,0/1),0跨行，1跨列；返回值是一个元素为Tensor的二元组，(各最大值，各索引)
        _, pred = t.max(out, 1)
        num_correct = (pred == label).sum()
        total_num_correct += num_correct.item()
    
    test_accuracy_history.append(total_num_correct / (batch_size*len(testloader)))
    print('%s'%(timeSince(start, epoch / num_epoches)))

filename = open('/data/yuanfu/VGG16_bn_mnist/train_accuracy_history','w')
for i,j in enumerate(train_accuracy_history):
    filename.write(str(i)+' epoch train_acc'+': '+str(j))
    filename.write('\n')
filename.close()

filename = open('/data/yuanfu/VGG16_bn_mnist/loss_per_epoch','w')
for i,j in enumerate(loss_per_epoch):
    filename.write(str(i)+' epoch loss_avg'+': '+str(j))
    filename.write('\n')
filename.close()

filename = open('/data/yuanfu/VGG16_bn_mnist/test_accuracy_history','w')
for i,j in enumerate(test_accuracy_history):
    filename.write(str(i)+' epoch test_acc'+': '+str(j))
    filename.write('\n')
filename.close()
