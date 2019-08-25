from __future__ import print_function, division

import torch    # 导入torch相关的包
import torch.nn as nn   # 导入torch.nn中的各种模块
import torch.optim as optim  # 导入优化参数的包
from torch.autograd import Variable   # 自动求解梯度
from torch.utils.data import DataLoader, sampler  # 加载数据
from torchvision import datasets, transforms
import torchvision.models as models

batch_size = 8

# 数据预处理，将图片转换成tensor类型，并把图片中心化，缩放到[-1,1]上
data_tf = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

train_dataset = datasets.MNIST(root="./data", train=True, transform=data_tf, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=data_tf, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(range(1600)))
test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(range(1600)))


class Rebuilt_Rsnet18(nn.Module):
    def __init__(self, numclasses=10):
        super(Rebuilt_Rsnet18, self).__init__()

        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # self.conv1 = resnet18.conv1
        self.resnet18.fc = nn.Linear(in_features=512, out_features=numclasses, bias=True)
        # self.fc = resnet18.fc

    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)

        x = self.resnet18.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.resnet18.fc(x)

        return x


ResNet = Rebuilt_Rsnet18()
print(ResNet)

learning_rate = 8e-5   # 学习率
epoch = 0  # 迭代次数
train_loss = 0
train_acc = 0

criterion = nn.CrossEntropyLoss()   # 损失函数，交叉熵损失,损失函数即起衡量作用。
optimizer = optim.Adam(ResNet.parameters(), lr=learning_rate)  # 随机梯度下降优化器

for data in train_loader:   # train_loader=6000/64
    image, label = data
    img, label = Variable(image), Variable(label)  # 把image，label数据从tensor转换成Variable

    out = ResNet(img)
    loss = criterion(out, label)  # 将out和label使用交叉熵计算损失
    train_loss += loss.item() * label.size(0)
    pred = torch.max(out, 1)[1]
    train_correct = (pred == label).sum()
    train_acc += train_correct.item()

    optimizer.zero_grad()  # 将参数的梯度初始化零
    loss.backward()  # 反向传播
    optimizer.step()  # 所有optimizer优化器都包含一个step()函数，用于执行单个优化步骤，实现参数更新。
                        # optimizer.step()  ：简化版本，自动调用backward
    epoch += 1

    if epoch % 100 == 0:
        print('epoch{} loss is {:.4f}'.format(epoch, loss.item()))


print('train Loss:{:.6f}, Acc:{:.6f}'.format(train_loss / 1600, train_acc / 1600))


# 测试网络
ResNet.eval()    # eval（）基类Module的成员函数，将模型设置为evaluation模式（测试模式），
                # 只对特定的模块类型有效，如Dropout和BatchNorm等
eval_loss = 0
eval_acc = 0
epoch = 0
for data in test_loader:   # 1000/64次
    img, label = data
    img, label = Variable(img), Variable(label)

    out = ResNet(img)   # 每次同时输入8张，输出8×10的张量
    loss = criterion(out, label)
    eval_loss += loss.item() * label.size(0)    # label.size(0)=8

    _, pred = torch.max(out, 1)  # 第一个向量返回每一行的最大值，第二个向量返回最大值的下标
    num_correct = (pred == label).sum()  #共64张图片
    eval_acc += num_correct.item()

print('Test Loss:{:.6f}, Acc:{:.6f}'.format(eval_loss / 1600, eval_acc / 1600))

