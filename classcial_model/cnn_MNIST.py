'''
使用torchvision加载并预处理MNIST数据集
定义网络
实例化网络，定义损失函数和优化器
训练网络并更新网络参数
测试网络
'''
import torch as t
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 超参数
batch_size = 128
learning_rate = 1e-2
num_epoches = 5


'''处理数据集'''
# 定义对数据的预处理
# Compose,将好几个变换组合在一起(下面读取数据时，需要将图片数据转化为Tensor才能在框架跑，想可视化图片则又需要变回去)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
# 图片为几通道，则有几个数[0.5],三通道则[0.5, 0.5, 0.5];中括号外再加个小括号可以([0.5, 0.5, 0.5]);
# 但只是(0.5, 0.5, 0.5)用小括号要报错的
# 获取训练集
trainset = datasets.MNIST(root='F:/shujuji/', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)  # 不要用多线程
# 测试集
testset = datasets.MNIST(root='F:/shujuji/', train=False,  transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

'''定义网络模型(动态图)'''
class Net(nn.Module):
    def __init__(self):  #子类重新定义__init__
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        #下式等价于nn.Module.__init__(self)
        super(Net, self).__init__()  # super()实现父类与子类的关联
        
        self.layer1 = nn.Sequential(
            # 一个特征一个维度，对应一个通道
            nn.Conv2d(1, 6, 3, padding=1),  # 得到6@28*28
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 得到6@14*14
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),  # 得到16@10*10
            nn.MaxPool2d(2, 2)  # 得到16@5*5
        )
        self.layer3 = nn.Sequential(
            nn.Linear(400, 120),  # 得到16*5*5=400
            nn.Linear(120, 84),
            nn.Linear(84, 10)
        )
       
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)  # view == reshape
        x = self.layer3(x)
        return x
    
'''实例化模型，定义损失函数和优化器'''
# 模型实例化
model = Net()
# 定义该实例模型的loss函数和其优化方法
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

'''训练模型，测试模型'''
for epoch in range(num_epoches):
    # 一趟训练
    model.train()
    for data in trainloader:  # 每次取一个batch_size
        img, label = data  # img.size:128*1*28*28
        output = model(img)
        loss = loss_fn(output, label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # 测试集上检验效果
    model.eval()  # 切换模式
    total_eval_loss = 0
    total_num_correct = 0
    for data in testloader:
        img, label = data
        out = model(img)
        print(out)
        print(label)
        # 这一个batch_size中128个图片，求得每张图平均损失就是下面的loss
        loss = loss_fn(out, label)
        total_eval_loss += loss.item() * label.size(0)  # Tensor.item()取出张量中打的值
        # out = batch_size*各类得分。torch.max(out,0/1),0跨行，1跨列；返回值是一个元素为Tensor的二元组，(各最大值，各索引)
        _, pred = t.max(out, 1)
        num_correct = (pred == label).sum()
        total_num_correct += num_correct.item()
    # 输出的其实相当于每张图片的平均loss
    print('epoch: %d; test_loss: %.3f; test_acc: %.3f'%(epoch, total_eval_loss / len(testset), total_num_correct / len(testset)))