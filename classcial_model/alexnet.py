import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, sampler
import torch.optim as optim
from torch.autograd import Variable



class Rebuilt_alexnet(nn.Module):
    def __init__(self, num_classes=10):
        super(Rebuilt_alexnet, self).__init__()

        alexnet = models.alexnet(pretrained=True)
        alexnet.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        self.features = alexnet.features

        alexnet.classifier = nn.Sequential()
        self.classifier = nn.Sequential(
            nn.Linear(9216, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


alexnet = Rebuilt_alexnet()
print(alexnet)

batch_size = 8
learning_rate = 8e-5

# 数据预处理，将图片转换成tensor类型，并把图片中心化，缩放到[-1,1]上
data_tf = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

train_dataset = datasets.MNIST(root="./data", train=True, transform=data_tf, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=data_tf, download=True)
# Dataloader(
#         dataset = Dataset- 从中​​加载数据的数据集。
#         batch_size
#         shuffle = 是否打乱顺序，默认为否
# 将60000份图片划分为60000/64份，每份64个，用于mini-batch输入
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(range(1600)))
test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(range(1600)), shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(alexnet.parameters(), lr=learning_rate)

epoch = 0
train_loss = 0
train_acc = 0

for data in train_loader:
    image, label = data
    image, label = Variable(image), Variable(label)

    out = alexnet(image)   # 1,前向传播输出output
    loss = criterion(out, label)   # 2,求损失
    train_loss += loss.item() * label.size(0)
    _, pred = torch.max(out, 1)
    train_correct = torch.sum(label == pred)
    train_acc += train_correct.item()
    optimizer.zero_grad()
    loss.backward()   # 3,反向传播
    optimizer.step()    # 4, 求梯度

    epoch += 1
    if epoch % 50 == 0:
        print('epoch{} loss is {:.4f}'.format(epoch, loss.item()))

print('train loss is {:.6f}, acc is {:.6f}'.format(train_loss/(1600), train_acc/(1600)))


alexnet.eval()
epoch = 0
test_loss = 0
test_acc = 0

for data in test_loader:
    image, label = data
    image, label = Variable(image), Variable(label)

    out = alexnet(image)
    loss = criterion(out, label)
    test_loss += loss.item() * label.size(0)
    _, pred = torch.max(out, 1)
    test_correct = torch.sum(pred==label)
    test_acc += test_correct.item()

    epoch += 1
    if epoch % 50 == 0:
        print('epoch{} loss is {:.4f}'.format(epoch, loss.item()))

print('test loss is {:.6f}, acc is {:.6f}'.format(test_loss / (1600), test_acc / (1600)))






