# encoding:utf-8
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
from utils.utils import progress_bar
import data.data as data

import os
import torch.backends.cudnn as cudnn
from PIL import Image
import model.HQPMR_model as HQPMR
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

trainset = data.MyDataset('/data/guijun/caffe-20160312/examples/compact_bilinear/train_images_shuffle.txt', transform=transforms.Compose([
                                                transforms.Resize((600, 600), Image.BILINEAR),
                                                transforms.RandomHorizontalFlip(),

                                                transforms.RandomCrop(448),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                                ]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                          shuffle=True, num_workers=4)

testset = data.MyDataset('/data/guijun/caffe-20160312/examples/compact_bilinear/test_images_shuffle.txt', transform=transforms.Compose([
                                                transforms.Resize((600, 600), Image.BILINEAR),
                                                transforms.CenterCrop(448),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                                ]))
testloader = torch.utils.data.DataLoader(testset, batch_size=8,
                                         shuffle=False, num_workers=4)
cudnn.benchmark = True



model = HQPMR.Net()

model.cuda()


criterion = nn.NLLLoss()
# criterion = nn.CrossEntropyLoss()

lr = 1.0
model.features.requires_grad = False

optimizer = optim.SGD([
                        {'params': model.sample_128.parameters(), 'lr': lr},
                        {'params': model.sample_256.parameters(), 'lr': lr},

                        {'params': model.fc_concat.parameters(), 'lr': lr},
], lr=1, momentum=0.9, weight_decay=1e-5)


def train(epoch):
    model.train()
    print('----------------------------------------Epoch: {}----------------------------------------'.format(epoch))
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.cuda(), target.cuda()
        model.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        progress_bar(batch_idx, len(trainloader), 'loss: ' + str('{:.4f}'.format(loss.data.item())) + ' | train')


def test():
    model.eval()
    print('----------------------------------------Test---------------------------------------------')
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(testloader):
        data, target = data.cuda(), target.cuda()
        output = model(data)
        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        progress_bar(batch_idx, len(testloader), 'test')
    test_loss /= len(testloader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss * 8., correct, len(testloader.dataset),
        100.0 * float(correct) / len(testloader.dataset)))


def adjust_learning_rate(optimizer, epoch):
    if epoch % 40 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

# test()
for epoch in range(1, 81):

    train(epoch)
    if epoch % 5 == 0:
        test()
    adjust_learning_rate(optimizer, epoch)


torch.save(model.state_dict(), 'firststep_bird.pth')
