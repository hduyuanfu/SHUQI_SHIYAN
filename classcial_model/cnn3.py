import read_mnist as raw
import torch, random, copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as OPT
import numpy as np
import torch.utils.data as Data

# print(raw.get_train_img().shape)
# print(raw.get_train_label().shape)
# print(raw.get_val_img().shape)
# print(raw.get_val_label().shape)

class HyperParams:
    def __init__(self):
        self.LR = 0.0001
        self.BS = 64

        self.RUN = 'train'

    def check(self):
        assert self.RUN in ['train', 'val'], 'You should set in train and val'

    def __str__(self):

        return '123'

# __C = HyperParams()
#
# class Dataset:
#     def __init__(self, __C):
#         self.__C = __C
#
#         if __C.RUN in ['train']:
#             self.imgs = raw.get_train_img()
#             self.labels = raw.get_train_label()
#         else:
#             self.imgs = raw.get_val_img()
#             self.labels = raw.get_val_label()
#
#         self.index_list = self.sampler(__C.RUN in ['train'])
#         self.global_index = 0
#
#     def getitem(self, index):
#         img = self.imgs[index].reshape([1, 28, 28])
#         label = self.labels[index].reshape([1])
#
#         # print(img.shape)
#         # print(label.shape)
#         # (1, 28, 28)
#         # (1,)
#
#         return torch.from_numpy(img), torch.from_numpy(label)
#
#     def getbatch(self):
#         if self.global_index + self.__C.BS > self.imgs.shape[0]:
#             self.global_index = 0
#             self.index_list = self.sampler(self.__C.RUN in ['train'])
#             return None, None
#
#         img_batch = []
#         label_batch = []
#
#         for ix in range(self.__C.BS):
#             real_index = self.index_list[self.global_index + ix]
#             img, label = self.getitem(real_index)
#             img_batch.append(img.unsqueeze(0))
#             label_batch.append(label.unsqueeze(0))
#
#         img_batch = torch.cat(img_batch, dim=0)
#         label_batch = torch.cat(label_batch, dim=0)
#
#         # print(img_batch.size(), label_batch.size())
#
#         self.global_index += self.__C.BS
#
#         return img_batch, label_batch
#
#
#     def sampler(self, rand=False):
#         index_list = list(range(0, self.imgs.shape[0]))
#         if rand:
#             random.shuffle(index_list)
#         return index_list


class Dataset(Data.Dataset):
    def __init__(self, __C):
        self.__C = __C

        if __C.RUN in ['train']:
            self.imgs = raw.get_train_img()
            self.labels = raw.get_train_label()
        else:
            self.imgs = raw.get_val_img()
            self.labels = raw.get_val_label()

    def __getitem__(self, index):
        img = self.imgs[index].reshape([1, 28, 28])
        label = self.labels[index].reshape([1])

        # print(img.shape)
        # print(label.shape)
        # (1, 28, 28)
        # (1,)

        return torch.from_numpy(img), torch.from_numpy(label)

    def __len__(self):
        return self.imgs.shape[0]

# dataset = Dataset(__C)
#
# while True:
#     img_batch, label_batch = dataset.getbatch()
#     if img_batch is None:
#         break
#     print(dataset.global_index)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.pool3 = nn.AvgPool2d(kernel_size=(7, 7))

        self.dropout1 = nn.Dropout2d(0.3)
        self.dropout2 = nn.Dropout2d(0.3)

        self.fc1 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(self.dropout1(F.relu(self.conv2(x))))
        x = F.relu(self.conv3(x))
        x = self.pool2(self.dropout2(F.relu(self.conv4(x))))
        x = F.relu(self.conv5(x))
        x = self.pool3(self.conv6(x)).view(-1, 256)
        x = self.fc1(x)

        return x





class Execution:
    def __init__(self, __C):
        self.__C = __C
        self.dataset_train = Dataset(__C)
        __C_eval = copy.deepcopy(__C)
        __C_eval.RUN = 'val'
        self.dataset_eval = Dataset(__C_eval)

    def train(self):
        net = Net()
        net.train()
        net.cuda()

        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        optim = OPT.Adam(net.parameters(), lr=self.__C.LR)
        dataloader = Data.DataLoader(self.dataset_train, batch_size=self.__C.BS, shuffle=True, num_workers=8)
        for epoch in range(0, 10000):
            for step, (img_batch, label_batch) in enumerate(dataloader):
                optim.zero_grad()

                img_batch = img_batch.cuda()
                label_batch = label_batch.cuda().view(-1)
                pred = net(img_batch)

                loss = loss_fn(pred, label_batch)
                # print(loss)

                loss.backward()
                optim.step()

            self.eval(net.state_dict())

    def eval(self, state_dict):
        net = Net()
        net.eval()
        net.cuda()
        net.load_state_dict(state_dict)
        correct = 0

        dataloader = Data.DataLoader(self.dataset_eval, batch_size=64, shuffle=False, num_workers=8)
        for step, (img_batch, label_batch) in enumerate(dataloader):

            img_batch = img_batch.cuda()
            pred = net(img_batch)
            # print(pred.size())

            pred_np = pred.cpu().detach().numpy()
            pred_np_argmax = np.argmax(pred_np, axis=-1)
            label_batch_np = label_batch.view(-1).numpy()
            correct += np.sum((pred_np_argmax == label_batch_np).astype(np.float32))

        print(correct/10000)


__C = HyperParams()
execu = Execution(__C)
execu.train()