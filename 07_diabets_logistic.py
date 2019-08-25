# having problem
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

train = np.loadtxt('F:/PYTHON/diabetes.csv', delimiter=',', dtype=np.float32)
#x_data = torch.from_numpy(train[:, 0:-1])
x_data = torch.from_numpy(train[:, 0:-1])
y_label = torch.from_numpy(train[:, -1])
y_label
#print(type(x_data))
#print(y_label.shape)
#print(x_data.shape)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(8, 6)
        #print(self.l1)  # Linear(in_features=8, out_features=6, bias=True)
        self.l2 = nn.Linear(6, 4)
        self.l3 = nn.Linear(4, 1)

        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.l1(x)
        out = self.Sigmoid(out)
        out = self.l2(out)
        #out = self.Sigmoid(out)
        out = self.l3(out)
        out = self.Sigmoid(out)
        return out

model = Model()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
epoch_num = 1

# training loop
for i in range(epoch_num):
    acc = 0
    y_pred = model(x_data)  # 这里出问题了
    sum = y_pred.squeeze(1)
    #print(sum)
    mask = torch.where(sum > 0.5, torch.ones(sum.size()),
                       torch.zeros(sum.size()))
    #print(mask)
    #print(y_label)
    acc = (mask == y_label).sum()
    loss = criterion(mask, y_label)
    #print(len(x_data))
    #print(acc)
    print("epoch = %d, loss = %.4f, acc = %.4f" %(i, loss.item(), acc.item()/len(x_data)))
    loss.requires_grad=True
    #print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


