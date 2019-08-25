import torch
import torch.nn as nn
 
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, 3, padding=1),    # 6@28*28
            nn.MaxPool2d(2, 2)                # 6@14*14
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),              # 16@10*10
            nn.MaxPool2d(2, 2)                # 16@5*5
        )
        self.layer3 = nn.Sequential(
            nn.Linear(400, 120),              # 16*5*5=400
            nn.Linear(120, 84),
            nn.Linear(84, 10)
        )
 
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x