import torch
import torch.nn as nn
import torch.nn.functional as F

import resnet_model

class Residual_Sampling(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding):
        super(Residual_Sampling, self).__init__()


        self.Maxpool1 = nn.Sequential(
            nn.MaxPool2d(kernel_size, stride=stride, padding=padding),
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False),
        )


        self.res = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, bias=False),

        )
        self.bn = nn.BatchNorm2d(out_dim)


    def forward(self, x):
        x = self.Maxpool1(x) + self.res(x)
        x = self.bn(x)
        return x

def mask_Generation(feature, alpha):
    batch_size = feature.size(0)
    kernel = feature.size(2)
    sum = torch.sum(feature.detach(), dim=1)

    avg = torch.sum(torch.sum(sum, dim=1), dim=1) / kernel ** 2

    mask = torch.where(sum > alpha * avg.view(batch_size, 1, 1), torch.ones(sum.size()).cuda(),
                       (torch.zeros(sum.size()) + 0.1).cuda())

    mask = mask.unsqueeze(1)
    return mask

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.sample_128 = Residual_Sampling(in_dim=128, out_dim=512, kernel_size=4, stride=4, padding=0)
        self.sample_256 = Residual_Sampling(in_dim=256, out_dim=512, kernel_size=2, stride=2, padding=0)

        self.fc_concat = torch.nn.Linear(512 ** 2 * 3, 200)
        self.pool_kernel4 = nn.MaxPool2d(kernel_size=4)
        self.pool_kernel2 = nn.MaxPool2d(kernel_size=2)

        self.softmax = nn.LogSoftmax(dim=1)
        self.features = resnet_model.resnet34(pretrained=True,
                                              model_root='/data/guijun/HBP_finegrained/pth/resnet34.pth')



    def extract_feature(self, feature1, feature2, feature3, batch_size):
        feature1 = self.sample_128(feature1)
        feature2 = self.sample_256(feature2)

        inter1 = (feature1 * feature2).view(batch_size, 512, 14 ** 2)
        inter2 = (feature3 * feature1).view(batch_size, 512, 14 ** 2)
        inter3 = (feature3 * feature2).view(batch_size, 512, 14 ** 2)

        inter1 = (torch.bmm(inter1, torch.transpose(inter1, 1, 2) / 14 ** 2)).view(batch_size, -1)
        inter2 = (torch.bmm(inter2, torch.transpose(inter2, 1, 2) / 14 ** 2)).view(batch_size, -1)
        inter3 = (torch.bmm(inter3, torch.transpose(inter3, 1, 2) / 14 ** 2)).view(batch_size, -1)

        result1 = torch.nn.functional.normalize(torch.sign(inter1) * torch.sqrt(torch.abs(inter1) + 1e-10))
        result2 = torch.nn.functional.normalize(torch.sign(inter2) * torch.sqrt(torch.abs(inter2) + 1e-10))
        result3 = torch.nn.functional.normalize(torch.sign(inter3) * torch.sqrt(torch.abs(inter3) + 1e-10))

        return result1, result2, result3

    def forward(self, x):
        batch_size = x.size(0)
        feature1, feature2, feature3 = self.features(x)

        slack_mask1 = mask_Generation(feature1, alpha=1.0)
        slack_mask2 = mask_Generation(feature2, alpha=0.9)
        slack_mask3 = mask_Generation(feature3, alpha=0.8)

        Aggregated_mask = self.pool_kernel4(slack_mask1) * self.pool_kernel2(slack_mask2) * slack_mask3
        Aggregated_mask_feature1 = F.interpolate(Aggregated_mask, scale_factor=4, mode='bilinear', align_corners=True)
        Aggregated_mask_feature2 = F.interpolate(Aggregated_mask, scale_factor=2, mode='bilinear', align_corners=True)

        feature1 = feature1 * Aggregated_mask_feature1
        feature2 = feature2 * Aggregated_mask_feature2
        feature3 = feature3 * Aggregated_mask

        result1, result2, result3 = self.extract_feature(feature1, feature2, feature3, batch_size)

        result = torch.cat((result1, result2, result3), 1)
        result = self.fc_concat(result)

        return self.softmax(result)
