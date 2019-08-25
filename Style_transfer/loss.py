import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentLoss(nn.Module):

    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input
        '''
        (1)在类函数里参与计算的变量，必须是__init__函数中的self.变量名,(实例化时传进来的变量必须全部在init中转化为
        self变量后才能在forward等函数中应用)或者forward等函数中被赋值的变量;
        (2)为什么在forward函数中给loss变量加self呢？因为不加self的话，不能用 实例.变量名 调用，提取不出来了，而我们的
        目的就是用self.loss暂存计算结果。
        (3)什么是模型类，(nn.Module),并且super(A, self).__init__()的就是，也只有这种类才能有forward函数，才能model(参数)
        等价于model.forward(参数)
        '''

def gram_matrix(input):
    b, ch, h, w = input.size()  # b=batch size(=1)
    # ch=number of feature maps
    # (h,w)=dimensions of b
    features = input.view(b * ch, h * w)  # resise F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return torch.div(G, b * ch * h * w)  # 或者G.div(b * ch * h * w)
    '''
    torch.div(a, b) ,a和b的尺寸是广播一致的，而且a和b必须是类型一致的，就是如果a是FloatTensor那么b也必须是FloatTensor，
    可以使用tensor.to(torch.float64)进行转换。例如：a = torch.randn(4, 4)，b = torch.randn(4)，则有torch.div(a, b);
    当然也可以是torch.div(a, 2)或torch.div(b, 0.5)
    '''
class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)  # loss值当做参数保留在这里
        '''这个函数具体再求什么我还不太知道'''
        return input  # input进来利用了一下没变化又出去了
