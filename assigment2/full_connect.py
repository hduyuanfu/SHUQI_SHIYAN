import numpy as np
from optim import *
import matplotlib.pyplot as plt

def affine_forward(x, w, b):
    '''
    输入:
    - x: 输入数据，shape(N,特征数D)
    - w: 权重数组，shape(D,M)
    - b: 偏置，shape(M,)
    返回一个元组：
    - out: 输出，shape(N,M)
    - cache: (x,w,b)
    '''
    out = None  # 初始化
    reshaped_x = x.reshape(x.shape[0], -1)
    # 或者reshaped_x = np.reshape(x,(x.shape[0], -1))
    out = reshaped_x.dot(w) + b
    cache = (x, w, b)  # 以备后面计算梯度时使用
    return out, cache

def batch_forward(x, gamma, beta, bn_param):
    '''
    批量归一化，发生在激活函数之前。
    输入：
    - x: data of shape (N,D)
    - gamma: 一个超参数向量,scale parameter(尺度参数) of shape(D,)
    - beta: 一个超参数向量,shift parameter(移位参数) of shape(D,)
    - bn_param: 带有如下关键字的字典：
    - mode: 'train' or 'test' ;required
    - eps: constant for numeric stability(数值稳定性常数，这里为了防止除0)  
    - momentum(动量，冲力)：constant for running/variance mean(在本文中相当于一次指数平滑法中的平滑系数)
    - running_mean: array of shape (D,),用于求特征的running mean
    - running_var: array of shape (D,),giving running variance of features
    返回一个元组：
    - out: of shape (N,D)
    - cache: 在反向传播中要用到的一组值的元组
    '''

    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    # 若字典中没有这个'eps'这个关键字，则新增键值对，并将1e-5赋值给eps
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))
    # 一维数组时np.zeros(4)==np.zeros((4,))，二维时必须是后面形式

    out, cache = None, None
    if mode == 'train':  # 训练模式
        sample_mean = np.mean(x, axis=0)  # shape(D,)
        sample_var = np.var(x, axis=0)
        x_hat = (x - sample_mean) / (np.sqrt(sample_var + eps))

        out = gamma * x_hat + beta
        cache = (x, sample_mean, sample_var, x_hat, eps, gamma, beta)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

    elif mode == 'test':  # 测试模式
        out = (x - running_mean) * gamma / (np.sqrt(running_var + eps)) + beta
    else:
        raise ValueError('Invalid forward batchnorm mode: "%s" '%mode)
        
    # 把更新过后的running_mean/var装进参数字典里
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache

def relu_forward(x):
    '''
    对一层计算值进行激活
    输入: 
    - x: 输入，可以是任何形状
    输出：
    - out: 输出，和x同shape
    - cache：x
    '''
    out = np.maximum(0, x)
    cache = x  # 缓冲输入进来的x矩阵
    return out, cache

def affine_relu_forward(x, w, b):
    '''
    完成一个层间的前向传输
    输入：
    - x: 一个前向层的输入,shape(N, D1)
    - w, b: 一个层的权重(D1, D2)和偏置(D2,)
    输出：
    - out: relu的输出,of shape (N, D2)
    - cache: 用于进行反向传播的一个对象
    '''
    a, fc_cache = affine_forward(x, w, b)  # 线性模型
    out, relu_cache = relu_forward(a)  # 激活函数
    cache = (fc_cache, relu_cache)  # 缓冲的是元组：(x,w,b,(a))
    return out, cache

def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    '''
    完成一个层间的前向传输
    输入：
    - x: 一个前向层的输入,shape(N, D1)
    - w, b: 一个层的权重(D1, D2)和偏置(D2,)
    - gamma, beta: both of shape(D2,),是批量归一化中的尺度和移位参数
    - bn_param: 批量归一化的参数组成的一个字典，包括参数：mode,eps,momentum,running_mean/var
    输出：
    - out: relu的输出,of shape (N, D2)
    - cache: 用于进行反向传播的一个对象
    '''
    a, fc_cache = affine_forward(x, w, b)  # 线性模型
    a_bn, bn_cache = batch_forward(a, gamma, beta, bn_param)  # BN层，在relu层之前
    out, relu_cache = relu_forward(a_bn)  # 激活函数
    cache = (fc_cache, relu_cache, bn_cache)  # 缓冲的是元组：(x,w,b,(a))
    return out, cache

def dropout_forward(x, dropout_param):
    '''
    前向传播的dropout
    输入：
    - x: 输入数据，可以是任何shape
    - dropout_param: 有如下关键字的字典：
    - p: dropout parameter;每个神经元被drop的概率
    - mode: 'test' or 'train';训练模式下实施dropout；测试模式下返回输入
    - seed: 用于随机数的产生，传递的seed使这个函数确定性的，这个对梯度检查有用
    输出：
    - out: 和x同样shape的数组
    - cache: 一个元组(dropout_param, mask)。在训练模式中，mask is the dropout mask
    that was used to multiply the input;测试模式中，mask is None
    '''
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None
    # 训练模式
    if mode == 'train':
        keep_prob = 1 - p
        mask = (np.random.rand(*x.shape) < keep_prob) / keep_prob  # 随机真值表作为遮罩
        out = mask * x
    # 测试模式
    elif mode == 'test':
        out = x
    
    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache

def softmax_loss(z, y):
    '''
    为softmax分类计算损失和梯度
    输入：
    - z: 输入数据，shape(N,C)，其中z[i,j]表示第i个输入-第j个类别的得分
    - y: 标签向量，shape(N,),其中y[i]表示是x[i]的标签，且0<=y[i]<C
    返回一个元组：
    - loss: 一个标量损失
    - dz: gradient of the loss with respect to z
    '''
    probs = np.exp(z - np.max(z, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = z.shape[0]
    #print(np.log(probs[np.arange(N), y]))
    #print(np.log(probs[np.arange(N), y]).shape)
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dz = probs.copy()
    dz[np.arange(N), y] -= 1
    dz /= N
    return loss, dz

def dropout_backward(dout, cache):
    '''
    对dropout层进行反向传播
    输入：
    - dout: 回流的梯/散度，任何shape
    - cache:来自于dropout_forward的元组(dropout_param, mask),也就是正向传播时产生的元组
    '''
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        dx = mask * dout  # 反向传播时使用与正向同样的mask将被遮罩的梯度置零
    elif mode == 'test':
        dx = dout
    
    return dx

def affine_backward(dout, cache):
    '''
    计算一个映射层的反向传播
    输入：
    - dout: 上一层的散度输出(反向的时候右边是上一层了)
    - cache: 元组：
    - z: 输入数据，shape(N, d1,d2,···dk),例如(N, 32, 32, 3)
    - w: 权重，shape(D, M)
    - b: 偏置，shape(M,)
    输出一个元组：
    - dz: 关于z的梯度，shape与输入z保持一致
    - dw: 关于w的梯度，与输入w的shape一致
    - db: 关于b的梯度，shape与输入b的输入一致
    '''
    z, w, b = cache
    dz, dw, db = None, None, None
    reshaped_x = np.reshape(z, (z.shape[0], -1))
    dz = np.reshape(dout.dot(w.T), z.shape)
    dw = (reshaped_x.T).dot(dout) 
    db = np.sum(dout, axis=0)  # 没必要使db保持二维，一维数组即可
    return dz, dw, db


def relu_backward(dout, cache):
    '''
    对Relu层进行反向传播
    输入：
    - dout: 回流的散度(梯度),任何shape
    - cache: 输入x,x和dout有相同的shape
    返回:
    - dx: 关于x的梯度
    '''
    dx, x = None, cache
    dx = (x > 0) * dout
    # 与所有x中元素为正的位置处，位置对应于dout矩阵的元素保留，其他都取0
    return dx

def affine_relu_backward(dout, cache):
    '''affine-rule层的反向传播'''
    fc_cache, relu_cache = cache  # fc_cache=(x,w,b),relu_cache=a
    da = relu_backward(dout, relu_cache)  # da=(x>0)*relu_cache
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

def batchnorm_backward(dout, cache):
    '''
    输入：
    - dout: 回流的散度(梯度),shape(N,D)
    - cache: 来自于batchnorm_forward的中间体变量
    返回一个元组：
    - dx: 关于输入x的元组，shape(N, D)
    - dgamma: 关于尺度参数的梯度，shape(D,)
    - dbeta: 关于移动参数的梯度，shape(D,)
    '''
    x, mean, var, x_hat, eps, gamma, beta = cache
    N = x.shape[0]
    dgamma = np.sum(dout * x_hat, axis=0)  # P35第5行公式
    dbeta = np.sum(dout * 1.0, axis=0)  # 第6行公式
    dx_hat = dout * gamma  # 第1行公式
    dx_hat_numerator = dx_hat / np.sqrt(var + eps)  # 第3行第1项（未求和）
    # numerator:分子；denominator：分母
    dx_hat_denominator = np.sum(dx_hat * (x - mean), axsi=0)  # 第2行前半部分
    dx_1 = dx_hat_numerator  # 第4行第1项
    dvar = -0.5 * ((var + eps)**(-1.5)) * dx_hat_denominator  # 第2行公式
    # 请注意，方差也是一个关于均值的函数
    dmean = -1.0 * np.sum(dx_hat_numerator, axis=0) + \
            dvar * np.mean(-2.0 * (x - mean), axis=0)  # 第3行公式（利用了上面的一部分）
    dx_var = dvar * 2.0 / N * (x - mean)  # 第4行第2项
    dx_mean = dmean * 1.0 / N  # 第4行第3项
    # 利用广播，shape(D,)和shape(N, D)可以运算
    dx = dx_1 + dx_var + dx_mean  # 第4行公式

    return dx, dgamma, dbeta

def affine_bn_relu_backward(dout, cache):
    '''
    对affine-batchnorm-relu层进行反向传播
    '''
    fc_cache, bn_cache, relu_cache = cache
    # fc_cache = (x, w, b),relu_cache = a
    da_bn = relu_backward(dout, relu_cache)  # Relu层
    da, dgamma, dbeta = batchnorm_backward(da_bn, bn_cache)  # BN层，反向传播时在Relu之后
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta

class TwoLayerNet(object):  # 我们的2层全连接神经网络，函数下划线，类名用大写
    '''
    首先，先初始化我们的神经网络。
    毕竟，数据从输入层第一次流入到神经网络里，参数(W,B)不能为空，(w,b)是
    一层的参数；(W,B)是所有层参数的统一。参数初始化也不能太大或太小，因此
    (W,B)的初始化时很重要的，对整个神经网络的训练影响巨大，但如何proper
    的初始化参数还没定论。
    '''
    def __init__(self
                ,input_dims=32*32*3  # 每张样本图片的数据维度大小
                ,hidden_dims=100     # 隐藏层的神经元个数
                ,num_classes=10      # 样本图片的分类类别个数
                ,weight_scale=1e-3): # 初始化参数的权重尺度（标准偏差）
        '''
        我们把需要学习的参数(W,B)都存在self.params字典中，
        其中每个元素都是numpy array:
        '''
        self.params = {}
        '''
        我们用标准差为weight_scale的高斯分布初始化参数W,
        偏置B的初始化都为0：
        (其中randn函数是基于零均值和标准差的一个高斯分布)
        '''
        self.params['W1'] = weight_scale * np.random.randn(input_dims, hidden_dims)
        self.params['b1'] = np.zeros((hidden_dims,))
        self.params['W2'] = weight_scale * np.random.randn(hidden_dims, num_classes)
        self.params['b2'] = np.zeros((num_classes,))
        '''
        可以看到，
        隐藏层的参数矩阵行数是3*32*32(即上一层输入层的神经元个数),列数是100，
        列数是自己定的没有什么依据；
        输出层的参数矩阵行数是100(即上一层隐藏层的神经元个数),列数为10，即种类
        '''

    # 接下里，我们最后定义一个loss函数就可以完成神经网络的构造
    def loss(self, X, y):
        '''
        首先，输入的数据X是一个多维的array,shape为(样本图片的个数N * 32*32*3),
        y是与输入数据X对应的正确标签，shape为(N,)。
        我们的loss函数目标输出一个损失值loss和一个grads字典，
        其中存有loss关于隐层和输出层的参数(W,B)的梯度值：
        '''
        loss, grads = 0, {}

        # 数据X在输入层和输出层的前向传播
        h1_out, h1_cache = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        # h1_out=rule(W1*X+b1),h1_cache=(X,W1,b1,(W1*X+b1))
        scores, out_cache = affine_forward(h1_out, self.params['W2'], self.params['b2'])
        # scores.shape=(N,C),N:样本数；C:分类数。out_cache=(h1_out,W2,b2)

        # 输出层后，结合正确标签y得出损失值和其在输出层的梯度：
        loss, dout = softmax_loss(scores, y)

        # 损失值loss的梯度在输出层和输入层间的反向传播：
        dout, dw2, db2 = affine_backward(dout, out_cache)
        grads['W2'] = dw2, grads['b2'] = db2
        _, dw1, db1 = affine_relu_backward(dout, h1_cache)
        grads['W1'], grads['b1'] = dw1, db1
        '''
        可以看到图片样本的数据梯度dout只起到了带路的作用,
        最终会被舍弃掉，我们只要loss关于参数的梯度，
        然后保存在字典里。
        '''
        return loss, grads


class FullyConnectedNet(object):
    '''
    一个任意隐藏层数和神经元数的全连接神经网络，其中Relu激活函数，softmax损失函数，
    同时可选的采用dropout和batch normalization(批量归一化)。那么，对于一个L层的
    神经网络来说，其框架是：

    {affine - [batch norm] - relu - [dropout]} * (L-1) - affine - softmax

    其中的[batch norm]和[dropout]是可选非必须的，框架中{···}部分会重复L-1次，代表
    L-1个隐藏层

    与我们在上面定义的TwoLayrtNet()类保持一致，所有待学习的参数都会存在self.param
    字典中，并且最终会被最优化Solver()类训练学习得到
    '''
    # 第一步是初始化我们的FullyConnectedNet()类：
    def __init__(self
                ,hidden_dims  # 一个列表，元素个数是隐藏层数，元素值为该层神经元数
                ,input_dim=28*28  # 输入默认神经元个数是3072个(匹配CIFAR-10)
                ,num_classes=10  # 默认输出神经元个数是10(匹配CIFAR-10)
                ,dropout=0  # 默认不开启dropout,若取(0,1)则表示失活概率
                ,use_batchnorm=False  # 默认不开启批量归一化，开启则为True
                ,reg=0.0  # 默认无L2正则化，取某scalar表示正则化强度
                ,weight_scale=1e-2  # 默认0.01，表示权重参数初始化的标准差
                ,dtype=np.float64  # 默认np.float64精度，要求所有计算都应在此精度下
                ,seed=None):  # 默认无随机种子，若有会传递给dropout层
        
        # 实例(instance)中增加变量并赋予初值，以方便后面的loss()函数调用:
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0  # 可见，若dropout为0，为False
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)  # 在loss函数里，用神经网络的层数来标记规模
        self.dtype = dtype
        self.params = {}  # self.params空字典保存待训练学习的参数

        # 定义所有隐藏层的参数到字典self.params中：
        in_dim = input_dim  # eg：in_dim = 0
        for i, h_dim in enumerate(hidden_dims):  # ed:(i,h_dim)=(0,H1)、(1,H2)...
            # eg: W1(D,H1),W2(H1,H2)...   ,小随机数为初始值。
            self.params['W%d'%(i+1,)] = weight_scale * np.random.randn(in_dim, h_dim)
            # eg: b1(H1,),b2(H2)...    ,0为初始值
            self.params['b%d'%(i+1)] = np.zeros((h_dim,))
            if use_batchnorm:
                # eg:gamma1(H1,),gamma2(H2)...   ,1为初始值
                # eg:beta1(H1,),beta2(H2,)...   ,0为初始值
                self.params['gamma%d'%(i+1,)] = np.zeros((h_dim,))
                self.params['beta%d'%(i+1)] = np.zeros((h_dim,))
            in_dim = h_dim  # 将该隐藏层的列数传递给下一层的行数
        
        # 定义输出层的参数到字典self.params中:
        self.params['W%d'%(self.num_layers,)] = weight_scale * np.random.randn(in_dim, num_classes)
        self.params['b%d'%(self.num_layers,)] = np.zeros((num_classes,))

        '''
        当开启dropout时，我们需要在每个神经元层中传递一个相同dropout参数字典self.dropout_param,
        以保证每一层的神经元们都知晓失活概率p和当前网络的模式状态mode(训练/测试)
        '''
        self.dropout_param = {}  # dropout的参数字典
        if self.use_dropout:  # 如果use_dropout的值是(0,1),即启用dropout
            # 设置mode默认为训练模式，取p为失活概率
            self.dropout_param = {'mode': 'train', 'p': dropout}
        
        if seed is not None:  # 如果有随机种子，存入seed
            self.dropout_param['seed'] = seed
            '''这个种子有什么用啊'''
        
        '''
        当开启批量归一化时，我们要定义1一个BN算法的参数列表self.bn_params,以用来跟踪记录
        每一层的平均值和标准差，其中，第0个参数self.bn_params[0]表示前向传播第1个BN层的
        参数，第1个元素self.bn_params[1]表示前向传播第2个BN层的参数，以此类推
        '''
        self.bn_params = []  # BN算法的参数列表
        if self.use_batchnorm:  # 如果开启批量归一化，设置每层mode默认为训练模式
            self.bn_params = [{'mode':'train'} for i in range(self.num_layers - 1)]
            # 上面self.bn_params列表的元素个数是hidden_layers的个数

        # 最后，调整参数字典self.params中所有待学习神经网络参数为指定计算精度:np.float64
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
        
    # 第二步是定义我们的损失函数
    def loss(self, X, y=None):
        '''
        和TwoLayerNet()一样：
        首先，输入的数据X是一个多维的array,shape为(样本图片的个数N * 3*32*32)，
        y是与输入数据X对应的正确标签，shape为(N,)。
        # 在训练模式下：#
        我们loss函数目标输出一个损失值loss和一个grads字典，其中存有loss关于隐藏层
        和输出层的参数(W,B,gamma,beta)的梯度值。
        # 在测试模式下：#
        我们的loss函数只需要直接给出输出层后的得分即可。
        '''
        # 把输入数据源矩阵X的精度调整一下
        X = X.astype(self.dtype)
        # 根据正确标签y是否为None来调整模式是test还是train
        mode = 'test' if y is None else 'train'

        '''
        确定了当前神经网络所处的模式状态后，
        就可以设置dropout的参数字典和BN算法的参数列表中的mode了，
        因为他们在不同的模式下行为是不同的。
        '''
        if self.dropout_param is not None:  # 如果开启dropout
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:  # 如果开启批量归一化
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        
        scores = None

        '''
        %前向传播%
        如果开启了dropout,我们需要将dropout的参数字典self.dropout_param在每一个
        dropout层中传递。
        如果开启了批量归一化，我们需要指定BN算法的参数列表self.bn_params[0]对应前
        向传播第一层的参数，self.bn_params[1]对应第二层参数，以此类推。
        '''
        fc_mix_cache = {}  # 初始化每层前向传播的缓冲字典
        if self.use_dropout:  # 如果开启了dropout,初始化其对应的缓冲字典
            dp_cache = {}
        
        # 从第一个隐藏层开始循环每一个隐藏层，传递数据out，保存每一层的缓冲cache
        out = X
        for i in range(self.num_layers - 1):  # 在每个hidden层中循环
            w, b = self.params['W%d'%(i+1)], self.params['b%d'%(i+1)]
            if self.use_batchnorm:  # 若开启批量归一化
                gamma = self.params['gamma%d'%(i+1)]
                beta = self.params['beta%d'%(i+1)]
                out, fc_mix_cache[i] = affine_bn_relu_forward(out, w, b, 
                                        gamma, beta, self.bn_params[i])
            else:  # 若未开启批量归一化
                out, fc_mix_cache[i] = affine_relu_forward(out, w, b)
            if self.use_dropout:  # 若开启dropout
                out, dp_cache[i] = dropout_forward(out, self.dropout_param)
        # 最后的输出层
        w = self.params['W%d'%(self.num_layers,)]
        b = self.params['b%d'%(self.num_layers)]
        out, out_cache = affine_forward(out, w, b)
        scores = out
        
        '''
        可以看到，上面对隐藏层的每次循环中，out变量实现了自我迭代更新：
        fc_mix_cache缓冲字典中顺序地存储了每个隐藏层的得分情况和模型参数(其中可包含BN层)；
        dp_cache缓冲字典中单独顺序地保存了每个dropout层的失活概率和遮罩；
        out_cache变量缓存了输出层处的信息；
        值得留意的是，若开启批量归一化的话，BN层的参数列表self.bn_params[i],
        从第一层开始多出'running_mean'和'running_var'的键值对保存在列表的每一个元素中，
        形如：[{'mode':'train','running_mean':***,'running_var':***}, {...}]
        '''
         # 接下来开始让loss函数区分不同模式：
        if mode == 'test':  # 若是测试模式，输出scores表示的预测的每个分类概率后，函数停止跳出。
            return scores
        
        '''
        %反向传播%
        既然运行到了这里，说明我们的神经网络是在训练模式下了，接下来我们要计算损失值，并且
        通过反向传播，计算损失函数关于模型参数的梯度！
        '''
        loss, grads = 0.0, {}  # 初始化loss变量和梯度字典grads
        loss, dout = softmax_loss(scores, y)
        loss += 0.5 * self.reg * np.sum(self.params['W%d'%(self.num_layers)]**2)
        '''
        你可能奇怪上面的loss损失值是不是有问题，还有其他隐藏层的权重矩阵的正则化呢？
        别着急，我们要loss损失值的求解，跟随梯度的反向传播一点一点的算出来~
        '''
        # 在输出层处梯度的反向传播，顺便把梯度保存在梯度字典grads中:
        dout, dw, db = affine_backward(dout, out_cache)
        grads['W%d'%(self.num_layers)] = dw + self.reg * self.params['W%d'%(self.num_layers)]
        grads['b%d'%(self.num_layers)] = db

        '''
        在每一个隐藏层处梯度的反向传播，不仅顺便更新了梯队字典grads,
        还迭代算出了损失值loss:
        '''
        for i in range(self.num_layers - 1):
            ri = self.num_layers - 2 - i  # 倒数第ri+1隐藏层
            # 迭代地补上每层的正则项给loss
            loss += 0.5 * self.reg * np.sum(self.params['W%d'%(ri+1)]**2)

            if self.use_dropout:  # 若开启dropout
                dout = dropout_backward(dout, dp_cache[ri])

            if self.use_batchnorm:  # 若开启批量归一化
                dout, dw, db, dgamma, dbeta = affine_bn_relu_backward(dout, fc_mix_cache[ri])
                grads['gamma%d'%(ri+1)] = dgamma
                grads['beta%d'%(ri+1)] = dbeta
            else:  # 若未开启批量归一化
                dout, dw, db = affine_relu_backward(dout, fc_mix_cache[ri])
            
            grads['W%d'%(ri+1)] = dw + self.reg * self.params['W%d'%(ri+1)]
            grads['b%d'%(ri+1)] = db

        return loss, grads


class Solver(object):
    '''
    我们定义的这个Solver类将会根据我们的神经网络模型框架——FullyConnectedNet()类，
    在数据源的训练集部分和验证集部分中，训练我们的模型，并且通过周期性的检查准确率的
    方式，以避免过拟合。

    在这个类中，包括__init__()，共定义5个函数，其中只有train()函数时最重要的，调用
    他后，会自动启动神经网络模型优化程序。

    训练结束后，经过更新在验证集上优化的模型参数会保存在model.params中。此外，损失值
    的历史训练信息会保存在solver.loss_history中，还有solver.train_acc_history和
    solver.val_acc_history中会分别保存训练集和验证集在每一次epoch时的模型准确率。
    '''
    
    '''
    # 神经网络模型中必须要有两个函数方法:模型参数model.params和损失函数model.loss(X,y)
    一个Solver类工作在一个模型对象上，这个模型必须符合如下API：
    - model.params必须是个字典，a dictionary mapping string parameter names to
    numpy arrays containing parameter values.
    -model.loss(X,y)是一个函数，计算training-time loss and gradients,and test-time
     classification scores,with the following inputs and outputs:
    输入:  #全局的输入变量
    - X: array giving a minibatch of input data of shape(N,d1,d2,...dk)
    - y: 标签数组，shape(N,),giving labels for X where y[i] is the label for X[i]
    返回：  # 全局的输出变量
    # 用标签用的存在与否标记训练模型还是测试模型，如果y is None,运行测试forward，
    然后返回：
    - scores: array of shape (N,C)
    如果 y is not None,运行一个training-time forward and backward pass and return
    a tuple of:
    - loss: 一个标量损失函数值
    - grads: 一个字典，与self.params有相同关键字；表征模型梯度
    '''
    #1# 初始化我们的Slover()类
    def __init__(self, model, data, **kwargs):
        '''
        构造一个新的Solver实例
        # 必须要输入的函数参数：模型和数据
        required arguments:
        - model: 一个模型对象
        - data: 一个关于训练和验证数据的字典，如下：
            'X_train': 
            'X_val':
            'y_train':
            'y_val':
        # 可选的输入参数:
        optional arguments:
         # 优化算法：默认为sgd
        - update_rule: a string giving the name of an update rule in optim.py,
        default is 'sgd'
         # 设置优化算法的超参数：
        - optim_config: 一个字典，包含超参数，超参数将被传递给用来选择更新规则；
        每个更新规则要求不同的超参数(see optim.py),但是所有的更新规则都需要一个
        'learning_rate'参数.
        - lr_decay: 一个关于学习率衰减的标量，每迭代一次学习率都应乘以这个衰减率一次
         # 在训练时，模型输入层接受样本图片的个数，默认为100
        - batch_size: size of minibatchs used to compute loss and gradient furing training
         # 在训练时，让神经网络模型一次全套训练的遍数
        - num_epochs: 一次训练时迭代次数
         # 在训练时，打印损失值的迭代次数
        - print_every: 整数，每迭代这些次打印训练损失
         # 是否在训练时输出训练过程
        - verbose: 布尔型。
        '''
        # 实例中增加变量并赋予初值，以方便后面的train()函数等调用：
        self.model = model     # 模型
        self.X_train = data['X_train']   # 训练样本图片数据
        self.y_train = data['y_train']   # 训练样本图片的标签
        self.X_val, self.y_val = data['X_val'], data['y_val']  # 验证样本数据和标签

        '''以下是可选择输入的类参数，逐渐一个一个剪切打包kwargs参数列表'''
        self.update_rule = kwargs.pop('update_rule', 'sgd')  # 默认优化算法sgd
        self.optim_config = kwargs.pop('optim_config', {})  # 默认设置优化算法超参数为空字典
        self.lr_decay = kwargs.pop('lr_decay', 1.0)  # 默认学习率不衰减
        self.num_epochs = kwargs.pop('num_epochs', 10)  # 默认神经网络训练十遍
        self.batch_size = kwargs.pop('batch_size', 100)  # 默认一次输入100张图片
        self.print_every = kwargs.pop('print_every', 10)  # 没十次打印
        self.verbose = kwargs.pop('verbose', True)  # 默认打印训练的中间过程
        '''异常处理：如果kwargs参数列表中除了上述元素外还有其他的就报错！'''
        if len(kwargs) > 0:
            extra = ', '.join('"%s"'% k for k in kwargs.keys())
            raise ValueError('Unrecongnized arguments %s'%extra)
        '''
        异常处理：如果kwargs参数列表中没有优化算法，就报错！
        将self.update_rule转化为优化算法的函数，即：
        self.update_rule(w, dw, config)=(next_w, config)
        '''
        '''
        if not hasattr(self.optim_config, self.update_rule):  # 若optim.py中没有写好的优化算法对应
            raise ValueError('Invalid update_rule "%s"'%self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)
        '''
        # 执行_reset()函数
        self._reset()

    #2# 定义我们的_reset()函数，其仅在类初始化函数__init__()中调用
    def _reset(self):
        '''重置一些用于记录优化的变量'''
        # set up some variables for  book_keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        # make a deep copy of the optim_config for each paramwter
        self.optim_configs = {}
        for p in self.model.params:
            d = {k:v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d
        '''
        上面根据模型中待学习的参数，创建了新的优化字典self.optim_configs.
        形如：{'b':{'learning_rate':0.0005},'w':{'learning_rate':0.0005}},
        为每个模型参数指定了相同的超参数。
        '''
    
    #3# 定义_step()函数，其仅在train()函数中调用
    def _step(self):
        '''训练模式下，样本图片数据的一次正向和反向传播，并且更新模型参数一次'''
        # make a minibatch of training data  # 输入数据准备
        num_train = self.X_train.shape[0]  # 要训练的数据集总数
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]  # 随机取得待输入神经元的样本图片数据
        y_batch = self.y_train[batch_mask]  # 随机取得待输入神经元的样本图片标签

        # 计算数据通过神经网络后得到的损失值和梯度字典
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)  # 把本次算的的损失值记录下来

        # 执行一次模型参数的更新
        for p, w in self.model.params.items():
            dw = grads[p]  # 取出模型参数p对应的梯度值
            config = self.optim_configs[p]  # 取出模型参数p对应的优化超参数
            next_w, next_config = sgd_momentum(w, dw, config)#self.update_rule(w, dw, config) # 优化算法
            self.model.params[p] = next_w  # 新参数替换旧的
            self.optim_configs[p] = next_config  # 新超参数替换旧的，如动量v 
    
    #4# 定义check_accuracy函数，其仅在train()函数中调用
    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        '''
        根据某图片样本数据，计算某与之对应的标签的准确率
        输入：
        - X: 一个数据数组，shape(N, d1, d2,...dk)
        - y:标签数组，shape(N,)
        - num_samples: if not none,对数据进行子集采样，
        仅在num_samples数据点上测试模型 
        - batch_size: split X, y into batchs of this size to avoid using 
        too much memory.
        返回：
        - acc: 模型实例的准确率，一个标量
        '''
        # maybe subsample the data
        N = X.shape[0]   # 样本图片X的总数
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # 计算预测 in batches
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)
        return acc

    #5# 定义最重要的train()函数
    def train(self):
        '''首先要确定下来总共要进行的迭代的次数num_iteraations，'''
        num_train = self.X_train.shape[0]  # 全部要用来训练的样本图片总数
        iterations_per_epoch = max(num_train // self.batch_size, 1)# 每遍迭代的次数
        num_iterations = self.num_epochs * iterations_per_epoch # 总迭代次数

        '''开始循环迭代'''
        for t in range(num_iterations):
            self._step()
            '''
            上面完成了一次神经网络的迭代。此时，模型的参数已经更新过一次，
            并且在self.loss_history中添加了一个新的loss值
            '''
            # maybe print trainging loss 从self.loss_history中取最新的loss值
            if self.verbose and t % self.print_every == 0:
                print('(Iteration %d / %d) loss : %f'%(t+1,num_iterations,self.loss_history[-1])) 
            # 每次epoch结束，增加epoch值，并衰减学习率
            epoch_end = (t+1) % iterations_per_epoch == 0
            if epoch_end:  # 只有当t==iterations_per_epoch-1时为True
                self.epoch += 1  # 从第一遍开始，从0自家1为每遍计数
                for k in self.optim_configs:  # 第一遍之后开始，每遍给学习率自乘一个衰减率
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay
            
            # 在第一次和最后一次iteration,以及每个epoch的最后，检查训练和验证准确率
            first_it = (t == 0)  # 起始的t
            last_it =(t == num_iterations - 1)  # 最后的t
            if first_it or last_it or epoch_end:  # 在最开始/最后/每遍结束时
                train_acc = self.check_accuracy(self.X_train, self.y_train,
                            num_samples=1000)  # 随机取1000个训练图看准确率
                val_acc = self.check_accuracy(self.X_val, self.y_val)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)

                if self.verbose:  # 在最开始/最后/每遍结束时，打印准确率等信息
                    print('(Epoch %d / %d) train_acc : %f; val_acc : %f'%(
                            self.epoch, self.num_epochs, train_acc, val_acc))

                # 在最开始/最后/每遍结束时，比较当前验证集的准确率和过往最佳验证集
                # keep track of the best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()  # copy()仅复制值过来
        '''
        结束循环迭代
        '''
        self.model.params = self.best_params  # 最后把得到的最佳模型参数存入到模型中
        plt.plot(np.array(self.train_acc_history))
        plt.plot(np.array(self.val_acc_history))
        plt.show()
    