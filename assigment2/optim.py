import numpy as np
def sgd(w, dw, config=None):
    '''
    随机梯度下降
    config format:
    - learning_rate: 一个标量学习率
    '''
    if config is None:config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config

def sgd_momentum(w, dw, config=None):
    '''
    带动量的随机梯度下降法
    config format:
    - learning_rate: 一个标量学习率
    - momentum: 位于(0,1)的标量，代表动量值
    - velocity: 一个numpy数组，与w,dw有相同shape,
    used to store a  moving average of the gradients。
    '''
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))

    next_w = None
    v = config['momentum'] * v - config['learning_rate'] * dw
    next_w = w + v
    config['velocity'] = v

    return next_w, config