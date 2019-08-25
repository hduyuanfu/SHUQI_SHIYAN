import os
import string
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from config import *
from model import RNN
from dataset import Dataset
import numpy as np

def evaluate(model, dataset, prime_str='陈志', predict_len=50, temperature=0.8):
    hidden = model.init_hidden()
    prime_input = dataset.get_variable(prime_str)
    predicted = prime_str
    for p in range(len(prime_str)-1):
        _, hidden = model(prime_input[p], hidden)
    input = prime_input[-1]

    for p in range(predict_len):
        output, hidden = model(input, hidden)  # output用于softmax,是输出字符与字典向量中所有字的关系紧密程度
        # print(len(output.detach().numpy().squeeze(0)))
        # 多项分布随机采样
        # exp()保证各项均为正数
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0] # int
        #index = np.argmax(output.data.view(-1))
        #print(index)
        # 拼接预测出的字符
        predicted_char = dataset.lang.idx2char[top_i.item()] #top_i.item()
        predicted += predicted_char
        input = dataset.get_variable(predicted_char)
    return predicted    

def train(model, optimizer, loss_fn, dataset, lines_length):
    start = time.time()
    PRINT_EVERY = 2
    #loss_avg = 0
    #start_index = 0
    #end_index = 0
    for j in range(1):
        loss_avg = 0
        start_index = 0
        end_index = 0
        for i in range(1, len(lines_length)): #len(lines_length)
            start_index += lines_length[i-1]
            end_index = start_index + lines_length[i]-1
            input, target = dataset.random_training_set(start_index, end_index)
            hidden = model.init_hidden()  #h0
            optimizer.zero_grad()
            loss = 0
            for c in range(lines_length[i]-2):
                output, hidden = model(input[c], hidden)#model.forward
                #print(len(output.detach().numpy().squeeze(0)))
                #index = np.argmin(np.fabs(output.detach().numpy().squeeze(0)))
                #print(np.fabs(output.detach().numpy().squeeze(0)))
                #print(output)
                #print(target[c])
                #print(target[c].unsqueeze(0))
                loss += loss_fn(output, target[c].unsqueeze(0))
                #loss += loss_fn(output.detach().squeeze(0), target[c])
                #print(loss)
            loss.backward()
            optimizer.step()
            each_loss_avg = loss.item() / lines_length[i]  # loss.data[0] / CHUNK_LEN
            loss_avg += each_loss_avg
            if i % PRINT_EVERY == 0:
                now = time.time()
                print('[耗时%s; 第%d epoch; 前%d行已经训练; 已完成%d%%训练任务; 平均损失: %.4f]'%(now-start, j, 
                        i, (i+j*len(lines_length))/(100*len(lines_length))*100, each_loss_avg))
                print(evaluate(model, dataset, '如', predict_len=150),'\n')
                #save_model(model, epoch)
    time1 = time.time()
    torch.save(model, 'F:/PYTHON/PY_song/model.pth')  # './model.pth'
    time2 = time.time()
    print('保存模型用时：%f seconds'%(time2-time1))


    
def generate(dataset, prime_str, predict_len, youwant=1):
    time1 = time.time()
    load_model=torch.load('F:/PYTHON/PY_song/model.pth')
    time2 = time.time()
    print('加载模型用时：%f seconds'%(time2-time1))
    for i in range(youwant):
        print(evaluate(load_model, dataset, prime_str, predict_len),'\n')

def main(you='train'):
    path = 'F:/PYTHON/TF_song/ha.txt'
    dataset = Dataset(path)
    HIDDEN_SIZE = 128
    N_LAYERS = 1
    LEARNING_RATE = 0.001
    model = RNN(dataset.lang.n_words, HIDDEN_SIZE, dataset.lang.n_words, N_LAYERS)  
    # model, start_epoch = load_previous_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    if you == 'train':
        train(model, optimizer, loss_fn, dataset, dataset.lang.length)
    else:
        generate(dataset, '如', predict_len=150, youwant=3)
if __name__ == '__main__':
    main(you='train')

