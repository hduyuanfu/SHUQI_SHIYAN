from preparedata import *
from model import *
from train import *
from evaluate import *
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 选择显卡
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(type(device))
input_lang, output_lang, pairs = prepareData('eng', 'fra', True)

# 超参数
hidden_size = 256
you = 'get'

if you == 'train':
    encoder = EncoderRNN(input_lang.n_words, hidden_size, device).to(device)
    #encoder = nn.DataParallel(encoder,device_ids=[0,1,2])
    attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, device, dropout_p=0.1).to(device)
    #attn_decoder = nn.DataParallel(attn_decoder,device_ids=[0,1,2])
    encoder, attn_decoder = trainIters(device, encoder, attn_decoder, input_lang, output_lang, 
                                        pairs, 75000, print_every=5000, plot_every=100)
    torch.save(encoder, 'encoder.pth')  # 保存实例  F:/PYTHON/RNN/encoder.pth
    torch.save(attn_decoder, 'attn_decoder.pth')
else:
    encoder = torch.load('encoder.pth')
    attn_decoder = torch.load('attn_decoder.pth')
    evaluateAndShowAttention(input_lang, output_lang, device, encoder, attn_decoder, "elle a cinq ans de moins que moi .")