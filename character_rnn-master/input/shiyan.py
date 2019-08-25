import numpy as np
import time
import torch
from torchvision import models


# data I/O
data = open('F:/PYTHON/character_rnn-master/input/nihao.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))  # vocabulary 词汇量
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
print(char_to_ix)
print(ix_to_char)

