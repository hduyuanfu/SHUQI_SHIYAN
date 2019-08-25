import string
import random
import torch
from torch.autograd import Variable
from config import *

class Lang(object):
    def __init__(self, filename):
        self.char2idx = {}
        self.idx2char = {}
        self.n_words = 0
        self.length = self.process(filename)

    def process(self, filename):
        length = []
        with open(filename, 'r', encoding='gbk') as f:  # encoding='gbk',encoding='UTF-8'
            for line in f.readlines():
                length.append(len(line))
                words = set(line)
                comm = words & set(self.char2idx)
                for word in words:
                    if word not in comm:
                        self.char2idx[word] = self.n_words
                        self.idx2char[self.n_words] = word
                        self.n_words += 1
        return length

class Dataset(object):
    def __init__(self, filename):  # init函数在实例化时会自动执行，不需要调用
        self.lang = Lang(filename)
        #print(self.lang.idx2char)
        self.data = self.load_file(filename)

    def load_file(self, filename):
        data = []
        with open(filename, 'r', encoding='gbk') as f:
            data = f.read()
        return data

    def random_chunk(self, start, end):
        #start_idx = random.randint(0, len(self.data) - chunk_len)
        #end_idx = start_idx + chunk_len + 1
        return self.data[start:end]

    def get_variable(self, string):
        tensor = torch.zeros(len(string)).long()  # FloatTensor->LongTensor
        for c in range(len(string)):
            tensor[c] = self.lang.char2idx[string[c]]
        return Variable(tensor)

    def random_training_set(self, start, end):
        chunk = self.random_chunk(start, end)
        input = self.get_variable(chunk[:-1])
        target = self.get_variable(chunk[1:])
        return input, target