import os
import torch
import sys
from torch.utils.data.dataset import Dataset
import numpy as np

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            print self.idx2word
            print self.word2idx
        return self.word2idx[word]

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, filename):
        self.dictionary = Dictionary()
        self.data = self.tokenize(filename)

    def tokenize(self, filename):
        filepath = os.path.join(filename)
        with open(filepath, 'r') as file:
            tokens = 0
            for line in file:
                # CSV Logic
                title = line.split(',')[2]
                words = title.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(filepath, 'r') as file:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in file:
                title = line.split(',')[2]
                words = title.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

# python utils/data_process.py in /training
corpus = Corpus('../data/data.example.csv')
