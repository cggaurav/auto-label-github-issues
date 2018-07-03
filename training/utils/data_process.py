import os
import torch
import sys
from torch.utils.data.dataset import Dataset
import numpy as np

LABELS = []
with open('../data/data.labels.csv', 'r') as file:
    for label in file:
        LABELS.append(label.strip())

# CSV functions()
def getTitle(line):
    return line.split(',')[2]

def cleanString(string):
    # 1. Replace /,.-=` etc with space then tokenize
    # 2. Make everything lowercase
    string = string.replace(',',' ').replace('.',' ').replace('=',' ').replace('`',' ').replace('/',' ').replace('-',' ').lower()
    return string


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.word2count = {}
        self.labels = LABELS
        self.label2idx = {}

        for i in range(len(self.labels)):
            self.label2idx[self.labels[i]] = i

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1

            try:
                self.word2count[word] = self.word2count[word] + 1
            except:
                self.word2count[word] = 1

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
                title = cleanString(getTitle(line))
                words = title.split()
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(filepath, 'r') as file:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in file:
                title = cleanString(getTitle(line))
                words = title.split()
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

class TxtDatasetProcessing(Dataset):
    def __init__(self, filename, sentence_length, corpus):
        file_labels = open(filename, 'r')
        labels = []
        for line in file_labels:
            # Just take the first label
            # TODO: Make cleaner
            label = line.split(',')[4].split('|')[0].strip()

            labels.append(label)

        file_labels.close()

        self.label = labels
        self.corpus = corpus
        self.sentence_length = sentence_length


    def __getitem__(self, index):
        # TODO
        return

    def __len__(self):
        # TODO
        return

if __name__=='__main__':

    # python utils/data_process.py in /training
    corpus = Corpus('../data/data.example.csv')

    print corpus.dictionary.word2idx
    print '-'
    print corpus.dictionary.idx2word
    print '-'
    print corpus.dictionary.word2count
    print '-'
    print corpus.dictionary.labels
    print '-'
    print corpus.dictionary.label2idx
    print '-'
    print corpus.data


    textdatasetprocessing = TxtDatasetProcessing('../data/data.example.csv', 32, corpus)

    print textdatasetprocessing.label
    print '-'
    print textdatasetprocessing.sentence_length