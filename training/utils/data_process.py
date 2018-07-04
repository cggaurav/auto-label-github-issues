import os
import torch
import sys
import copy
from torch.utils.data.dataset import Dataset
import numpy as np

LABELS = []
with open('../data/data.labels.csv', 'r') as file:
    for label in file:
        LABELS.append(label.strip())

## CSV functions()
def CSVLineGetTitle(line):
    return line.split(',')[2]

def CSVLineGetLabel(line):
    # Make this cleaner
    # TODO: Get all the other labels
    try:
        return line.strip().split(',')[4].strip().split('|')[0].strip()
    except:
        # Default to help wanted becuase incomplete data
        return "help wanted"

def normalizeText(string):
    # 1. Replace /,.-=` etc with space then tokenize
    # 2. Make everything lowercase
    string = string.strip().replace(',',' ').replace('.',' ').replace('=',' ').replace('`',' ').replace('/',' ')
    string = string.replace('-',' ').replace('"', ' ').replace('#',' ').replace(':',' ').replace(';',' ')
    string = string.replace('*', ' ').strip().lower()
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
                title = normalizeText(CSVLineGetTitle(line))
                words = title.split()
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(filepath, 'r') as file:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in file:
                title = normalizeText(CSVLineGetTitle(line))
                words = title.split()
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

    def transform(self, title):
        title = normalizeText(title)

        text = torch.LongTensor(np.zeros(len(title.split()), dtype=np.int64))
        text_count = 0

        for word in title.split():
            if word.strip() in self.dictionary.word2idx:
                text[text_count] = self.dictionary.word2idx[word.strip()]
                text_count = text_count + 1

        return text

# DOCS: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
# https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
class TxtDatasetProcessing(Dataset):
    def __init__(self, filename, corpus):
        self.corpus = corpus
        self.file = []

        with open(filename, 'r') as file:
            for line in file:
                self.file.append(line)

        file.close()

    def __getitem__(self, index):
        count = 0

        text = None
        label = None
        text_count = 0

        for line in self.file:
            # Lets process the right index
            if count == index:

                # print line
                title = normalizeText(CSVLineGetTitle(line))
                labelled = normalizeText(CSVLineGetLabel(line))

                text = torch.LongTensor(np.zeros(len(title.split()), dtype=np.int64))

                for word in title.split():
                    if word.strip() in self.corpus.dictionary.word2idx:
                        text[text_count] = self.corpus.dictionary.word2idx[word.strip()]
                        text_count = text_count + 1

                # TODO: How does this look?
                # If only one label, then [1, 0, 0, 0, 0 ]
                # If multiple labels, then [1, 0, 1, 0, 0 ]
                label = torch.LongTensor([self.corpus.dictionary.label2idx[labelled]])

            count = count + 1

        return text, label


    def __len__(self):
        # TODO, length of training data, # of lines in CSV
        count = 0
        for line in self.file:
            count += 1
        return count

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

    textdatasetprocessing = TxtDatasetProcessing('../data/data.example.csv', corpus)

    print textdatasetprocessing