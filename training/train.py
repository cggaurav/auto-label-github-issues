import os
import torch
import copy

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import utils.data_process as DataProcess
import utils.lstm_classifier as LSTMC

USE_PLOT = False
SAVE_MODEL = True

if SAVE_MODEL:
    import pickle
    from datetime import datetime

INPUT_FILE = '../data/data.example.csv'
## CONFIGURATIONS
EPOCHS = 50
BATCH_SIZE = 1 # KISS
LEARNING_RATE = 0.01
EMBEDDING_DIM = 100
HIDDEN_DIM = 50
NLABEL = 6
USE_GPU = torch.cuda.is_available()

def adjust_learning_rate(optimizer, epoch):
    lr = LEARNING_RATE * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

if __name__=='__main__':

    corpus = DataProcess.Corpus(INPUT_FILE)

    model = LSTMC.LSTMClassifier(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, 
                           vocab_size=len(corpus.dictionary), label_size=NLABEL, batch_size=BATCH_SIZE, use_gpu=USE_GPU)
    if USE_GPU:
        model = model.cuda()

    dtrain_set = DataProcess.TxtDatasetProcessing(INPUT_FILE, corpus)

    train_loader = DataLoader(dtrain_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    # TODO: Change for more label output
    loss_function = nn.CrossEntropyLoss()
    train_loss = []
    train_accuracy = []

    for epoch in range(EPOCHS):
        optimizer = adjust_learning_rate(optimizer, epoch)

        total_accuracy = 0.0
        total_loss = 0.0
        total = 0.0

        for iter, train_data in enumerate(train_loader):
            train_inputs, train_labels = train_data
            train_labels = torch.squeeze(train_labels)

            if USE_GPU:
                train_inputs, train_labels = Variable(train_inputs.cuda()), train_labels.cuda()
            else: train_inputs = Variable(train_inputs)

            model.zero_grad()
            model.BATCH_SIZE = len(train_labels)
            model.hidden = model.init_hidden()
            output = model(train_inputs.t())

            loss = loss_function(output, Variable(train_labels))
            loss.backward()
            optimizer.step()

            # calc training acc
            _, predicted = torch.max(output.data, 1)
            total_accuracy += (predicted == train_labels).sum()
            total += len(train_labels)
            total_loss += loss.data[0]

        train_loss.append(total_loss / total)
        train_accuracy.append(total_accuracy / total)

        print('[Epoch: %3d/%3d] Training Loss: %.3f, Training Accuracy: %.3f' % (epoch, EPOCHS, train_loss[epoch], train_accuracy[epoch]))

    param = {}
    param['learning_rate'] = LEARNING_RATE
    param['batch_size'] = BATCH_SIZE
    param['embedding_dim'] = EMBEDDING_DIM
    param['hidden_dim'] = HIDDEN_DIM

    result = {}
    result['param'] = param

    if USE_PLOT:
        import PlotFigure as PF
        PF.PlotFigure(result, SAVE_MODEL)

    if SAVE_MODEL:
        # TODO: Save `.pth` file
        modelfile = 'models/LSTM_classifier_' + datetime.now().strftime("%d-%h-%m-%s") + '.pkl'
        result['modelfile'] = modelfile

        fp = open(modelfile, 'wb')

        pickle.dump(result, fp)

        fp.close()
        print('File %s is saved.' % modelfile)