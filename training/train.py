import os
import torch

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import utils.data_process as DataProcess
import utils.lstm_classifier as LSTMC

from datetime import datetime


USE_PLOT = False
SAVE_MODEL = True

INPUT_FILE = '../data/data.example.csv'

## CONFIGURATIONS
EPOCHS = 50
BATCH_SIZE = 1 # KISS
LEARNING_RATE = 0.01
EMBEDDING_DIM = 100
HIDDEN_DIM = 50
NLABEL = 6
USE_GPU = torch.cuda.is_available()

# ISSUE: https://github.com/pytorch/pytorch/issues/6932#issuecomment-384509898
# NOTE: USE TORCH 0.3.1
print "Torch version : %s" % torch.__version__
print "Using GPU : %s" % USE_GPU

def adjust_learning_rate(optimizer, epoch):
    lr = LEARNING_RATE * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['learning_rate'] = lr
    return optimizer

if __name__=='__main__':

    CORPUS = DataProcess.Corpus(INPUT_FILE)

    model = LSTMC.LSTMClassifier(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, 
                vocab_size=len(CORPUS.dictionary), label_size=NLABEL, batch_size=BATCH_SIZE, use_gpu=USE_GPU)
    if USE_GPU:
        model = model.cuda()

    dtrain_set = DataProcess.TxtDatasetProcessing(INPUT_FILE, CORPUS)

    train_loader = DataLoader(dtrain_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    # TODO: Change for more label output
    # DOCS: https://pytorch.org/docs/stable/nn.html#crossentropyloss

    # TODO: Understand loss functions
    # https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss
    loss_function = nn.BCELoss()
    train_loss = []
    train_accuracy = []

    for epoch in range(EPOCHS):
        optimizer = adjust_learning_rate(optimizer, epoch)

        total_accuracy = 0.0
        total_loss = 0.0
        total = 0.0

        for iter, train_data in enumerate(train_loader):

            print "Epoch: %s | Training : %s" % (epoch, iter)

            train_inputs, train_labels = train_data

            train_labels = torch.squeeze(train_labels)

            if USE_GPU:
                train_inputs, train_labels = Variable(train_inputs.cuda()), train_labels.cuda()
            else: 
                train_inputs = Variable(train_inputs)

            model.zero_grad()

            # model.BATCH_SIZE = len(train_labels)

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
        modelfilename = 'models/GITHUB_ISSUE_CLASSIFIER_' + datetime.now().strftime("%d_%h_%m") + '.pth'

        result['modelfilename'] = modelfilename

        state = {
            'epoch': EPOCHS,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'embedding_dim': EMBEDDING_DIM,
            'hidden_dim': HIDDEN_DIM,
            'batch_size': BATCH_SIZE,
            'label_size': NLABEL,
            'learning_rate': LEARNING_RATE,
            'vocab_size': len(CORPUS.dictionary)
        }

        torch.save(state, modelfilename)

        print('File %s is saved.' % modelfilename)