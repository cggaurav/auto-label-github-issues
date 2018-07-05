import os
import torch
from torch.autograd import Variable

import utils.data_process as DataProcess
import utils.lstm_classifier as LSTMC

MODELFILE_NAME = './models/GITHUB_ISSUE_CLASSIFIER_05_Jul_07.pth'
INPUT_FILE = '../data/data.example.csv' # QUESTION: Why load the corpus again?

CORPUS = DataProcess.Corpus(INPUT_FILE)

LABELS = DataProcess.getLabels()
LABELS2IDX = DataProcess.getLabel2Idx()

# CONFIGURATIONS
USE_GPU = torch.cuda.is_available()


# ISSUE: https://github.com/pytorch/pytorch/issues/6932#issuecomment-384509898
# NOTE: USE TORCH 0.3.1
print "Torch version : %s" % torch.__version__
print "Using GPU : %s" % USE_GPU

checkpoint = torch.load(MODELFILE_NAME)
MODEL = LSTMC.LSTMClassifier(embedding_dim=checkpoint['embedding_dim'], hidden_dim=checkpoint['hidden_dim'], 
        vocab_size=checkpoint['vocab_size'], label_size=checkpoint['label_size'], batch_size=checkpoint['batch_size'], use_gpu=USE_GPU)

MODEL.load_state_dict(checkpoint['state_dict'])
MODEL.eval()

if USE_GPU:
    MODEL = MODEL.cuda()

def predict(text):
    print "Predicting for : %s" % text
    probability = MODEL.forward(Variable(CORPUS.transform(text)))
    score, class_index = probability.max(1) 
    return LABELS[class_index.data[0]], score.data[0]

if __name__=='__main__':
    # TESTS
    from pprint import pprint
    pprint(predict("This issue is a bug"))
    pprint(predict("This issue is a documentation issue"))
