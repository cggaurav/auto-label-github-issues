import os
import torch
from pprint import pprint

import utils.data_process as DataProcess
import utils.lstm_classifier as LSTMC

MODELFILE_NAME = './models/GITHUB_ISSUE_CLASSIFIER_04_Jul_07.pth'
USE_GPU = torch.cuda.is_available()

# ISSUE: https://github.com/pytorch/pytorch/issues/6932#issuecomment-384509898
# NOTE: USE TORCH 0.3.1
print "Torch version : %s" % torch.__version__
print "Using GPU : %s" % USE_GPU

# QUESTION: Why load the corpus again?
INPUT_FILE = '../data/data.example.csv'
CORPUS = DataProcess.Corpus(INPUT_FILE)

# CONFIGURATIONS
USE_GPU = torch.cuda.is_available()

if __name__=='__main__':
    checkpoint = torch.load(MODELFILE_NAME)

    model = LSTMC.LSTMClassifier(embedding_dim=checkpoint['embedding_dim'], hidden_dim=checkpoint['hidden_dim'], 
			vocab_size=checkpoint['vocab_size'], label_size=checkpoint['label_size'], batch_size=checkpoint['batch_size'], use_gpu=USE_GPU)

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    if USE_GPU:
    	model = model.cuda()

    # TEST
    probability = model.forward(CORPUS.transform("proxier issue"))
    score, class_index = probability.max(1) 
    print probability, score, class_index