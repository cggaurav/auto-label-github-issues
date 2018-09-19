import os
import torch

from torch.autograd import Variable

import utils.data_process as DataProcess
import utils.lstm_classifier as LSTMC

# TODO: Load this from ENV
CORPUSFILE_NAME = os.getenv('CORPUS_FILE', './models/GITHUB_ISSUE_CLASSIFIER_19_Sep_09.corpus.pkl')
MODELFILE_NAME = os.getenv('MODEL_FILE', './models/GITHUB_ISSUE_CLASSIFIER_19_Sep_09.model.pth')
TESTFILE_NAME = os.getenv('TEST_FILE', '../data/data.test.csv')

CORPUS = DataProcess.Corpus(CORPUSFILE_NAME)

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

def predict(issue):
    issue = str(issue)
    # return "Hello, Auto Label Github Issues (pass ?issue=)| " + issue
    probability = MODEL.forward(Variable(CORPUS.transform(issue)))
    normalized_scores = torch.nn.functional.softmax(probability)
    score, class_index = normalized_scores.max(1)


def test():
	with open(TESTFILE_NAME, 'r') as file:
	    for line in file:
            # TODO
	    	print line


# Load the model and run the server
if __name__ == "__main__":
	test()