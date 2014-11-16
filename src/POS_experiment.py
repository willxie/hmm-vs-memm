'''
Created on Nov 13, 2014

@author: czar
'''
from filter import readSentences, createConditionalProbabilitiesTables
from hmm import HMM, viterbi
import numpy as np


numberOfSamples = 10000
numberOfTestingSamples = 10
rootExp =  '/home/czar/TO_DELETE'#'/home/czar/dev/GraphModels/finalProject/data/pos/wsj' # PATH TO DATA #'/home/czar/TO_DELETE'#

sentences = readSentences(rootExp, numberOfSamples)
trainSet = sentences[:-numberOfTestingSamples]
testSet = sentences[-numberOfTestingSamples:]
trainSet = sentences
map_symbol_index, map_POS_index, transition_probabilities, emission_probabilities = createConditionalProbabilitiesTables(sentences, True)
map_index_symbol =  {v: k for k, v in map_symbol_index.items()}
map_index_POS =  {v: k for k, v in map_POS_index.items()}
symbolList = []
for key in sorted(map_index_symbol.keys()):
    symbolList.append(map_index_symbol[key])
initialProbabilities = [0 for i in range(len(map_POS_index.keys()))]
initialProbabilities[map_POS_index["<S>"]] = 1
initialProbabilities = np.asarray(initialProbabilities)

print [sum(transition_probabilities[:,i]) for i in range(transition_probabilities.shape[1])]
print [sum(emission_probabilities[i]) for i in range(emission_probabilities.shape[0])]
model = HMM(len(map_POS_index.keys()), A=transition_probabilities, B=emission_probabilities, V=symbolList, Pi=initialProbabilities)

numberCorrect = 0
for sentence in trainSet:
    wordSeq = ['<S>']
    POSSeq = ['<S>']
    for word, POS in sentence:
        wordSeq.append(word)
        POSSeq.append(POS)
    wordSeq.append('<\S>')
    POSSeq.append('<\S>')
    wordSeq[2] = 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
    resultPOS = viterbi(model, wordSeq, scaling=False)
    returnedSeq = [map_index_POS[x] for x in resultPOS[0]]
    if returnedSeq == POSSeq:
        numberCorrect += 1
print numberCorrect/float(len(trainSet))