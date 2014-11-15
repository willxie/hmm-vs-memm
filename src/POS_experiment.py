'''
Created on Nov 13, 2014

@author: czar
'''
from filter import readSentences, createConditionalProbabilitiesTables
from hmm import HMM
import numpy as np


numberOfSamples = 100
numberOfTestingSamples = 10
rootExp = '/home/czar/dev/GraphModels/finalProject/data/pos/wsj' # PATH TO DATA

sentences = readSentences(rootExp, numberOfSamples)
trainSet = sentences[:-numberOfTestingSamples]
testSet = sentences[-numberOfTestingSamples:]
map_symbol_index, map_POS_index, transition_probabilities, emission_probabilities = createConditionalProbabilitiesTables(sentences)
map_index_symbol =  {v: k for k, v in map_symbol_index.items()}
symbolList = []
for key in sorted(map_index_symbol.keys()):
    symbolList.append(map_index_symbol[key])
initialProbabilities = [0 for i in range(len(map_POS_index.keys()))]
initialProbabilities[map_POS_index["<S>"]] = 1
initialProbabilities = np.asarray(initialProbabilities)

print emission_probabilities.shape 
# print map_POS_index
# print [sum(transition_probabilities[i]) for i in range(transition_probabilities.shape[0])]
#   
# print map_symbol_index
# print [sum(emission_probabilities[i]) for i in range(emission_probabilities.shape[0])]
model = HMM(len(map_POS_index.keys()), A=transition_probabilities, B=emission_probabilities, V=symbolList, Pi=initialProbabilities)