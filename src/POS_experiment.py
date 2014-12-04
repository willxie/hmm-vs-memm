'''
Created on Nov 13, 2014

@author: czar
'''
from filter import readSentences, createConditionalProbabilitiesTables
from hmm import HMM, viterbi
import numpy as np


numberOfSamples = 10000
numberCrosses = 20

numberSentences = 0
numberTags = 0
numberSentencesCorrect = 0
numberTagsCorrect = 0

rootExp =  '/home/czar/dev/GraphModels/finalProject/data/pos/brown' # PATH TO DATA #'/home/czar/TO_DELETE'#

sentences = readSentences(rootExp, numberOfSamples)
crossSize = len(sentences)/ numberCrosses
for i in range(numberCrosses):
    if i != 0:
        train1 = sentences[:i*crossSize: (i)*crossSize]
    else:
        train1 = []
    train2 = sentences[(i+1)*crossSize:]
    trainSet = train1 + train2
    
    testSet = sentences[i*crossSize: (i+1)*crossSize]
    
    # trainSet = sentences
    map_symbol_index, map_POS_index, transition_probabilities, emission_probabilities = createConditionalProbabilitiesTables(trainSet, True)
    map_index_symbol =  {v: k for k, v in map_symbol_index.items()}
    map_index_POS =  {v: k for k, v in map_POS_index.items()}
    symbolList = []
    for key in sorted(map_index_symbol.keys()):
        symbolList.append(map_index_symbol[key])
    initialProbabilities = [0 for x in range(len(map_POS_index.keys()))]
    initialProbabilities[map_POS_index["<S>"]] = 1
    initialProbabilities = np.asarray(initialProbabilities)
    
    # print [sum(transition_probabilities[:,i]) for i in range(transition_probabilities.shape[1])]
    # print [sum(emission_probabilities[i]) for i in range(emission_probabilities.shape[0])]
    model = HMM(len(map_POS_index.keys()), A=transition_probabilities, B=emission_probabilities, V=symbolList, Pi=initialProbabilities)
    
    # testing data
    iterSentencesCorrect = 0
    iterSentences = 0
    iterTagsCorrect = 0
    iterTags = 0
    for sentence in testSet:
        wordSeq = ['<S>']
        POSSeq = ['<S>']
        for word, POS in sentence:
            wordSeq.append(word)
            POSSeq.append(POS)
        wordSeq.append('<\S>')
        POSSeq.append('<\S>')
        resultPOS = viterbi(model, wordSeq, scaling=False)
        returnedSeq = [map_index_POS[x] for x in resultPOS[0]]
        
        if returnedSeq == POSSeq:
            numberSentencesCorrect += 1
            iterSentencesCorrect += 1
        numberSentences +=1
        iterSentences += 1
        
        for x, y in zip(POSSeq, returnedSeq):
            if x==y:
                numberTagsCorrect +=1
                iterTagsCorrect += 1
            numberTags += 1
            iterTags += 1
    print ','.join(["test", str(i), str(iterSentencesCorrect/float(iterSentences)), str(iterTagsCorrect/float(iterTags))])
    # training data
    iterSentencesCorrect = 0
    iterSentences = 0
    iterTagsCorrect = 0
    iterTags = 0
    for sentence in trainSet:
        wordSeq = ['<S>']
        POSSeq = ['<S>']
        for word, POS in sentence:
            wordSeq.append(word)
            POSSeq.append(POS)
        wordSeq.append('<\S>')
        POSSeq.append('<\S>')
        resultPOS = viterbi(model, wordSeq, scaling=False)
        returnedSeq = [map_index_POS[x] for x in resultPOS[0]]
        
        if returnedSeq == POSSeq:
            iterSentencesCorrect += 1
        iterSentences += 1
        
        for x, y in zip(POSSeq, returnedSeq):
            if x==y:
                iterTagsCorrect += 1
            iterTags += 1
    print ','.join(["train", str(i), str(iterSentencesCorrect/float(iterSentences)), str(iterTagsCorrect/float(iterTags))])

print numberSentencesCorrect/float(numberSentences)
print numberTagsCorrect/float(numberTags)