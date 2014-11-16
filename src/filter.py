'''
Created on Nov 11, 2014

@author: czar
'''

from os import listdir
from os.path import isfile, join
import sys
import numpy as np

# POS_TAGS = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB"]

def readFile(systemRun, numberOfSentencesToTrain, currentCount):
    sentences = []
    previousTag = None
    sentence = []
    with open(systemRun, 'rb') as datafile:
        for line in datafile:
            if not line.strip(): 
                continue
            if line.startswith("==="):
                if previousTag is None:
                    sentence = []
                    continue
                if currentCount == numberOfSentencesToTrain:
                    return numberOfSentencesToTrain
                else: 
                    sentences.append(sentence)
                    currentCount += 1
                    sentence = []
                    continue
            if line.startswith("["):
                line = line[1:-2].strip()
            line = line.split()
            for observation in line:
                wordTagTuple = observation.split('/')
                if len(wordTagTuple) == 2:
                    word, tag = wordTagTuple
                else: # for words that have /
                    tag = wordTagTuple[-1]
                    word = '/'.join(wordTagTuple[:-1])
                word = word.upper()
                sentence.append((word, tag))
    sentences.append(sentence)
    currentCount += 1
    return sentences

def getCountsFromSentences(sentences):
    SYMBOLS_SEEN = set()
    POS_TAGS_SEEN = set()
    
    map_wordPOS_count = {}
    map_POSPOS_count = {}
    
    map_POS_count = {}
    map_word_count = {}
    for sentence in sentences:
        previousTag = '<S>'
        POS_TAGS_SEEN.add(previousTag)
        
        word = '<S>'
        SYMBOLS_SEEN.add(word)
        
        map_word_count[word] = map_word_count.get(word, 0) + 1
        map_POS_count[previousTag] = map_POS_count.get(previousTag, 0) + 1
        
        key = tuple([word,previousTag])
        map_wordPOS_count[key] = map_wordPOS_count.get(key, 0) + 1
        for word, tag in sentence:
            # add to vocabulary
            POS_TAGS_SEEN.add(tag)
            SYMBOLS_SEEN.add(word)
            
            # count tag and word instances
            map_word_count[word] = map_word_count.get(word, 0) + 1
            map_POS_count[tag] = map_POS_count.get(tag, 0) + 1
            
            # count word,TAG instances
            key = tuple([word,tag])
            map_wordPOS_count[key] = map_wordPOS_count.get(key, 0) + 1
             
            #count tag tag instances
            key = tuple([tag,previousTag])
            map_POSPOS_count[key] = map_POSPOS_count.get(key, 0) + 1
             
            previousTag = tag
        tag = "<\S>"
        POS_TAGS_SEEN.add(tag)
        word = '<\S>'
        SYMBOLS_SEEN.add(word)
        
        map_POS_count[tag] = map_POS_count.get(tag, 0) + 1
        map_word_count[word] = map_word_count.get(word, 0) + 1
        
        key = tuple([tag,previousTag])
        map_POSPOS_count[key] = map_POSPOS_count.get(key, 0) + 1
        key = tuple([word,tag])
        map_wordPOS_count[key] = map_wordPOS_count.get(key, 0) + 1
        
    return (SYMBOLS_SEEN, POS_TAGS_SEEN, map_wordPOS_count, map_POSPOS_count, map_POS_count, map_word_count)
        
def dirTraverse(path, numberOfSentencesToTrain, currentCount, sentences):
    files = [ f for f in listdir(path)]
    for fileName in files:
        if currentCount == numberOfSentencesToTrain:
            break
        fileName = "/".join([path,fileName])
        if isfile(fileName):
            sentencesRead = readFile(fileName, numberOfSentencesToTrain, currentCount)
            currentCount += len(sentencesRead)
            sentences = sentences + sentencesRead
        else:
            currentCount, sentences = dirTraverse(fileName, numberOfSentencesToTrain, currentCount, sentences)
    return (currentCount, sentences)

def createEmissionProbabilities(symbolsSeen, POS_tagsSeen, map_symbol_index, map_POS_index, map_wordPOS_count, map_POS_count):
    emissionProbs = np.zeros((len(POS_tagsSeen), len(symbolsSeen)))
    for POS in POS_tagsSeen:
        for word in symbolsSeen:
            key = tuple([word, POS])
            numerator = float(map_wordPOS_count.get(key,0))
            denominator = float(map_POS_count[POS])
            emissionProbs[map_POS_index[POS]][map_symbol_index[word]] = numerator/denominator
    return emissionProbs

def createTransitionProbabilities(POS_tagsSeen, map_POS_index, map_POSPOS_count, map_POS_count):
    transitionProb = np.zeros((len(POS_tagsSeen), len(POS_tagsSeen)))
    for pPOS in POS_tagsSeen:
        for cPOS in POS_tagsSeen:
            key = tuple([cPOS,pPOS])
            numerator = float(map_POSPOS_count.get(key,0))
            denominator = float(map_POS_count[pPOS])
            transitionProb[map_POS_index[cPOS]][map_POS_index[pPOS]] = numerator/denominator
    return transitionProb            
def readSentences(rootPath, numberOfSentencesToTrain):
    sentenceCount, sentences = dirTraverse(rootPath, numberOfSentencesToTrain, 0, [])
    return sentences
def createConditionalProbabilitiesTables(sentences):
    symbolsSeen, POS_tagsSeen, map_wordPOS_count, map_POSPOS_count, map_POS_count, map_word_count = getCountsFromSentences(sentences)
    map_symbol_index = {v: k for k, v in dict(enumerate(symbolsSeen)).items()}
    map_POS_index = {v: k for k, v in dict(enumerate(POS_tagsSeen)).items()}
    
    emission_probabilities = createEmissionProbabilities(symbolsSeen, POS_tagsSeen, map_symbol_index, map_POS_index, map_wordPOS_count, map_POS_count)
    transition_probabilities = createTransitionProbabilities(POS_tagsSeen, map_POS_index, map_POSPOS_count, map_POS_count)
#     print map_POS_index
#     print transition_probabilities
#     print [sum(transition_probabilities[i]) for i in range(transition_probabilities.shape[0])]
#      
#     print map_symbol_index
#     print [sum(emission_probabilities[i]) for i in range(emission_probabilities.shape[0])]
    return (map_symbol_index, map_POS_index, transition_probabilities, emission_probabilities)
def main():
    path = "/home/czar/dev/GraphModels/finalProject/data/pos/wsj"
    number = 1
    createConditionalProbabilitiesTables(path, number)
if __name__ == "__main__":
    main()