'''
Created on Nov 11, 2014

@author: czar
'''

from os import listdir
from os.path import isfile, join
import sys
import numpy as np

# POS_TAGS = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB"]
SYMBOLS_SEEN = set()
POS_TAGS_SEEN = set()
POS_TAGS_SEEN.add("<S>")
POS_TAGS_SEEN.add("<\S>")

map_wordPOS_count = {}
map_POSPOS_count = {}

map_POS_count = {}
map_word_count = {}

map_symbol_index = {}
map_POS_index = {}

def readFile(systemRun, numberOfSentencesToTrain, currentCount):
    previousTag = None
    with open(systemRun, 'rb') as file:
        for line in file:
            if not line.strip(): 
                continue
            if line.startswith("==="):
                if previousTag is None:
                    previousTag = "<S>"
                    map_POS_count[previousTag] = map_POS_count.get(previousTag, 0) + 1
                    continue
                tag = "<\S>"
                map_POS_count[tag] = map_POS_count.get(tag, 0) + 1
                key = tuple([previousTag,tag])
                map_POSPOS_count[key] = map_POSPOS_count.get(key, 0) + 1
                if currentCount == numberOfSentencesToTrain:
                    return numberOfSentencesToTrain
                else: 
                    previousTag = "<S>"
                    map_POS_count[previousTag] = map_POS_count.get(previousTag, 0) + 1
                    currentCount += 1
                    continue
            if line.startswith("["):
                line = line[1:-2].strip()
            line = line.split()
            for observation in line:
                word, tag = observation.split('/')
                word = word.upper()
                POS_TAGS_SEEN.add(tag)
                SYMBOLS_SEEN.add(word)
                
                # count word,TAG instances
                key = tuple([word,tag])
                map_wordPOS_count[key] = map_wordPOS_count.get(key, 0) + 1
                
                #count tag tag instances
                key = tuple([previousTag,tag])
                map_POSPOS_count[key] = map_POSPOS_count.get(key, 0) + 1
                
                # count tag and word instances
                map_word_count[word] = map_word_count.get(word, 0) + 1
                map_POS_count[tag] = map_POS_count.get(tag, 0) + 1
                
                previousTag = tag
    tag = "<\S>"
    map_POS_count[tag] = map_POS_count.get(tag, 0) + 1
    key = tuple([previousTag,tag])
    map_POSPOS_count[key] = map_POSPOS_count.get(key, 0) + 1
    return currentCount

def dirTraverse(path, numberOfSentencesToTrain, currentCount):
    files = [ f for f in listdir(path)]
    for fileName in files:
        if currentCount == numberOfSentencesToTrain:
            break
        fileName = "/".join([path,fileName])
        if isfile(fileName):
            currentCount = readFile(fileName, numberOfSentencesToTrain, currentCount)
        else:
            currentCount = dirTraverse(fileName, numberOfSentencesToTrain, currentCount)
    return currentCount
def createEmissionProbabilities(map_symbol_index, map_POS_index):
    emissionProbs = np.zeros((len(SYMBOLS_SEEN), len(POS_TAGS_SEEN)))
    for POS in POS_TAGS_SEEN:
        for word in SYMBOLS_SEEN:
            key = tuple([word, POS])
            numerator = float(map_wordPOS_count.get(key,0))
            denominator = float(map_word_count[word])
            emissionProbs[map_symbol_index[word]][map_POS_index[POS]] = numerator/denominator
    return emissionProbs

def createTransitionProbabilities(map_POS_index):
    transitionProb = np.zeros((len(POS_TAGS_SEEN), len(POS_TAGS_SEEN)))
    for pPOS in POS_TAGS_SEEN:
        for cPOS in POS_TAGS_SEEN:
            key = tuple([pPOS,cPOS])
            numerator = float(map_POSPOS_count.get(key,0))
            denominator = float(map_POS_count[pPOS])
            transitionProb[map_POS_index[pPOS]][map_POS_index[cPOS]] = numerator/denominator
    return transitionProb            
                     
def main():
    numberOfSentencesToTrain = int(sys.argv[2])
    rootPath = sys.argv[1]
    dirTraverse(rootPath, numberOfSentencesToTrain, 0)
    map_symbol_index = {v: k for k, v in dict(enumerate(SYMBOLS_SEEN)).items()}
    map_POS_index = {v: k for k, v in dict(enumerate(POS_TAGS_SEEN)).items()}
    
    emission_probabilities = createEmissionProbabilities(map_symbol_index, map_POS_index)
    transition_probabilities = createTransitionProbabilities(map_POS_index)
    print map_POS_index
    print [sum(transition_probabilities[i]) for i in range(transition_probabilities.shape[0])]
    
    print map_symbol_index
    print [sum(emission_probabilities[i]) for i in range(emission_probabilities.shape[0])]
if __name__ == "__main__":
    main()