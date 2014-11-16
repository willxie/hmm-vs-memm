'''
Created on Nov 11, 2014

@author: oropivan
'''
import numpy

from hmm import HMM, viterbi

stateNum = 2
transition_probabilities = numpy.array( [   [.5,.5],
                                            [.4,.6]  ] )
emission_probabilities = numpy.array(   [   [0.2, 0.3, 0.3, 0.2], \
                                            [0, 0.5, 0.2, 0.3] ] )
#symbols
symbolList = ["A","C", "G", "T"]
Pi = [0.5,0.5]

model = HMM(stateNum, A=transition_probabilities, B=emission_probabilities, V=symbolList, Pi=Pi)
print viterbi(model, [ "G", "G", "C", "A", "C", "T", "G", "A", "A"])