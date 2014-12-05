import sys
from memm import *

if len(sys.argv) == 1:
	epsilon = 0.1 				# This is convergence threshold for Lambda
	num_sentence_to_train = 100  # Training sentences
else:
	epsilon = float(sys.argv[1])
	num_sentence_to_train = int(sys.argv[2])
print("epsilon = "),
print(epsilon)
print("num_sentence_to_train = "),
print(num_sentence_to_train)

# Begin training
numpy.set_printoptions(threshold=sys.maxint)

sentences = readSentences("../data/pos/wsj", num_sentence_to_train)

symbolsSeen, POS_tagsSeen, map_wordPOS_count, map_POSPOS_count, map_POS_count, map_word_count = getCountsFromSentences(sentences)
map_symbol_index, map_POS_index, transition_probabilities, emission_probabilities = createConditionalProbabilitiesTables(sentences, False)

# Note that (number of unique words) M <= m (number of words)
N = len(POS_tagsSeen)
M = len(symbolsSeen)
map_index_symbol =  {v: k for k, v in map_symbol_index.items()}
map_index_POS =  {v: k for k, v in map_POS_index.items()}

iter_count = 0
C = 6 						# This should be number of features + 1
max_num_features = C - 1
Lambda = {}
TPM = initTPM(map_index_symbol, map_index_POS)

# Divide (o,s) into |S| buckets
buckets = divideBuckets(sentences, map_POS_index)

last_feature_list =  buildLastFeature(max_num_features, C, map_index_symbol, map_index_POS)

# Initialize Lambda as 1 then learn from training data
# Lambda is different per s' (previous state)
F = buildAverageFeature(buckets, map_POS_index, max_num_features, last_feature_list)

Lambda = initLambda(F)
E = initExpectation(F)

# GIS, run until convergence
while True:
	print("iteratoin = {0}".format(iter_count))
	Lambda0 = copy.deepcopy(Lambda)
	for tag in map_POS_index:
		buildTPM(TPM, Lambda, max_num_features, map_index_symbol, map_index_POS, map_POS_index, last_feature_list, tag)
	for tag in map_POS_index:
		buildExpectation(E, buckets[tag], max_num_features, last_feature_list, TPM, map_POS_index, map_symbol_index, map_index_POS, tag)
	for tag in map_POS_index:
		buildNextLambda(Lambda, C, F, E, tag)
	iter_count += 1

	if checkLambdaConvergence(Lambda0, Lambda, epsilon):
		print " ".join(["iter_count:", str(iter_count)])
		break;

numpy.save("TPM_wsj_{0}_{1}".format(epsilon, num_sentence_to_train), TPM)
numpy.save("Lambda_wsj_{0}_{1}".format(epsilon, num_sentence_to_train), Lambda)

print("training done")
# End training