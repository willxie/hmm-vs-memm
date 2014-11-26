import sys
import numpy
from filter import readSentences, createConditionalProbabilitiesTables, getCountsFromSentences

# Count the number of words in total observed
def countWords(sentences):
	count = 0
	for sentence in sentences:
		for word, tag in sentence:
			count += 1
	return count

# Build the last, (o, s) specific feature
# Note tags and words are stored in all uppercase
def buildLastFeature(max_num_features, C, map_index_symbol, map_index_POS):
	last_feature_list = {}

	for i in range(len(map_index_POS)):
		for j in range(len(map_index_symbol)):
			word = map_index_symbol[j]
			tag = map_index_POS[i]
			total = 0
			for l in range(max_num_features):
				total += feature(l, word, tag)
			last_feature_list[(word, tag)]	= C - total

	return last_feature_list

# Indicator feature function that reutrns 1 if feature is matched otherwise 0
# This includes the n + 1 feature, which requires (word, tag) to access
# Total number of unique features = max_num_features + N * M
# WORDS ARE CASE INSENSITIVE
# f_<b, s>(O_t, S_t)
def feature(num_features, word, state, last_feature_list = {}, word_tag_tuple = ()):
	word = word.upper()

	# Done by inspection of the first file in Brown
	if num_features == 0:
		return 1 if state == "NNS" and word.endswith("S") else 0
	elif num_features == 1:
		return 1 if state == "VBD" and word.endswith("ED") else 0
	elif num_features == 2:
		return 1 if state == "VBG" and word.endswith("ING") else 0
	elif num_features == 3:
		return 1 if state == "TO" and word == "TO" else 0
	elif num_features == 4:
		return 1 if state == "DT" and word == "THE" else 0
	else:
		temp_tuple = (word, state)
		if temp_tuple == word_tag_tuple:
			return last_feature_list[(word, state)] # This should error if last_feature_list is not passed in
		else:
			return 0
	'''
	featureList =
	# Nouns
	featureIndicator["NNS"] = {
		0: word.endswith("s")
	}
	# Verbs
	featureIndicator["VBG"] = {
		0: word.endswith("ing"),
		1: word.endswith("ed")
	}
	featureIndicator["VBN"] = {
		0: word.endswith("ing"),
		1: word.endswith("ed")
	}
	featureIndicator["NNP"] = {
		0: word.endswith("ing"),
		1: word.endswith("ed"),
		2: word == "JURY"
	}
	'''

# Calculate the training data average for each feature
# Length = max_num_features + N * M
def buildAverageFeature(sentences, m, max_num_features, last_feature_list):
	F = {}

	# Regular features
	for i in range(max_num_features):
		F[i] = float(0)
		for sentence in sentences:
			for word, tag in sentence:
				F[i] += feature(i, word, tag)
		F[i] = F[i] / m

	# (o, s) dependent features
	for word_tag_tuple in last_feature_list:
		F[word_tag_tuple] = float(0)
		for sentence in sentences:
			for word, tag in sentence:
				F[word_tag_tuple] += feature(max_num_features, word, tag, last_feature_list, word_tag_tuple)
		F[word_tag_tuple] = F[word_tag_tuple] / m

	return F

# Calculate the expectation for each feature
# param m number of observations
def buildExpectation(sentences, m, max_num_features, last_feature_list, TPM, map_POS_index, map_symbol_index, map_index_POS):
	E = {}
	N = len(map_POS_index)
	M = len(map_symbol_index)

	for i in range(max_num_features):
		E[i] = float(0)
		previous_tag = ""
		for sentence in sentences:
			for word, tag in sentence:
				if previous_tag == "":
					# TODO Uniform distribution for first state
					for j in range(N):
						E[i] += 1 * feature(i, word, map_index_POS[j])
				else:
					l = map_POS_index[previous_tag]
					k = map_symbol_index[word.upper()]
					for j in range(N):
						E[i] += TPM[l*M+k][j] * feature(i, word, map_index_POS[j])
				previous_tag = tag
		E[i] = E[i] / m

	for word_tag_tuple in last_feature_list:
		E[word_tag_tuple] = float(0)
		previous_tag = ""
		for sentence in sentences:
			for word, tag in sentence:
				if previous_tag == "":
					for j in range(N):
						# TODO Uniform distribution for first state
						E[word_tag_tuple] +=  feature(max_num_features, word, map_index_POS[j], last_feature_list, word_tag_tuple)
				else:
					l = map_POS_index[previous_tag]
					k = map_symbol_index[word.upper()]
					for j in range(N):
						E[word_tag_tuple] += TPM[l*M+k][j] * feature(max_num_features, word, map_index_POS[j], last_feature_list, word_tag_tuple)
				previous_tag = tag

		E[word_tag_tuple] = E[word_tag_tuple] / m

	return E

# Use Generalized iterative scaling to learn Lambda parameter
def GIS(Lambda, C, F, E):
	for key in Lambda:
#		assert F[key] != 0, "F[%s] == 0" % key
#		assert E[key] != 0, "E[%s] == 0" % key
		assert not(F[key] == 0 and E[key] != F[key]), "F[{0}] == 0 but not E".format(key)
		assert not(E[key] == 0 and E[key] != F[key]), "E[{0}] == 0 but not F".format(key)
		if F[key] == 0 and E[key] == 0:
			# Speical feature-not-found-in-observation case but still F[key] = E[key] satisfying the requirement
			Lambda[key] = Lambda[key]
		else:
			Lambda[key] = Lambda[key] + 1 / C * numpy.log(F[key]/E[key])

# Make each row of TPM add up to 1
def normalizeTPM(TPM):
	TPM_row_sum = numpy.sum(TPM, axis = 1)

	for i in range(TPM.shape[0]):
		for j in range(TPM.shape[1]):
			TPM[i][j] = TPM[i][j] / TPM_row_sum[i]
	return TPM

# Create un-normalized transition probability matrix (TPM) given previous state and current observation
def buildTPM(Lambda, max_num_features, map_index_symbol, map_index_POS, last_feature_list):
	N = len(map_index_POS)
	M = len(map_index_symbol)
	TPM = numpy.zeros(shape = (N * M, N), dtype = float)	# TPM (N * M) x N transitional probability matrix

	# Calculate states
	for i in range(0, N):					# Previous state
		for k in range(0, M):				# Current observation
			for j in range(0, N):			# Current/target state
				word = map_index_symbol[k]
				tag = map_index_POS[j]
				# Sum(Lambda_a * feature_a)
				for l in range(0, max_num_features): # Normal features
					TPM[i*M+k][j] += Lambda[l] * feature(l, word, tag)
				# Special feature
				word_tag_tuple = (word.upper(), tag)
				TPM[i*M+k][j] += Lambda[word_tag_tuple] * feature(max_num_features, word, tag, last_feature_list, word_tag_tuple)
	# Raise to exponential
	return numpy.exp(TPM)

# Use viterbi algorithm to find the most probable sequence of tags
# Based on the hmm.py implementation
# param Pi_state_index index for tag "<S>"
# param word_sequence make sentences into single list with appended <S> and <\S>
def MEMMViterbi(TPM, Pi_state_index, word_sequence, m, map_symbol_index, map_POS_index):
	N = len(map_POS_index)      
	M = len(map_symbol_index)

	# Delta[s, t], Psi[s, t]
	Delta = numpy.zeros([N, m], float)		# Track Max probabilities for each t
	Psi =  numpy.zeros([N, m], int) 		# Track Maximal States for each t

	# Given the starting state (t = -1), calculate max prob. each state to current observation
	# Note that because MEMM takes both state and obs as given, only consider for each state with fix obs
	for j in range(N):
		# Initial last tag is assumed to be Pi_state_index
		word_index = map_symbol_index[word_sequence[0].upper()]
		Delta[j, 0] = TPM[Pi_state_index*M+word_index][j]
        
	# Inductive Step:
	for t in range(1, m):			
		word_index = map_symbol_index[word_sequence[t].upper()]
		for j in range(N):			# For each destination state at t
			temp = numpy.zeros(N, float)
			for i in range(N):		# For each source state at t - 1
				temp[i] = Delta[i, t-1] * TPM[i*M+word_index][j] # 1 x N vector that stores 
			Delta[j, t] = temp.max()
			Psi[j, t] = temp.argmax()

	# Calculate State Sequence, Q*:
	Q_star = [numpy.argmax(Delta[ :,M-1])] 
	for t in reversed(range(M-1)) :
		Q_star.insert(0, Psi[Q_star[0],t+1])

	return (Q_star, Delta, Psi)

# test section
numpy.set_printoptions(threshold=sys.maxint)

sentences = readSentences("/Volumes/Storage/git/graphical_models_memm_vs_hmm/data/pos/brown/", 1)
#sentences2 = readSentences("/Volumes/Storage/git/graphical_models_memm_vs_hmm/data_bak/pos/brown/ca/", 10)

symbolsSeen, POS_tagsSeen, map_wordPOS_count, map_POSPOS_count, map_POS_count, map_word_count = getCountsFromSentences(sentences)

map_symbol_index, map_POS_index, transition_probabilities, emission_probabilities = createConditionalProbabilitiesTables(sentences, False)

# Note that (number of unique words) M <= m (number of words)
N = len(POS_tagsSeen)
M = len(symbolsSeen)
map_index_symbol =  {v: k for k, v in map_symbol_index.items()}
map_index_POS =  {v: k for k, v in map_POS_index.items()}
m = countWords(sentences)
C = 6 # This should be number of features + 1
max_num_features = C - 1
Lambda = {}

last_feature_list =  buildLastFeature(max_num_features, C, map_index_symbol, map_index_POS)
F = buildAverageFeature(sentences, m, max_num_features, last_feature_list) # Consider coverting to numpy representation
print(F)
# Initialize Lambda as 1 then learn from training data
Lambda = F.copy()
for key in Lambda:
	Lambda[key] = 1

TPM = buildTPM(Lambda, max_num_features, map_index_symbol, map_index_POS, last_feature_list)
TPM = normalizeTPM(TPM)

E = buildExpectation(sentences, m, max_num_features, last_feature_list, TPM, map_POS_index, map_symbol_index, map_index_POS)

GIS(Lambda, C, F, E)
