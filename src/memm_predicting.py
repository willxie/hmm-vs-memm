from memm import *

numpy.set_printoptions(threshold=sys.maxint)

if len(sys.argv) == 1:
	num_sentence_to_learn = 100  # Training sentences
else:
	num_sentence_to_learn = int(sys.argv[1])

print("num_sentence_to_learn = "),
print(num_sentence_to_learn)

sentences = readSentences("../data/pos/wsj", num_sentence_to_learn)

# MEMM model to use
TPM = numpy.load("TPM_wsj_0.1_100.npy")
Lambda = numpy.load("Lambda_wsj_0.1_100.npy")

symbolsSeen, POS_tagsSeen, map_wordPOS_count, map_POSPOS_count, map_POS_count, map_word_count = getCountsFromSentences(sentences)
map_symbol_index, map_POS_index, transition_probabilities, emission_probabilities = createConditionalProbabilitiesTables(sentences, False)
map_index_symbol =  {v: k for k, v in map_symbol_index.items()}
map_index_POS =  {v: k for k, v in map_POS_index.items()}

Pi_state_index = map_POS_index["<S>"]
viterbi_tuple = []
num_sentence = 0
num_sentence_correct = 0
num_tags_correct = 0
num_tags = 0 

for sentence in sentences:

	word_sequence = []
	pos_sequence = []

	# word_sequence.append('<S>')
	# pos_sequence.append('<S>')
	for word, tag in sentence:
		word_sequence.append(word)
		pos_sequence.append(tag)
	# word_sequence.append('<\S>')
	# pos_sequence.append('<\S>')

	m = len(word_sequence)
	if m == 0:
		continue

	viterbi_tuple = MEMMViterbi(TPM, Pi_state_index, word_sequence, map_symbol_index, map_POS_index)
	viterbi_sequence = [map_index_POS[x] for x in viterbi_tuple[0]]

	print("sentence {0}".format(num_sentence))
	# print(word_sequence)
	# print("*"*80)
	# print(viterbi_sequence)
	# print("*"*80)
	# print(pos_sequence)
	# print("*"*80)

	if viterbi_sequence == pos_sequence:
		num_sentence_correct += 1
	num_sentence += 1

	for x, y in zip(pos_sequence, viterbi_sequence):
		if x == y:
			num_tags_correct += 1
		num_tags += 1

print num_sentence_correct/float(num_sentence)
print num_tags_correct/float(num_tags)
