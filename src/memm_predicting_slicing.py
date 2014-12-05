from memm import *

numpy.set_printoptions(threshold=sys.maxint)

if len(sys.argv) == 1:
	epsilon = 0.1 				# This is convergence threshold for Lambda
	num_sentence_to_learn = 200  # Training sentences
else:
	epsilon = float(sys.argv[1])
	num_sentence_to_learn = int(sys.argv[2])

print("epsilon = "),
print(epsilon)
print("num_sentence_to_learn = "),
print(num_sentence_to_learn)

# sentences = readSentences("../data/pos/brown", num_sentence_to_learn)

num_sentence = 0
num_sentence_correct = 0
num_tags_correct = 0
num_tags = 0 

sentences_total = readSentences("../data/pos/wsj", num_sentence_to_learn)

numberCrosses = 20
crossSize = len(sentences_total)/ numberCrosses

for slice_index in range(numberCrosses):
	print("slice_index = {0}".format(slice_index))
	if slice_index != 0:
		train1 = sentences_total[:slice_index*crossSize]
	else:
		train1 = []
	train2 = sentences_total[(slice_index+1)*crossSize:]

	# Training
	# sentences = train1 + train2
	# Testing
	sentences = sentences_total[slice_index*crossSize: (slice_index+1)*crossSize]

	# MEMM model to use
	TPM = numpy.load("TPM_wsj_{0}_{1}_{2}.npy".format(epsilon, num_sentence_to_learn, slice_index))
	Lambda = numpy.load("Lambda_wsj_{0}_{1}_{2}.npy".format(epsilon, num_sentence_to_learn, slice_index))

	symbolsSeen, POS_tagsSeen, map_wordPOS_count, map_POSPOS_count, map_POS_count, map_word_count = getCountsFromSentences(sentences)
	map_symbol_index, map_POS_index, transition_probabilities, emission_probabilities = createConditionalProbabilitiesTables(sentences, False)
	map_index_symbol =  {v: k for k, v in map_symbol_index.items()}
	map_index_POS =  {v: k for k, v in map_POS_index.items()}

	Pi_state_index = map_POS_index["<S>"]
	viterbi_tuple = []
	num_sentence_per_slice = 0
	num_sentence_correct_per_slice = 0
	num_tags_correct_per_slice = 0
	num_tags_per_slice = 0 

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

		# print("sentence {0}".format(num_sentence))
		
		# print(word_sequence)
		# print("*"*80)
		# print(viterbi_sequence)
		# print("*"*80)
		# print(pos_sequence)
		# print("*"*80)

		if viterbi_sequence == pos_sequence:
			num_sentence_correct += 1
			num_sentence_correct_per_slice += 1
		num_sentence += 1
		num_sentence_per_slice += 1

		for x, y in zip(pos_sequence, viterbi_sequence):
			if x == y:
				num_tags_correct += 1
				num_tags_correct_per_slice += 1
			num_tags += 1
			num_tags_per_slice += 1

	# Output per slice correctness
	print("Per slice sentence correct = {0}".format(num_sentence_correct_per_slice/float(num_sentence_per_slice)))
	print("Per slice tags correct = {0}".format(num_tags_correct_per_slice/float(num_tags_per_slice)))

print("Total sentence correct = {0}".format(num_sentence_correct/float(num_sentence)))
print("Total tags correct = {0}".format(num_tags_correct/float(num_tags)))
