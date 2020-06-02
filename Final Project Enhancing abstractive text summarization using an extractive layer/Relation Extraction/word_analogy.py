import os
import pickle
import numpy as np
import time

from scipy.spatial.distance import cosine


model_path = './models/'
loss_model = 'cross_entropy'
# loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))
# model_filepath = os.path.join(model_path, 'word2vec_nce_2k_96_128_3_6_16_256_0.06.model')

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))
inv_dict = {v: k for k, v in dictionary.items()}


"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""

questions = []
choices = []

output_file = open("word_analogy_dev_predictions_%s"%(loss_model), "w")
# output_file = open("word_analogy_test_predictions_%s"%(loss_model), "w")


# with open('./word_analogy_dev.txt') as f:
with open('./word_analogy_test.txt') as f:

    for line in f:
    	line = line.strip()
    	splits = line.split("||")
    	questions.append(splits[0])
    	choices.append(splits[1])

for q,c in zip(questions, choices):

	opline = ""

	print("----")
	q_vec_diffs = np.zeros((128, ))

	qs = q.replace("\"", "").split(",")
	i = 0
	for q_p in qs:
		opline = opline + q_p
		parts = q_p.split(":")
		# print(parts[0])
		# print(parts[1])

		q_vec_diffs = np.add(q_vec_diffs, (embeddings[dictionary[parts[0]]] - embeddings[dictionary[parts[1]]]))
		i += 1
		# print(embeddings[dictionary[parts[0]]].shape)

	q_vec_diffs = q_vec_diffs / i

	# print(q_vec_diffs.shape)
	# avg_distance = np.mean(q_cosine_distance)
	# print(avg_distance)

	# c_cosine_distance = []

	max_similarity = np.NINF
	min_similarity = np.Infinity

	options = ""
	# ans_1 = ""
	# ans_2 = ""

	cs = c.replace("\"", "").split(",")
	for c_p in cs:
		options = options + " " + "\"" + c_p + "\""
		parts = c_p.split(":")
		diff = (embeddings[dictionary[parts[0]]] - embeddings[dictionary[parts[1]]])
		sim = cosine(diff, q_vec_diffs)

		if (sim < min_similarity) :
			ans_1 = "\"" + c_p + "\""
			min_similarity = sim

		if (sim > max_similarity) :
			ans_2 = "\"" + c_p + "\""
			max_similarity = sim

		print("sim for " + c_p + " = " + str(sim))

	opline = options[1:] + " " + ans_1 + " " + ans_2
	# print(opline)
	output_file.write(opline+"\n")

output_file.close()



############################## 20 similar words ################################

#top 20 words
# words = ('first', 'american', 'would')

# # print(embeddings.shape)

# for word in words:
#  	embed = embeddings[dictionary[word]]
#  	e = np.expand_dims(embed, axis=1)
#  	# print(e.shape)

#  	sims = np.squeeze(np.matmul(embeddings, e))
#  	print(sims.shape)
#  	idxs = np.argpartition(sims, -20)[-20:]
#  	print('Most similar to '+ word + ':' )
#  	for idx in idxs:
#  		print(inv_dict[idx])
