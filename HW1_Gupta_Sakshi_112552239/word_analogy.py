import os
import pickle
import numpy as np
import sklearn
from scipy import spatial
import tensorflow as tf
import pandas as pd
model_path = './models/'
#loss_model = 'cross_entropy'
loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

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
if loss_model == 'nce':
    opFile = open("word_analogy_test_predictions_nce.txt","w")
else:
    opFile = open("word_analogy_test_predictions_cross_entropy.txt","w")

file = open("word_analogy_dev.txt", "r")
#d=[]
ans=""
for w in file:
    #dividing the samples and the test word pairs
    _,word=w.strip().split("||")
    test_words=word.strip().split(",")
    d=[]
    for w1 in test_words:
        test1,test2=w1.strip().split(":")
        v_1=test1[1:] #taking all after "
        v_2=test2[:-1] #taking all before "
        v1 = embeddings[dictionary[v_1]]
        v2 = embeddings[dictionary[v_2]]
        #d.append(np.dot(v1,v2))
        d.append(spatial.distance.cosine(v1,v2))
    
    ans=ans+word.replace(","," ")+" "+test_words[d.index(max(d))]+" "+test_words[d.index(min(d))]+"\n"

opFile.write(ans)
file.close()
opFile.close()
