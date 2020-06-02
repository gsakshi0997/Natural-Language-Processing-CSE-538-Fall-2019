import tensorflow as tf
import numpy as np
import math
def cross_entropy_loss(inputs, true_w):
  
    #==========================================================================

    #inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    #true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    #Write the code that calculate A = log(exp({u_o}^T v_c))
    
    int_A=tf.exp(tf.reduce_sum(tf.multiply(inputs,true_w),1))
    delta=math.exp(-14)
    A = tf.log(int_A+delta)
    
   
    int_B=tf.reduce_sum(tf.exp(tf.matmul(true_w,tf.transpose(inputs))),1)
    B =tf.log(int_B+delta)
    
    return tf.subtract(B, A)

    #==========================================================================
    
    
def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """
    
    
    #bias=tf.reshape(biases,[tf.size(biases),1])
    wo=tf.squeeze(tf.nn.embedding_lookup(weights,labels))
    bo=tf.squeeze(tf.nn.embedding_lookup(biases,labels))
    unigram_prob=tf.reshape(unigram_prob,[-1,1])
    prob_o=tf.squeeze(tf.nn.embedding_lookup(unigram_prob,labels))
    s_wo_wc=tf.reduce_sum(tf.multiply(inputs,wo),1)
    s_wo_wc_b=tf.add(bo,s_wo_wc)
    l_kwo=tf.log(tf.multiply(prob_o,len(sample)))
    sigma_o=tf.log(tf.sigmoid(tf.subtract(s_wo_wc_b,l_kwo)))            
    #s_wo_wc_b=tf.reshape(s_wo_wc_b,[1,tf.size(s_wo_wc_b)])
    
    
    #l_kwo=tf.reshape(l_kwo,[1,tf.size(l_kwo)])
    #l_kwo1=tf.reshape(l_kwo,[tf.size(s_wo_wc_b)[0],1])
    
    
    wx=tf.squeeze(tf.nn.embedding_lookup(weights,sample))
    bx=tf.squeeze(tf.nn.embedding_lookup(biases,sample))
    prob_x=tf.squeeze(tf.nn.embedding_lookup(unigram_prob,sample))
    s_wx_wc=tf.matmul(inputs,tf.transpose(wx))
    bx=tf.reshape(bx,[-1,1])
    s_wx_wc_b=tf.add(tf.transpose(s_wx_wc),bx)
    l_kwx=tf.log(tf.multiply(prob_x,len(sample)))
    l_kwx=tf.reshape(l_kwx,[-1,1])
    
    sig=tf.sigmoid(tf.subtract(s_wx_wc_b,l_kwx))
    ones=tf.ones(tf.shape(sig))
    sig1=tf.subtract(ones,sig)
    sigma_x=tf.log(sig1+math.exp(-14))
    
    ans=tf.add(sigma_o,sigma_x)
    return tf.negative(ans)
    