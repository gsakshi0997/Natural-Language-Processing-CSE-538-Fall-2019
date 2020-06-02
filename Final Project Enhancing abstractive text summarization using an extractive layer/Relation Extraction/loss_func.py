import tensorflow as tf

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A = 


    And write the code that calculate B = log(sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """

    #v_x_u = tf.matmul(true_w, tf.transpose(inputs))

    A = tf.reduce_sum(tf.multiply(inputs, true_w), 1)

    v_x_u = tf.matmul(inputs, tf.transpose(true_w))

    temp_exp = tf.math.exp(v_x_u)
    temp_reduced = tf.reduce_sum(temp_exp, 1)
    temp_reduced_adjusted =  temp_reduced + 1e-10
    B = tf.math.log(temp_reduced_adjusted)

    return tf.subtract(B, A)

def nce_loss(inputs, weights, biases, labels, samples, unigram_prob):
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

    unigram_probability_tensor = tf.convert_to_tensor(unigram_prob, dtype=tf.float32)

    ###############################  Negative ##################
    weights_for_neg = tf.nn.embedding_lookup(weights, samples)
    # print(weights_for_neg.get_shape)

    s_wx_wc_pre_b = tf.matmul(inputs, tf.transpose(weights_for_neg))
    # print(s_wx_wc_pre_b.get_shape)

    negative_bias = tf.nn.embedding_lookup(biases, samples)  
    negative_bias = tf.reshape(negative_bias, [-1]) 
    s_wx_wc = tf.nn.bias_add(s_wx_wc_pre_b, negative_bias)

    negative_unigram_prob = tf.nn.embedding_lookup(unigram_probability_tensor, samples)


    Pr_x_c = tf.sigmoid(tf.subtract(s_wx_wc, tf.log(1e-10 + tf.scalar_mul(len(samples), negative_unigram_prob))))


    ###############################  Positive ##################

    weights_for_pos = tf.nn.embedding_lookup(weights, tf.squeeze(labels))
    # print(weights_for_pos.get_shape)

    s_wx_wo_pre_b = tf.matmul(inputs, tf.transpose(weights_for_pos))
    # print(s_wx_wo_pre_b.get_shape)

    positive_bias = tf.nn.embedding_lookup(biases, labels)  
    positive_bias = tf.reshape(positive_bias, [-1]) 
    s_wo_wc = tf.nn.bias_add(s_wx_wo_pre_b, positive_bias)


    positive_unigram_prob = tf.nn.embedding_lookup(unigram_probability_tensor, tf.squeeze(labels))

    Pr_x_o = tf.sigmoid(tf.subtract(s_wo_wc, tf.log(1e-10 + tf.scalar_mul(len(samples), positive_unigram_prob))))

    ###############################  Final ##################

    final_left = a = tf.log(Pr_x_o + 1e-10) 
    final_right = tf.reduce_sum(tf.log(1 - Pr_x_c + 1e-10), 1)

    # print(final_right.get_shape)

    return tf.scalar_mul(-1, tf.add(final_left, final_right))



    