## MODEL IMPLEMENTATION


### 1. Deep Averaging Network :

To implement DAN, the class “DanSequenceToVector” needs to be modified. First, in the “__init__” function, we create “num_layers” number of “Dense” layers. ReLu activation is used. After averaging, the output is the same size as embedding dimension of a word token. Hence, all the dense layers, take in and output a vector of “embedding” dimension. 

In the “call” function, we first read the shape of “vector_sequence” to identify “batch_size”, “max_token_number” and “embedding_size”.  Next, we apply sequence mask to the vector sequence. However, since the mask is a 2D vector, we expand dimensions along the embedding dimension. We tile all the values(0 or 1) along the embedding dimension. Then, we use matrix multiplication to make embeddings of invalid words all zeroes.

Next, we need to consider dropouts for training phase. “tf.random.uniform” is used to generate an array of size (batch_size max_token). The range of values are uniformly distributed over range 0 to 99. Next, to have dropout probability of p(0 >= p >= 1), we compare all elements whose value is greater than or equal to p*100. Elements whose values are less than this value are those who will be dropped out. Hence, these indexes are set to 0. Other values are set to 1. After tiling the dropout vector in 3d, we multiply it with vector_sequence to obtain the final input.
 
We then use reduce_sum to average along the word_token dimension. We then divide by the number of actual words, using mask, for every row. This gives the averaged vector per sentence. 

Next, we pass this vector through the first dense layer. The output of first dense layer is input tot the second and so on. As we progress, we store the output of each layer, to be used in the probing model, for a later task.


### 2. Gated Recurrent Unit :

Similar to DAN, we first use the “init” function to build “num_layers” number of GRU modules. We set “return_sequences” to true for each, since, we need all the hidden states to pass to the next GRU layer.
In “call”, we pass the “vector_sequence” to the first GRU layer, we then pass all the returned sequences as input to the next GRU layer. Also, at each, step, we obtain the output from the final GRU cell and using all such outputs, across layers, we construct the “layer_representations”. The output from last layer and last GRU cell is the “combined_vector”


### 3. Probing Model :

In the “init” function, a single dense layer is created (in addition to pre-existing code). No non-linear activations are used. The number of outputs is same as “classes_num”.

In the call function, input is passed through the pretrained model. Then, we get the “layer_representations” from the output. Then, we pick the layer_representation which corresponds to the “layer_num” passed in “init”. This representation is then passed through the single dense layer to get the logits output.
