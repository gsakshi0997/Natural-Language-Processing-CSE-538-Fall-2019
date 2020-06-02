## Implementation

### a. word2vec_basic.py : 
	
This file is the main class that takes in a text corpus and generates a trained model. The "generate_batch" function is implemented as follows. First, we define a variable that is refelective of the current number of input-output pairs added in the batch. A while loop ensures that this number does not cross the batch_size passed in to the method.

Inside the loop, we first derive the index of the central word such that this word has skip_window neighbours to the left atleast. Then, we consider skip_windows left and right to form pair, ensuring that we dont pick more than num_skips.


### b. loss_fun.py

For cross_entropy loss, we first calculate A, ensuring we only multiply corresponding vectors, with the use of element wise multiplication and then summing to get a column vector. For B, since we need to multiply all combinations of vectors, we use matrix multiplication (after transpose) to get a (batch_size, batch_size) product. After using the exponenetial function on each element, we sum to get a column vector. We then take a log to arrive at B. To avoid nan issue when taking, a small constant value is added.

For nce loss, we first calculate similarity scores of batch values with negative(samples) and then positive values(labels). To get the weight vectors from vocabulary, we use embedding lookup. We then use these terms to calculate Pr(D = 1;wo|wc) and Pr(D = 1;wo|wc). again, to avoid nan from log(0), we add a small constant value. Finally, using the formula specified, we calculate the loss function.

### c. word_analogy.py

First, read the parameters from the specified model file. Then read the file with the samples. For every line read, split the three examples and then calculate the average vector. Then for the 4 options, find the vector which has most and least cosine similarity. Output to a file with the necessary format.

