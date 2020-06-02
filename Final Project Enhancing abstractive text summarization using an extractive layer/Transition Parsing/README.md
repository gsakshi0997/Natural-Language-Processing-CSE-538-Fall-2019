## MODEL IMPLEMENTATION


### 1) parsing_system.py: apply

First, check if the transition is a shift. If yes, call configuration.shift(). 

If not, then parse the transition type and label. Then, get the top two elements on Stack. If left shift, add an arc from top word to second top word. Afterwards, remove the second top word from stack.

If right shift, add arc from second top word to top word. Then, pop the top stack word.


### 2) data.py: get_configuration_features

We construct a feature vector of size 48. It has 3 components, Sw, St and Sl.

Sw has the information of word token at either various positions on the stack/buffer or else, the left children/right children(or grandchildren) of said words. For this, we first use configuration get word_id and then the word. We use vocabulary to then get the universal if for this.

St is compromised of corresponding POS tag of word in Sw. First, we get pos tag of a word from configuration and then perform a lookup in vocabulary.

Sl is vector of length 12. It has labels of all but first 6 items in Sw. We get label using word token and then lookup with vocabulary to get label_id.


### 3) model.py : DependencyParser: __init__ and call

In the init, first declare a single embeddings variable, which is a vertical concatenation of Ew, Et and El. Hence, it has a size of (Vocab+unique_tag_count+unique_label_count x embedding dimension). The method argument passed is used to decide if this vector is trainable.

3 weight vectors, W1_x, W1_t and W1_l are initialized.  Dimensions are [hidden_dim x (embedd * 18(or 12))]. Bias variable is of size [hidden_dim]. 

Another variable of dimension [hidden_dim x num_transition] is the weight vector for the second layer.

For initial values, bias is initialized to all zeros. For embedding vector, a uniform distribution between -0.01 and 0.01 is used. For weights, we use a truncated normal of mean 0 and stddev 0.001.

In call function, input tensor is of shape [batch_size x 48]. First, we separate word, tags and label parts with indexing. Then, we perform lookup on embeddings vector to get 3d vector.  Next, we reshape the 3d to 2d vector so that the embeddings are horizontally concatenated to [batch_size x (embed_size x feature_size)].

Next, we multiply corresponding weights to the above obtained vector and add bias. We then pass the obtained vector through the requested activation layer. Next, we multiply this output to the second layer weight to obtain logits.


### 4) model.py : DependencyParser : compute_loss

To avoid nans, stable soft-max is implemented. For this, we first mask out all the invalid transitions in the logits. Invalid transitions are those that are -1 in the labels vector. Next, for every row, we subtract the elements from the maximum value. Next, we mask out the obtained vector again. Next, we take exponential and divide each element with the row sum.

Next, loss is computed. For this, for every row, obtain the logit value corresponding to the right transition. After taking a log for this value; for every row, add it to the total loss. We obtain the expected loss, by dividing total sum by number of rows. While taking log, adding a very small constant can help avid nans. Also, multiply -1 to final value.

For regularization value, square the 2 weight vectors, bias value and embedding vector. Sum all the elements of all mentioned above. Next multiply by regularization value and divide by 2.
