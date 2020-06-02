# std lib imports
from typing import Dict

# external libs
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models


class SequenceToVector(models.Model):
    """
    It is an abstract class defining SequenceToVector enocoder
    abstraction. To build you own SequenceToVector encoder, subclass
    this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    """
    def __init__(self,
                 input_dim: int) -> 'SequenceToVector':
        super(SequenceToVector, self).__init__()
        self._input_dim = input_dim

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Main Classifier.

        Parameters
        ----------
        vector_sequence : ``tf.Tensor``
            Sequence of embedded vectors of shape (batch_size, max_tokens_num, embedding_dim)
        sequence_mask : ``tf.Tensor``
            Boolean tensor of shape (batch_size, max_tokens_num). Entries with 1 indicate that
            token is a real token, and 0 indicate that it's a padding token.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.

        Returns
        -------
        An output dictionary consisting of:
        combined_vector : tf.Tensor
            A tensor of shape ``(batch_size, embedding_dim)`` representing vector
            compressed from sequence of vectors.
        layer_representations : tf.Tensor
            A tensor of shape ``(batch_size, num_layers, embedding_dim)``.
            For each layer, you typically have (batch_size, embedding_dim) combined
            vectors. This is a stack of them.
        """
        # ...
        # return {"combined_vector": combined_vector,
        #         "layer_representations": layer_representations}
        raise NotImplementedError


class DanSequenceToVector(SequenceToVector):
    """
    It is a class defining Deep Averaging Network based Sequence to Vector
    encoder. You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this DAN encoder.
    dropout : `float`
        Token dropout probability as described in the paper.
    """
    def __init__(self, input_dim: int, num_layers: int, dropout: float = 0.2):
        super(DanSequenceToVector, self).__init__(input_dim)
        # TODO(students): start

        self._layers = []
        self._layers.append(layers.Dense(self._input_dim, input_shape=(self._input_dim,), activation='relu'))

        num_layers -= 1

        while num_layers > 0:
            self._layers.append(layers.Dense(self._input_dim, activation='relu'))
            num_layers -= 1

        self._dropout = dropout

        # TODO(students): end

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start

        # Get dimensions
        batch_size, max_token, embedding_size = vector_sequence.get_shape().as_list()

        # Add 3rd dimenion
        sequence_mask_xd = tf.expand_dims(sequence_mask, 2)
        # Duplicate along 3rd dimension
        sequence_mask_3d = tf.tile(sequence_mask_xd, [1, 1, embedding_size])

        # words to be considered
        actual_words_mask = sequence_mask

        # Make embeddings of non valid words all zeros
        vector_sequence_valid = tf.math.multiply(vector_sequence, sequence_mask_3d)

        # print(max_token)

        #Add dropouts for training, greater than 2 to avoid dropout for bi-gram task
        if training and max_token > 2:
            # uni_distribution_np = np.random.randint(100, size=(batch_size, max_token))
            # p_distrubution_np = uni_distribution_np >= (self._dropout * 100)
            # p_distrubution = tf.convert_to_tensor(p_distrubution_np, dtype = vector_sequence.dtype)

            uni_distribution_np = tf.random.uniform((batch_size,max_token), minval=0, maxval=100, dtype=tf.float32)
            p_distrubution_np = uni_distribution_np >= (self._dropout * 100)
            p_distrubution = tf.cast(p_distrubution_np, vector_sequence.dtype)

            p_distrubution_xd = tf.expand_dims(p_distrubution, 2)
            p_distrubution_3d = tf.tile(p_distrubution_xd, [1, 1, embedding_size])

            actual_words_mask = tf.math.multiply(actual_words_mask, p_distrubution)

            # Make embedding of words as zeros when need to dropout
            vector_sequence_valid = tf.math.multiply(vector_sequence_valid, p_distrubution_3d)


        sum_representation = tf.reduce_sum(vector_sequence_valid, 1)

        word_counts = tf.reduce_sum(actual_words_mask, 1)

        # print(sum_representation.get_shape().as_list())
        # print(word_counts.get_shape().as_list())

        averaged_representation = sum_representation / tf.expand_dims(word_counts, 1)
        # print(averaged_representation.get_shape().as_list())

        layer_op = averaged_representation

        layer_representations = None

        for layer in self._layers:
            layer_op = layer(layer_op)
            current_representation = tf.expand_dims(layer_op, 1)

            if layer_representations is None:
                layer_representations = current_representation
            else:
                layer_representations = tf.concat([layer_representations, current_representation], axis = 1)

        combined_vector = layer_op

        # print(combined_vector.get_shape().as_list())
        # print(layer_representations.get_shape().as_list())

        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}


class GruSequenceToVector(SequenceToVector):
    """
    It is a class defining GRU based Sequence To Vector encoder.
    You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this GRU-based encoder. Note that each layer
        is a GRU encoder unlike DAN where they were feedforward based.
    """
    def __init__(self, input_dim: int, num_layers: int):
        super(GruSequenceToVector, self).__init__(input_dim)
        # TODO(students): start

        self._layers = []
        self._layers.append(layers.GRU(units = self._input_dim, return_sequences = True))

        num_layers -= 1

        while num_layers > 0:
            self._layers.append(layers.GRU(units = self._input_dim, return_sequences = True))
            num_layers -= 1

        # TODO(students): end

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start

        # firstop = self._layers[0](vector_sequence, mask = sequence_mask)
        # print(firstop[:,-1,:].get_shape().as_list())

        layer_op = vector_sequence
        layer_representations = None

        for layer in self._layers:
            layer_op = layer(layer_op, mask = sequence_mask, training = training)
            current_representation = tf.expand_dims(layer_op[:,-1,:], 1)

            if layer_representations is None:
                layer_representations = current_representation
            else:
                layer_representations = tf.concat([layer_representations, current_representation], axis = 1)
            # print(layer_representations.get_shape().as_list())

        combined_vector = layer_op[:,-1,:]

        # print(combined_vector.get_shape().as_list())
        # print(layer_representations.get_shape().as_list())

        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}
