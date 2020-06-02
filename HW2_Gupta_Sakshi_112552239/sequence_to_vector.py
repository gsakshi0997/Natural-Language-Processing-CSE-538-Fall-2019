# std lib imports
from typing import Dict
import pdb
# external libs
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
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
        self.dropout = dropout

        self.layers_dan = [tf.keras.layers.Dense(input_dim, activation='relu') for i in range(num_layers)]
        # TODO(students): end

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start
        a = sequence_mask
        if training:
            #print ("works")
            prob = tf.random.uniform(sequence_mask.shape)
            a=prob>self.dropout
        a = tf.expand_dims(a, axis=2)
        mat2 = vector_sequence * tf.cast(a , tf.float32)
        a_up = tf.reduce_sum(mat2 , 1)
        count1 = tf.cast(tf.where(sequence_mask == 1) , tf.float32)
        averg = tf.math.divide(a_up , len(count1))
        l = [averg]
        for i in range(len(self.layers_dan)):
            l1 = self.layers_dan[i](l[i])
            #print (l1)
            l.append(l1)
        combined_vector = l[-1]
        layer_representations = tf.stack(l, axis=1)
        #pdb.set_trace()
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
        self.num_layers = num_layers
        self.layers_gru = [tf.keras.layers.GRU(input_dim, return_sequences=True,return_state=True) for j in range(num_layers)]

        # TODO(students): end

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start
        l = []
        result = vector_sequence
        for i in range(self.num_layers):
            result,state = self.layers_gru[i](result, mask=sequence_mask)
            #pdb.set_trace()
            b=result[:,-1,:]
            l.append(b)
        combined_vector = l[-1]
        layer_representations=tf.stack(l,axis=1)
        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}
