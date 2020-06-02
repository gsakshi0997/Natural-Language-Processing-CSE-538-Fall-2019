# inbuilt lib imports:
from typing import Dict
import math

# external libs
import tensorflow as tf
from tensorflow.keras import models, layers

import numpy as np

# project imports


class CubicActivation(layers.Layer):
    """
    Cubic activation as described in the paper.
    """
    def call(self, vector: tf.Tensor) -> tf.Tensor:
        """
        Parameters
        ----------
        vector : ``tf.Tensor``
            hidden vector of dimension (batch_size, hidden_dim)

        Returns tensor after applying the activation
        """
        # TODO(Students) Start
        # Comment the next line after implementing call.

        return tf.pow(vector, 3)

        # TODO(Students) End


class DependencyParser(models.Model):
    def __init__(self,
                 embedding_dim: int,
                 vocab_size: int,
                 num_tokens: int,
                 hidden_dim: int,
                 num_transitions: int,
                 regularization_lambda: float,
                 trainable_embeddings: bool,
                 activation_name: str = "cubic") -> None:
        """
        This model defines a transition-based dependency parser which makes
        use of a classifier powered by a neural network. The neural network
        accepts distributed representation inputs: dense, continuous
        representations of words, their part of speech tags, and the labels
        which connect words in a partial dependency parse.

        This is an implementation of the method described in

        Danqi Chen and Christopher Manning.
        A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

        Parameters
        ----------
        embedding_dim : ``str``
            Dimension of word embeddings
        vocab_size : ``int``
            Number of words in the vocabulary.
        num_tokens : ``int``
            Number of tokens (words/pos) to be used for features
            for this configuration.
        hidden_dim : ``int``
            Hidden dimension of feedforward network
        num_transitions : ``int``
            Number of transitions to choose from.
        regularization_lambda : ``float``
            Regularization loss fraction lambda as given in paper.
        trainable_embeddings : `bool`
            Is the embedding matrix trainable or not.
        """
        super(DependencyParser, self).__init__()
        self._regularization_lambda = regularization_lambda
        self._embedding_dim = embedding_dim

        if activation_name == "cubic":
            self._activation = CubicActivation()
        elif activation_name == "sigmoid":
            self._activation = tf.keras.activations.sigmoid
        elif activation_name == "tanh":
            self._activation = tf.keras.activations.tanh
        else:
            raise Exception(f"activation_name: {activation_name} is from the known list.")

        # Trainable Variables
        # TODO(Students)

        /**** Default ****/
        self.embeddings = tf.Variable(tf.random.uniform(shape = (vocab_size, embedding_dim), minval = -0.01, maxval = 0.01, dtype=tf.dtypes.float32), trainable = trainable_embeddings)

        self._w1_x = tf.Variable(tf.random.truncated_normal([hidden_dim, embedding_dim * 18], mean=0.0, stddev=0.01))
        self._w1_t = tf.Variable(tf.random.truncated_normal([hidden_dim, embedding_dim * 18], mean=0.0, stddev=0.01))
        self._w1_l = tf.Variable(tf.random.truncated_normal([hidden_dim, embedding_dim * 12], mean=0.0, stddev=0.01))
        self._b1 = tf.Variable(tf.zeros([hidden_dim]))

        self._w2 = tf.Variable(tf.random.truncated_normal([hidden_dim, num_transitions], mean=0.0, stddev=0.01))


        /**** Best ****/
        # self.embeddings = tf.Variable(tf.random.uniform(shape = (vocab_size, embedding_dim), minval = -0.001, maxval = 0.001, dtype=tf.dtypes.float32), trainable = trainable_embeddings)
        #
        # self._w1_x = tf.Variable(tf.random.truncated_normal([hidden_dim, embedding_dim * 18], mean=0.0, stddev=0.001))
        # self._w1_t = tf.Variable(tf.random.truncated_normal([hidden_dim, embedding_dim * 18], mean=0.0, stddev=0.001))
        # self._w1_l = tf.Variable(tf.random.truncated_normal([hidden_dim, embedding_dim * 12], mean=0.0, stddev=0.001))
        # self._b1 = tf.Variable(tf.zeros([hidden_dim]))
        #
        # self._w2 = tf.Variable(tf.random.truncated_normal([hidden_dim, num_transitions], mean=0.0, stddev=0.001))

        # TODO(Students) End

    def stable_softmax(self, X, labels):

        # print(labels)

        label_mask1 = tf.equal(labels, 1)
        mask1 = tf.cast(label_mask1, tf.float32)

        label_mask2 = tf.equal(labels, 0)
        mask2 = tf.cast(label_mask2, tf.float32)

        mask = mask1 + mask2

        shape = X.get_shape().as_list()
        trans = shape[1]

        # print(mask)
        X_masked = tf.math.multiply(X, mask)
        maxes = tf.reduce_max(X_masked, axis=1)
        maxes = tf.tile(tf.expand_dims(maxes, axis=-1), multiples=[1,trans])
        X_adj = X_masked - maxes
        X_adj = tf.math.multiply(X_adj, mask)

        X_exp = tf.math.exp(X_adj)
        # print(X_exp)

        sums = tf.reduce_sum(X_exp, axis=1)
        sums = tf.tile(tf.expand_dims(sums, axis=-1), multiples=[1,trans])
        # print(maxes.get_shape().as_list())

        return X_exp /sums;

    def call(self,
             inputs: tf.Tensor,
             labels: tf.Tensor = None) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Dependency Parser.

        Parameters
        ----------
        inputs : ``tf.Tensor``
            Tensorized version of the batched input text. It is of shape:
            (batch_size, num_tokens) and entries are indices of tokens
            in to the vocabulary. These tokens can be word or pos tag.
            Each row corresponds to input features a configuration.
        labels : ``tf.Tensor``
            Tensor of shape (batch_size, num_transitions)
            Each row corresponds to the correct transition that
            should be made in the given configuration.

        Returns
        -------
        An output dictionary consisting of:
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start

        tf_embeddings = self.embeddings
        batch_size = inputs.get_shape().as_list()[0]

        Xws = inputs[:, :18]
        Xts = inputs[:, 18:36]
        Xls = inputs[:, 36:]

        # print(Xws.get_shape().as_list())
        # print(Xts.get_shape().as_list())
        # print(Xls.get_shape().as_list())

        lookups = tf.nn.embedding_lookup(tf_embeddings, Xws)
        # print(test.get_shape().as_list())
        Xw_embed = tf.reshape(lookups, (batch_size, 18 * self._embedding_dim))

        lookups = tf.nn.embedding_lookup(tf_embeddings, Xts)
        # print(test.get_shape().as_list())
        Xt_embed = tf.reshape(lookups, (batch_size, 18 * self._embedding_dim))

        lookups = tf.nn.embedding_lookup(tf_embeddings, Xls)
        # print(test.get_shape().as_list())
        Xl_embed = tf.reshape(lookups, (batch_size, 12 * self._embedding_dim))

        temp = tf.matmul(Xw_embed, self._w1_x, transpose_b = True) + tf.matmul(Xt_embed, self._w1_t, transpose_b = True)  + tf.matmul(Xl_embed, self._w1_l, transpose_b = True)
        temp2 = tf.nn.bias_add(temp, self._b1)
        h = self._activation(temp2)

        logits = tf.matmul(h, self._w2)
        # print(logits)

        output_dict = {"logits": logits}

        if labels is not None:
            output_dict["loss"] = self.compute_loss(logits, labels)
        return output_dict
        # TODO(Students) End

    def compute_loss(self, logits: tf.Tensor, labels: tf.Tensor) -> tf.float32:
        """
        Parameters
        ----------
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.

        Returns
        -------
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start

        logits = self.stable_softmax(logits, labels)

        label_mask = tf.equal(labels, 1)
        # print(label_mask)
        mask = tf.cast(label_mask, tf.float32)
        logged = tf.math.log(logits + 1e-15)
        # print(logged)
        loss = tf.reduce_sum(tf.math.multiply(logged, mask))

        loss = -1 * loss/logits.get_shape().as_list()[0]

        regularization = 0
        regularization += tf.reduce_sum(tf.pow(self._w1_x, 2))
        regularization += tf.reduce_sum(tf.pow(self._w1_t, 2))
        regularization += tf.reduce_sum(tf.pow(self._w1_l, 2))
        regularization += tf.reduce_sum(tf.pow(self._b1, 2))
        regularization += tf.reduce_sum(tf.pow(self._w2, 2))
        regularization += tf.reduce_sum(tf.pow(self.embeddings, 2))
        regularization = regularization * self._regularization_lambda / 2

        # TODO(Students) End
        return loss + regularization
