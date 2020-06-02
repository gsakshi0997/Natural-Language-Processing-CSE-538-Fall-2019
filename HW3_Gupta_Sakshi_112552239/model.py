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
        return tf.pow(vector,3)
        #raise NotImplementedError
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

        if activation_name == "cubic":
            self._activation = CubicActivation()
        elif activation_name == "sigmoid":
            self._activation = tf.keras.activations.sigmoid
        elif activation_name == "tanh":
            self._activation = tf.keras.activations.tanh
        else:
            raise Exception(f"activation_name: {activation_name} is from the known list.")

        # Trainable Variables
        # TODO(Students) Start
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.num_transitions = num_transitions
        self.trainable_embeddings = trainable_embeddings
        self.embedding_matrix = np.random.normal(size=(self.vocab_size, self.embedding_dim),scale=1. / math.sqrt(self.embedding_dim))
        self.embedding_matrix = np.asarray(self.embedding_matrix, dtype='float32')
        self.embeddings = tf.Variable(self.embedding_matrix,trainable=trainable_embeddings)
        self.W1 = tf.Variable(tf.random.truncated_normal(shape=(self.num_tokens*self.embedding_dim,self.hidden_dim), stddev = 0.005),trainable=True)
        self.W2 = tf.Variable(tf.random.truncated_normal(shape=(self.hidden_dim,self.num_transitions),stddev=0.005), trainable=True)
        self.b1 = tf.Variable(tf.zeros(shape=(1,self.hidden_dim)))
        # TODO(Students) End

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
        shape=self.embedding_dim*self.num_tokens
        emb_input=tf.reshape(tf.nn.embedding_lookup(self.embeddings,inputs),(-1,shape))
        we=tf.matmul(emb_input, self.W1)
        hidden_1=self._activation(tf.add(we,self.b1))


        logits=tf.matmul(hidden_1,self.W2)

        # TODO(Students) End
        output_dict = {"logits": logits}

        if labels is not None:
            output_dict["loss"] = self.compute_loss(logits, labels)
        return output_dict

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
        a=labels>-1
        mask=tf.cast(a,tf.float32)
        logits_max=tf.math.reduce_max(logits,axis=1)
        logits_new=tf.reshape(logits_max,(-1,1))
        logits=tf.math.exp(logits-logits_new)
        logits_mask=logits*mask
        logits_mask=tf.reduce_sum(logits_mask,axis=1)
        logits_mask=tf.reshape(logits_mask,(-1,1))
        # pt=logits/logits_mask

        label_act=tf.nn.relu(labels)
        label_act=tf.cast(label_act,tf.float32)
        logits_2=logits*label_act
        logits_num = tf.reshape(tf.reduce_sum(logits_2,1),(-1,1))
        pt=tf.math.log((logits_num/logits_mask)+1e-14)
        pt=tf.reduce_mean(pt)

        loss=tf.math.negative(pt)
        pam1=tf.nn.l2_loss(self.embeddings)
        pam2 =tf.nn.l2_loss(self.W1)
        pam3 = tf.nn.l2_loss(self.W2)
        pam4=tf.nn.l2_loss(self.b1)
        if self.trainable_embeddings:
            regularization=pam1+pam2+pam3+pam4
        else:
            regularization=pam2+pam3+pam4

        regularization=regularization*self._regularization_lambda
        # TODO(Students) End
        return loss + regularization
