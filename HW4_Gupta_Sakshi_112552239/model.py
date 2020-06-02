import tensorflow as tf
from tensorflow.keras import layers, models

from util import ID_TO_CLASS


class MyBasicAttentiveBiGRU(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
        super(MyBasicAttentiveBiGRU, self).__init__()

        self.num_classes = len(ID_TO_CLASS)

        self.decoder = layers.Dense(units=self.num_classes)
        self.omegas = tf.Variable(tf.random.normal((hidden_size*2, 1)))
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))

        ### TODO(Students) START
        self.gru_layer=tf.keras.layers.GRU(hidden_size,return_sequences=True)    #initializing the gru layer
        self.bi_gru=tf.keras.layers.Bidirectional(self.gru_layer)   #initializing the RNN
        ### TODO(Students) END

    def attn(self, rnn_outputs):
        ### TODO(Students) START
        rnn_outputs1=tf.tanh(rnn_outputs) #applying tanh to the concatenated embeddings
        score_word=tf.tensordot(rnn_outputs1 , self.omegas,axes=1) #score for each word
        score_norm=tf.nn.softmax(score_word , axis=1)     #normalizing score

        embed_prod=score_norm * rnn_outputs
        embed_prod=tf.reduce_sum(embed_prod,axis=1)
        output=tf.tanh(embed_prod)
        ### TODO(Students) END

        return output

    def call(self, inputs , pos_inputs , training):
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)

        ### TODO(Students) START
        cond=inputs!=0
        input=tf.concat([word_embed , pos_embed],axis=2)
        #input=word_embed
        out=self.bi_gru(input , mask=cond)  #passing concatenated embeddings to the bidirectional RNN
        out1=self.attn(out) #Applying attention
        logits=self.decoder(out1)
        ### TODO(Students) END

        return {'logits': logits}


class MyAdvancedModel(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, training: bool = False):
        super(MyAdvancedModel, self).__init__()
        ### TODO(Students) START
        self.num_classes = len(ID_TO_CLASS)
        self.decoder = layers.Dense(units=self.num_classes)
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))
        self.layer_1=tf.keras.layers.Conv1D(filters=64,kernel_size=2,activation=tf.nn.relu)
        self.layer_2=tf.keras.layers.Conv1D(filters=64,kernel_size=2,activation=tf.nn.relu)
        self.dropout=tf.keras.layers.Dropout(rate=0.5)
        ### TODO(Students END
    def maxpool(self,inp,pool_size1:int):

        out = tf.keras.layers.MaxPool1D(pool_size=pool_size1)(inp)
        return out

    def flatten(self,inp):
        out=tf.keras.layers.Flatten()(inp)
        return out
    def call(self, inputs, pos_inputs, training):
        #raise NotImplementedError
        ### TODO(Students) START
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)

        input=tf.concat([word_embed , pos_embed],axis=2)
        pool_size1=input.shape[1]-1

        out1=self.layer_1(input)
        out2=self.layer_2(input)

        out1=self.maxpool(out1,pool_size1)
        out2=self.maxpool(out2,pool_size1)
        out1=self.dropout(out1)
        out2=self.dropout(out2)
        final_out=tf.concat([out1,out2],axis=2)

        final_out=self.flatten(final_out)
        logits=self.decoder(final_out)

     ### TODO(Students END
        return {'logits': logits}