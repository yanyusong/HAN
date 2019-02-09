import keras
import tensorflow as tf
import keras.backend as K
import numpy as np
from keras.layers import *
from keras.layers import Embedding, Input, Dense, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model
from config import Model_Config
from attention_layer import AttentionWithContext


class HAN_Model(Model_Config):
    def __init__(self, vocabulary_size):
        self.vocabulary_size = vocabulary_size

    def build_share_layers(self):
        # 加注意力机制
        # 词 到 句
        word_input = Input(shape=(self.sentence_max_word_num,), dtype='float32')
        word_sequences = Embedding(self.vocabulary_size, self.embedding_dim)(word_input)
        # word_input = Input(shape=(sentence_max_word_num,),dtype='float32')
        # word_sequences = Embedding(vocabulary_size,embedding_dim)(word_input)
        word_lstm = Bidirectional(CuDNNGRU(self.rnn_unit_num, return_sequences=True, kernel_regularizer=self.l2_reg))(
            word_sequences)
        # TimeDistributed 时间分布层，将一个 layer 应用到每一个时间序列上，一般用 Dense，实现LSTM每一个时间维度上的全连接，
        # 是一个变维度的过程，可以一对多，多对一。
        word_dense = TimeDistributed(Dense(self.td_fc_unit_num, kernel_regularizer=self.l2_reg))(
            word_lstm)  # batch_size , squence_length ,td_fc_unit_num
        word_att = AttentionWithContext()(word_dense)
        word_encoder = Model(word_input, word_att)  # batch_size ,  ？

        # 加注意力机制
        # 句 到 文章
        sent_input = Input(shape=[self.doc_max_sentence_num, self.sentence_max_word_num], dtype='int32')
        sent_encoder = TimeDistributed(word_encoder)(
            sent_input)  # 将句子的模型 应用到 文档中的每一个句子 batch_size, doc_max_sentence_num,sentence_vec_dim/注意力机制生成的句向量维度
        sent_lstm = Bidirectional(CuDNNGRU(self.rnn_unit_num, return_sequences=True, kernel_regularizer=self.l2_reg))(
            sent_encoder)  # output (batch_size, doc_max_sentence_num, rnn_unit_num*2)
        # 调整增大文章向量维度为 400
        sent_dense = TimeDistributed(Dense(self.td_doc_fc_unit_num, kernel_regularizer=self.l2_reg))(
            sent_lstm)  # 统一维度，将 layer 应用到每一个句向量上
        sent_att = Dropout(self.drop_rate)(AttentionWithContext()(sent_dense))
        return sent_att, sent_input

    def buid_mutil_layers(self, share_layers, mutil_layers_num):
        mutil_layers = []
        for i in range(mutil_layers_num):
            dense = Dense(self.fc_num, activation='relu', name='dense_{}'.format(i))(share_layers)
            dense_with_dropout = Dropout(self.drop_rate)(dense)
            layers = Dense(self.class_num, activation='softmax', name='preds_{}'.format(i))(
                dense_with_dropout)
            mutil_layers.append(layers)
        return mutil_layers

    def build(self, mutil_layers_num, optimizer, loss_dict, loss_weights_dict, metrics):
        share_layers, inputs = self.build_share_layers()
        outputs = self.buid_mutil_layers(share_layers, mutil_layers_num)
        model = Model(inputs=inputs,
                      outputs=outputs)
        model.compile(optimizer=optimizer,
                      loss=loss_dict,
                      loss_weights=loss_weights_dict,
                      metrics=metrics)
        model.summary()

        return model
