# -*- coding: UTF-8 -*-

import sys
import os
CURRENT_PATH = os.path.dirname(__file__)
sys.path.append(os.path.dirname(CURRENT_PATH))
import tensorflow as tf
import modules.general as general
import modules.cnn as cnn
import modules.highway_network as hn
import modules.attention as attention
import modules.relation_network as rn
import modules.fcn as fcn


class DIIN(object):
    def __init__(
        self, 
        emb_train=True, 
        embeddings=None,
        vocab_size=100,##
        emb_dim=50,##
        chars_vocab_size=50,##
        chars_emb_dim=50,##
        filters_out_channels=[100],
        filters_width=[5],
        char_out_size=100,
        weight_decay=0.0,
        highway_num_layers=2,
        self_attention_layers=1,
        label_size=3,

        seq_len=48, 
        chars_len=16):
        self.emb_train = emb_train
        self.embeddings = embeddings
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.chars_vocab_size = chars_vocab_size
        self.chars_emb_dim = chars_emb_dim
        self.filters_out_channels = filters_out_channels
        self.filters_width = filters_width
        self.char_out_size = char_out_size
        self.weight_decay = weight_decay
        self.highway_num_layers = highway_num_layers
        self.self_attention_layers = self_attention_layers
        self.label_size = label_size

        self.seq_len = seq_len
        self.chars_len = chars_len

    def forward(
        self, 
        premise_x, 
        hypothesis_x, 
        premise_char, 
        hypothesis_char,
        premise_pos, 
        hypothesis_pos, 
        premise_exact_match, 
        hypothesis_exact_match,
        is_train, 
        dropout_p):
        # Fucntion for embedding lookup and dropout at embedding layer
        def emb_drop(E, x):
            emb = tf.nn.embedding_lookup(E, x)
            emb_drop = tf.cond(
                is_train, lambda: tf.nn.dropout(emb, dropout_p), lambda: emb
            )
            return emb_drop

        # Get lengths of unpadded sentences    
        _, prem_mask = general.unpadded_length(premise_x)
        _, hyp_mask = general.unpadded_length(hypothesis_x)

        # Embedding layer
        with tf.variable_scope("word_emb"):
            if self.embeddings is None:
                E = tf.get_variable(
                    "word_emb_mat", shape=[self.vocab_size+2, self.emb_dim],
                    initializer=tf.glorot_normal_initializer(), trainable=True)
            else: 
                # PADDING and UNKNOWN
                pad_and_unknown = tf.get_variable(
                    "pad_and_unknown", shape=[2, self.emb_dim],
                    initializer=tf.glorot_normal_initializer(), trainable=True)
                E = tf.Variable(self.embeddings, trainable=self.emb_train)
                E = tf.concat([pad_and_unknown, E], axis=0)
            premise_in = emb_drop(E, premise_x)   #P
            hypothesis_in = emb_drop(E, hypothesis_x)  #H
        
        with tf.variable_scope("char_emb"):
            char_emb_mat = tf.get_variable(
                "char_emb_mat", shape=[self.chars_vocab_size+1, self.chars_emb_dim],
                initializer=tf.glorot_normal_initializer(), trainable=True)
            char_pre = tf.nn.embedding_lookup(char_emb_mat, premise_char)
            char_hyp = tf.nn.embedding_lookup(char_emb_mat, hypothesis_char)

            assert sum(self.filters_out_channels) == self.char_out_size
            with tf.variable_scope("conv") as scope:
                conv_pre = cnn.multi_conv1d(
                    char_pre, self.filters_width, self.filters_out_channels,
                    "VALID", is_train, dropout_p, self.weight_decay, scope="conv")
                scope.reuse_variables()  
                # (batch_size, seq_len, char_out_size)
                conv_hyp = cnn.multi_conv1d(
                    char_hyp, self.filters_width, self.filters_out_channels,
                    "VALID", is_train, dropout_p, self.weight_decay, scope="conv")

            premise_in = tf.concat([premise_in, conv_pre], axis=2)
            hypothesis_in = tf.concat([hypothesis_in, conv_hyp], axis=2)

        # premise_in = tf.concat((premise_in, tf.cast(premise_pos, tf.float32)), axis=2)
        # hypothesis_in = tf.concat((hypothesis_in, tf.cast(hypothesis_pos, tf.float32)), axis=2)

        # premise_in = tf.concat([premise_in, tf.cast(premise_exact_match, tf.float32)], axis=2)
        # hypothesis_in = tf.concat([hypothesis_in, tf.cast(hypothesis_exact_match, tf.float32)], axis=2)

        with tf.variable_scope("highway_network") as scope:
            premise_in = hn.highway_network(
                premise_in, self.highway_num_layers,
                None, is_train, dropout_p, self.weight_decay)
            scope.reuse_variables()
            hypothesis_in = hn.highway_network(
                hypothesis_in, self.highway_num_layers,
                None, is_train, dropout_p, self.weight_decay)

        with tf.variable_scope("self_attention") as scope:
            for i in range(self.self_attention_layers):
                premise_in = attention.self_attention(
                    premise_in, prem_mask, "dot-product",
                    is_train, self.weight_decay, dropout_p,
                    scope="prem_self_att_layer_{}".format(i))
                hypothesis_in = attention.self_attention(
                    hypothesis_in, hyp_mask, "dot-product",
                    is_train, self.weight_decay, dropout_p,
                    scope="hypo_self_att_layer_{}".format(i))

        rn_out = rn.relation_network(
            premise_in, hypothesis_in, self.label_size,
            self.weight_decay, is_train, dropout_p)
        
        self.logits = fcn.multi_denses(rn_out, self.label_size, num_layers=3)
        
        self.predict = tf.nn.softmax(self.logits, axis=-1)
        
                                
    def build_graph(self):
        self.prem_x = tf.placeholder(tf.int32, [None, None], name='premise')
        self.hyp_x = tf.placeholder(tf.int32, [None, None], name='hypothesis')
        self.prem_char = tf.placeholder(tf.int32, [None, None, None], name='premise_char')
        self.hyp_char = tf.placeholder(tf.int32, [None, None, None], name='hypothesis_char')
        self.prem_pos = tf.placeholder(tf.int32, [None, None, 47], name='premise_pos')
        self.hyp_pos = tf.placeholder(tf.int32, [None, None, 47], name='hypothesis_pos')
        self.prem_em = tf.placeholder(tf.int32, [None, None, 1], name='premise_exact_match')
        self.hyp_em = tf.placeholder(tf.int32, [None, None, 1], name='hypothesis_exact_match')

        self.y = tf.placeholder(tf.int32, [None], name='label_y')

        self.dropout_p = tf.placeholder(tf.float32, [], name='drop_keep_rate')
        self.is_train = tf.placeholder('bool', [], name='is_train')

        self.forward(
            self.prem_x, self.hyp_x, self.prem_char, self.hyp_char,
            self.prem_pos, self.hyp_pos, self.prem_em, self.hyp_em,
            self.is_train, self.dropout_p)

        ################################################
        # self.debug = tf.Print(self.logits, [self.is_train], "debug message:\n", summarize=100)
        ################################################

    def build_loss(self, is_l2loss=True):
        self.losses = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.y, logits=self.logits))
        if is_l2loss:
            pass

    def build_train_op(
        self, opt=None, lr=1e-3, 
        gradient_clip_val=1, global_step=None):
        if opt is None:
            opt = tf.train.AdamOptimizer(lr)
            # opt = tf.train.AdadeltaOptimizer(lr)
        
        if global_step is None:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.losses, tvars), gradient_clip_val)
        self.train_op = opt.apply_gradients(zip(grads, tvars), global_step=self.global_step)

        # self.train_op = opt.minimize(self.losses, global_step=global_step)

    def update(
        self, sess, label_y,
        premise_x, hypothesis_x, 
        premise_char, hypothesis_char,
        premise_pos, hypothesis_pos, 
        premise_exact_match, hypothesis_exact_match,
        is_train=True, dropout_p=0.5):
        _, losses, logits, global_step = sess.run(
            [
                # self.debug,
                self.train_op, self.losses, self.predict, self.global_step],
            {   
                self.y: label_y,
                self.prem_x: premise_x,
                self.hyp_x: hypothesis_x,
                self.prem_char: premise_char,
                self.hyp_char: hypothesis_char,
                self.prem_pos: premise_pos,
                self.hyp_pos: hypothesis_pos,
                self.prem_em: premise_exact_match,
                self.hyp_em: hypothesis_exact_match,
                self.is_train: is_train,
                self.dropout_p: dropout_p
            }
        )

        return losses, logits, global_step

    def predict(
        self, sess,
        premise_x, hypothesis_x, 
        premise_char, hypothesis_char,
        premise_pos, hypothesis_pos, 
        premise_exact_match, hypothesis_exact_match,
        is_train=False, dropout_p=0.5):
        logits = sess.run(
            [self.predict],
            {
                self.prem_x: premise_x,
                self.hyp_x: hypothesis_x,
                self.prem_char: premise_char,
                self.hyp_char: hypothesis_char,
                self.prem_pos: premise_pos,
                self.hyp_pos: hypothesis_pos,
                self.prem_em: premise_exact_match,
                self.hyp_em: hypothesis_exact_match,
                self.is_train: is_train,
                self.dropout_p: dropout_p
            }
        )

        return logits


import numpy as np


if __name__ == "__main__":
    a_little_data = open(sys.path[0]+"/a_little_data.txt", 'r')
    is_word = False
    is_char = False
    is_label = False
    word_dict = {}
    char_dict = {}
    label_y = []
    premise = []
    hypothesis = []
    premise_x = []
    hypothesis_x = []
    tag = 0
    for line in a_little_data:
        
        if line.strip("\n") == "WORD_DICT":
            is_word = True
            is_char = False
            is_label = False
            continue
        if line.strip("\n") == "CHAR_DICT":
            is_char = True
            is_word = False
            is_label = False
            continue
        if line.strip("\n") == "DATA":
            is_char = False
            is_label = True
            is_word = False
            continue

        if is_word and line != "\n":
            word_dict[line.split(" ")[0]] = int(line.split(" ")[1])
        if is_char and line != "\n":
            char_dict[line.split(" ")[0]] = int(line.split(" ")[1])
        if is_label and line != "\n":
            if tag%5 == 0:
                label_y.append(int(line.split(" ")[1]))
            elif tag%5 == 1:
                premise.append(line.strip("\n"))
            elif tag%5 == 2:
                hypothesis.append(line.strip("\n"))
            elif tag%5 == 3:
                premise_x.append(eval(line))
            else:
                hypothesis_x.append(eval(line))
            tag += 1

    premise_char = []  
    # premise_char
    for line in premise:
        words_list = []
        for word in line.split(' '):
            chars_list = []
            for char in word:
                chars_list.append(char_dict[char.lower()])
            if len(chars_list) < 16:
                chars_list = chars_list + [0]*(16-len(chars_list))
            words_list.append(chars_list)
        if len(words_list) < 48:
            words_list = words_list + [[0]*16]*(48-len(words_list))
        premise_char.append(words_list)
        
    
    hypothesis_char = []   
    # hypothesis_char
    for line in hypothesis:
        words_list = []
        for word in line.split(' '):
            chars_list = []
            for char in word:
                chars_list.append(char_dict[char.lower()])
            if len(chars_list) < 16:
                chars_list = chars_list + [0]*(16-len(chars_list))
            words_list.append(chars_list)
        if len(words_list) < 48:
            words_list = words_list + [[0]*16]*(48-len(words_list))
        hypothesis_char.append(words_list)

    
    label_y = np.array(label_y)
    premise_x = np.array(premise_x)
    premise_x = np.pad(premise_x, ((0,0),(0,48-11)), "constant",constant_values=(0,0))
    hypothesis_x = np.array(hypothesis_x)
    hypothesis_x = np.pad(hypothesis_x, ((0,0),(0,48-11)), "constant",constant_values=(0,0))
    premise_char = np.array(premise_char)
    hypothesis_char = np.array(hypothesis_char)
    print(label_y.shape)
    print(premise_char.shape)
    print(premise_x.shape)
    print(hypothesis_char.shape)
    print(hypothesis_x.shape)

    batch_size = 3
    max_seq_len = 48
    max_chars_len = 16
    premise_pos = np.random.randint(low=0, high=100, size=(batch_size, max_seq_len, 47), dtype=np.int32)
    hypothesis_pos = np.random.randint(low=0, high=100, size=(batch_size, max_seq_len, 47), dtype=np.int32)
    premise_exact_match = np.random.randint(low=0, high=2, size=(batch_size, max_seq_len, 1), dtype=np.int32)
    hypothesis_exact_match = np.random.randint(low=0, high=2, size=(batch_size, max_seq_len,1), dtype=np.int32)
    is_train = True
    dropout_p = 0.5

    model_log = open(CURRENT_PATH+"/model.log","w")

    diin = DIIN()
    diin.build_graph()
    diin.build_loss()
    diin.build_train_op()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            losses, logits= diin.update(
                sess, label_y, premise_x, hypothesis_x,
                premise_char, hypothesis_char, premise_pos, premise_pos,
                premise_exact_match, hypothesis_exact_match, 
                is_train, dropout_p)
            print("losses:", losses, file=model_log)
            print("logits:", np.argmax(logits, axis=1), file=model_log)
