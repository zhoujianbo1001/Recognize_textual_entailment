# -*- coding: UTF-8 -*-

import sys
import os
CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.dirname(CURRENT_PATH))
import tensorflow as tf
import modules.general as general
import modules.cnn as cnn
import modules.highway_network as hn
import modules.attention as attention
import modules.dense_net as dn
import modules.relation_network as rn
import modules.fcn as fcn
import params

CONFIGS = params.load_configs()

class DIIN(object):
    def __init__(self, embeddings=None):
        self.embeddings = embeddings

    def forward(
        self, 
        is_train,
        premise_x, 
        hypothesis_x, 
        premise_char, 
        hypothesis_char,
        premise_pos, 
        hypothesis_pos, 
        premise_exact_match, 
        hypothesis_exact_match):
        # Get lengths of unpadded sentences
        _, prem_mask = general.unpadded_length(premise_x)
        _, hyp_mask = general.unpadded_length(hypothesis_x)

        # Fucntion for embedding lookup and dropout at embedding layer
        def emb_drop(E, x):
            emb = tf.nn.embedding_lookup(E, x)
            emb_drop = tf.cond(
                is_train, lambda: tf.nn.dropout(emb, CONFIGS.emb_dropout_kp), lambda: emb
            )
            return emb_drop
        # Embedding layer
        with tf.variable_scope("word_emb"):
            if self.embeddings is None:
                E = tf.get_variable(
                    "word_emb_mat", shape=[CONFIGS.vocab_size+2, CONFIGS.emb_dim],
                    initializer=tf.glorot_normal_initializer(), trainable=True)
            else: 
                # PADDING and UNKNOWN
                pad_and_unknown = tf.get_variable(
                    "pad_and_unknown", shape=[2, CONFIGS.emb_dim],
                    initializer=tf.glorot_normal_initializer(), trainable=True)
                E = tf.Variable(self.embeddings, trainable=CONFIGS.emb_train)
                E = tf.concat([pad_and_unknown, E], axis=0)
            premise_in = emb_drop(E, premise_x)   #P
            hypothesis_in = emb_drop(E, hypothesis_x)  #H
        # self.debug = hypothesis_in

        # using characters embedding.
        if CONFIGS.use_char_emb:
            char_prem, char_hyp = apply_char_emb(is_train, premise_char, hypothesis_char)
            premise_in = tf.concat([premise_in, char_prem], axis=2)
            hypothesis_in = tf.concat([hypothesis_in, char_hyp], axis=2)
        # using pos
        if CONFIGS.use_pos:
            premise_in = tf.concat((premise_in, tf.cast(premise_pos, tf.float32)), axis=2)
            hypothesis_in = tf.concat((hypothesis_in, tf.cast(hypothesis_pos, tf.float32)), axis=2)
        # using em
        if CONFIGS.use_em:
            premise_in = tf.concat([premise_in, tf.cast(premise_exact_match, tf.float32)], axis=2)
            hypothesis_in = tf.concat([hypothesis_in, tf.cast(hypothesis_exact_match, tf.float32)], axis=2)

        # hightway network
        with tf.variable_scope("highway_network") as scope:
            # [batch_size, prem_len, dim]
            premise_in = hn.highway_network(
                premise_in, CONFIGS.hn_num_layers,
                CONFIGS.hn_out_size, is_train, CONFIGS.hn_dropout_kp, CONFIGS.weight_decay)
            scope.reuse_variables()
            # [batch_size, hyp_len, dim]
            hypothesis_in = hn.highway_network(
                hypothesis_in, CONFIGS.hn_num_layers,
                CONFIGS.hn_out_size, is_train, CONFIGS.hn_dropout_kp, CONFIGS.weight_decay)
        # self.debug = premise_in
        # self.debug = hypothesis_in

        # self attention
        with tf.variable_scope("self_attention") as scope:
            for i in range(CONFIGS.self_attention_layers):
                # [batch_size, prem_len, dim]
                premise_in = attention.attention(
                    premise_in, premise_in, prem_mask, "dot-product", False,
                    scope="prem_self_att_layer_{}".format(i))
                # [batch_size, hyp_len, dim]
                hypothesis_in = attention.attention(
                    hypothesis_in, hypothesis_in, hyp_mask, "dot-product", False,
                    scope="hypo_self_att_layer_{}".format(i))
        # self.debug = hypothesis_in
        
        fcn_out, self.fcn_in = apply_fcn(is_train, premise_in, hypothesis_in)
        self.model_out = fcn_out        
        self.debug = self.fcn_in

        self.logits = tf.nn.softmax(self.model_out, axis=-1)
        
                                
    def build_graph(self):
        self.prem_x = tf.placeholder(tf.int32, [None, CONFIGS.seq_len], name='premise')
        self.hyp_x = tf.placeholder(tf.int32, [None, CONFIGS.seq_len], name='hypothesis')
        self.prem_char = tf.placeholder(tf.int32, [None, None, CONFIGS.chars_len], name='premise_char')
        self.hyp_char = tf.placeholder(tf.int32, [None, None, CONFIGS.chars_len], name='hypothesis_char')
        self.prem_pos = tf.placeholder(tf.int32, [None, None, 47], name='premise_pos')
        self.hyp_pos = tf.placeholder(tf.int32, [None, None, 47], name='hypothesis_pos')
        self.prem_em = tf.placeholder(tf.int32, [None, None, 1], name='premise_exact_match')
        self.hyp_em = tf.placeholder(tf.int32, [None, None, 1], name='hypothesis_exact_match')
        
        self.y = tf.placeholder(tf.int32, [None], name='label_y')
        self.is_train = tf.placeholder('bool', [], name='is_train')

        self.forward(
            self.is_train, self.prem_x, self.hyp_x, self.prem_char, self.hyp_char,
            self.prem_pos, self.hyp_pos, self.prem_em, self.hyp_em)

    def build_loss(self, is_l2loss=True):
        self.losses = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.y, logits=self.model_out))
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
        grads, _= tf.clip_by_global_norm(tf.gradients(self.losses, tvars), gradient_clip_val)
        self.train_op = opt.apply_gradients(zip(grads, tvars), global_step=self.global_step)
        # self.train_op = opt.minimize(self.losses, global_step=global_step)

    def update(
        self, sess, label_y,
        premise_x, hypothesis_x, 
        premise_char, hypothesis_char,
        premise_pos, hypothesis_pos, 
        premise_exact_match, hypothesis_exact_match,
        is_train=True):
        debug, _, losses, logits, global_step = sess.run(
            [
                self.debug,
                self.train_op, 
                self.losses, 
                self.logits, 
                self.global_step
            ],
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
                self.is_train: is_train
            }
        )

        return losses, logits, global_step, debug

    def predict(
        self, sess,
        premise_x, hypothesis_x, 
        premise_char, hypothesis_char,
        premise_pos, hypothesis_pos, 
        premise_exact_match, hypothesis_exact_match,
        is_train=False):
        logits = sess.run(
            self.logits,
            {
                self.prem_x: premise_x,
                self.hyp_x: hypothesis_x,
                self.prem_char: premise_char,
                self.hyp_char: hypothesis_char,
                self.prem_pos: premise_pos,
                self.hyp_pos: hypothesis_pos,
                self.prem_em: premise_exact_match,
                self.hyp_em: hypothesis_exact_match,
                self.is_train: is_train
            }
        )

        return logits

def apply_char_emb(is_train, premise_char, hypothesis_char):
    with tf.variable_scope("char_emb"):
        char_emb_mat = tf.get_variable(
            "char_emb_mat", shape=[CONFIGS.chars_vocab_size+1, CONFIGS.chars_emb_dim],
            initializer=tf.glorot_normal_initializer(), trainable=True)
        char_pre = tf.nn.embedding_lookup(char_emb_mat, premise_char)
        char_hyp = tf.nn.embedding_lookup(char_emb_mat, hypothesis_char)

        assert sum(CONFIGS.chars_filters_out_channels) == CONFIGS.chars_out_size
        with tf.variable_scope("conv") as scope:
            conv_prem = cnn.multi_conv1d(
                char_pre, CONFIGS.chars_filters_width, CONFIGS.chars_filters_out_channels,
                "VALID", is_train, CONFIGS.chars_dropout_kp, CONFIGS.weight_decay, scope="conv")
            scope.reuse_variables()  
            # (batch_size, seq_len, char_out_size)
            conv_hyp = cnn.multi_conv1d(
                char_hyp, CONFIGS.chars_filters_width, CONFIGS.chars_filters_out_channels,
                "VALID", is_train, CONFIGS.chars_dropout_kp, CONFIGS.weight_decay, scope="conv")

        return conv_prem, conv_hyp

def apply_fcn(is_train, premise_in, hypothesis_in):
    # max pooling
    # [batch_size, dim]
    prem_max = tf.reduce_max(premise_in, 1)
    hyp_max = tf.reduce_max(hypothesis_in, 1)
    # mean pooling
    # [batch_size, dim]
    prem_mean = tf.reduce_mean(premise_in, 1)
    hyp_mean = tf.reduce_mean(hypothesis_in, 1)
    # concat
    # [batch_size, 4*dim]
    fcn_in = tf.concat([prem_max, hyp_max, prem_mean, hyp_mean], -1)

    # fcn_in = tf.contrib.layers.layer_norm(fcn_in)
    fcn_out = fcn.multi_denses(
        fcn_in, CONFIGS.label_size, CONFIGS.fcn_num_layers, func_activation=CONFIGS.fcn_func_activation,
        is_train=is_train, weight_decay=CONFIGS.weight_decay,
        dropout_p=CONFIGS.fcn_dropout_kp)

    return fcn_out, fcn_in

def apply_relation_network(is_train, premise_in, hypothesis_in):
    rn_out = rn.relation_network(
        premise_in, hypothesis_in, CONFIGS.rn_output_size,
        is_train, CONFIGS.weight_decay, CONFIGS.rn_dropout_p)
    return rn_out

def apply_dense_net(premise_in, hypothesis_in):
    premise_in = tf.expand_dims(premise_in, 2)
    hypothesis_in = tf.expand_dims(hypothesis_in, 2)
    premise_in = tf.tile(premise_in,[1,1,premise_in.shape[1], 1])
    hypothesis_in = tf.tile(hypothesis_in, [1,1,hypothesis_in.shape[1],1])

    dn_in = tf.multiply(premise_in, hypothesis_in)
    dn_out = dn.dense_net(
        inputs=dn_in, 
        out_size=CONFIGS.label_size, 
        first_scale_down_ratio=CONFIGS.dn_first_scale_down_ratio,
        first_scale_down_filter=CONFIGS.dn_first_scale_down_filter,
        num_blocks=CONFIGS.dn_num_blocks,
        grow_rate=CONFIGS.dn_grow_rate,
        num_block_layers=CONFIGS.dn_num_block_layers,
        filter_height=CONFIGS.dn_filter_height,
        filter_width=CONFIGS.dn_filter_width,
        transition_rate=CONFIGS.dn_transition_rate,
        scope="dense_net"
    )
    return dn_out, dn_in

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
    print(label_y)
    premise_x = np.array(premise_x)
    premise_x = np.pad(premise_x, ((0,0),(0,48-11)), "constant",constant_values=(0,0))
    hypothesis_x = np.array(hypothesis_x)
    hypothesis_x = np.pad(hypothesis_x, ((0,0),(0,48-11)), "constant",constant_values=(0,0))
    premise_char = np.array(premise_char)
    hypothesis_char = np.array(hypothesis_char)
    # print(label_y.shape)
    # print(premise_char.shape)
    # print(premise_x.shape)
    # print(hypothesis_char.shape)
    # print(hypothesis_x.shape)

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
    diin.build_train_op(lr=CONFIGS.lr)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            losses, logits, global_step, debug= diin.update(
                sess, label_y, premise_x, hypothesis_x,
                premise_char, hypothesis_char, premise_pos, premise_pos,
                premise_exact_match, hypothesis_exact_match, 
                is_train)
            print(debug, file=model_log)
            print("losses:", losses, file=model_log)
            print("losses:", losses)
            print("logits:", np.argmax(logits, axis=1), file=model_log)
            print("logits:", logits)
