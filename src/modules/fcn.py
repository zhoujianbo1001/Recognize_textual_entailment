# -*- coding: utf-8 -*-

import tensorflow as tf

def multi_denses(
    inputs, out_size, num_layers, weight_decay=0.0, 
    is_train=False, dropout_p=0.5, scope="multi_denses"):
    with tf.variable_scope(scope):
        units = inputs.get_shape().as_list()[-1]
        for i in range(num_layers):
            if i == num_layers-1:
                units = out_size
            inputs = tf.cond(tf.equal(is_train, True), lambda: tf.nn.dropout(inputs, dropout_p), lambda: inputs)
            inputs = tf.layers.dense(
                inputs, units, activation=tf.nn.relu, use_bias=True, 
                bias_initializer=tf.glorot_normal_initializer(),
                kernel_initializer=tf.glorot_normal_initializer(),
                bias_regularizer=lambda x: weight_decay*tf.nn.l2_loss(x),
                kernel_regularizer=lambda x: weight_decay*tf.nn.l2_loss(x))

        return inputs