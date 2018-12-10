# -*- coding: UTF-8 -*-

import tensorflow as tf

def relation_network(
    input1, 
    input2,
    output_size,
    is_train=False,
    weight_decay=0.0,
    dropout_p=0.5,
    scope="relation_network"):
    """
    The module of relation network
    Params:
        input1: the tensor shape is (batch_size, len_1, dim)
        input2: the tensor shape is (batch_size, len_2, dim)
        output_size: the number is the size of output
    Returns:
        output: the tensor shape is (batch_size, output_size)
    """
    with tf.variable_scope(scope):
        input1_shape = tf.shape(input1)
        input2_shape = tf.shape(input2)
        dim = input1.shape[-1]
        batch_size = input1_shape[0]
        len_1 = input1_shape[1]
        len_2 = input2_shape[1]
        input1 = tf.expand_dims(input1, 2)
        input2 = tf.expand_dims(input2, 2)
        input1 = tf.tile(input1, [1, 1, len_2, 1])
        input2 = tf.tile(input2, [1, len_1, 1, 1])
        input1 = tf.reshape(input1, [batch_size, -1, dim])
        input2 = tf.reshape(input2, [batch_size, -1, dim])
        inputs = tf.concat([input1, input2], -1)

        inputs = tf.cond(tf.equal(is_train, True), lambda: tf.nn.dropout(inputs, dropout_p), lambda: inputs)
        inputs = tf.layers.dense(
            inputs, dim, use_bias=False, 
            kernel_initializer=tf.glorot_normal_initializer(),
            kernel_regularizer=lambda x: weight_decay*tf.nn.l2_loss(x),
            name="dense_g")
        inputs = tf.reduce_sum(inputs, axis=1)

        inputs = tf.cond(tf.equal(is_train, True), lambda: tf.nn.dropout(inputs, dropout_p), lambda: inputs)
        output = tf.layers.dense(
            inputs, output_size, use_bias=False, 
            kernel_initializer=tf.glorot_normal_initializer(),
            kernel_regularizer=lambda x: weight_decay*tf.nn.l2_loss(x),
            name="dense_f")

        return output

import numpy as np

if __name__ == "__main__":
    # [2, 3, 3]
    input1 = np.array(
        [
            [
                [1,1,1],
                [2,2,2],
                [3,3,3]
            ],
            [
                [1,1,1],
                [2,2,2],
                [3,3,3]
            ]
        ]
    ).astype(np.float32)
    # [2, 4, 3]
    input2 = np.array(
        [
            [
                [4,4,4],
                [5,5,5],
                [6,6,6],
                [7,7,7]
            ],
            [
                [4,4,4],
                [5,5,5],
                [6,6,6],
                [7,7,7]
            ]
        ]
    ).astype(np.float32)
    dim = 3
    lhs = tf.placeholder(tf.float32, [None, None, dim], "lhs")
    rhs = tf.placeholder(tf.float32, [None, None, dim], "rhs")
    output_size = 3

    output = relation_network(lhs, rhs, output_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        out = sess.run(
            output,
            feed_dict={
                lhs:input1,
                rhs:input2
            })
        print(out)


