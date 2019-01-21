# -*- coding: utf-8 -*-

import tensorflow as tf

def dense_net(
    inputs, out_size, first_scale_down_ratio, first_scale_down_filter, num_blocks, grow_rate, num_block_layers, 
    filter_height, filter_width, transition_rate, weight_decay=0.0, is_train=False, dropout_p=1.0, scope="dense_net"):
    """
    The module of DenseNet
    Params:
        inputs: the shape is (batch_size, height, width, in_channels)
    Returns:
        outs: the shape is (batch_size, out_size)
    Notes:
        the inputs' shape must not (None, None, None, in_channels), should be
        (None, seq_len, seq_len, in_channels)
    """
    with tf.variable_scope(scope):
        inputs = tf.cond(tf.equal(is_train, True), lambda: tf.nn.dropout(inputs, dropout_p), lambda: inputs)

        in_channels = inputs.get_shape().as_list()[-1]
        dim = inputs.get_shape().as_list()[-1] * first_scale_down_ratio
        filter = tf.get_variable(
            "first_filter", [first_scale_down_filter, first_scale_down_filter, in_channels, dim],
            initializer=tf.glorot_normal_initializer(), regularizer=lambda x: weight_decay*tf.nn.l2_loss(x))
        feature_map = tf.nn.conv2d(inputs, filter, [1,1,1,1], "SAME")
        feature_map = tf.nn.relu(feature_map)
        
        for i in range(num_blocks):
            # [batch_size, height, width, in_channels + num_block_layers*grow_rate]
            feature_map = dense_block(
                feature_map, grow_rate, num_block_layers, filter_height, filter_width, 
                padding="SAME", weight_decay=weight_decay, scope="block_{}".format(i))
            feature_map = tf.contrib.layers.layer_norm(feature_map)
            # [batch_size, int(height/2), int(width/2), in_channels*trasition_rate]
            feature_map = dense_transition_layer(
                feature_map, transition_rate, weight_decay=weight_decay, scope="transition_{}".format(i))
        
        feature_shape = feature_map.get_shape().as_list()
        feature_map = tf.reshape(feature_map, [-1, feature_shape[1]*feature_shape[2]*feature_shape[3]])
        feature_map = tf.contrib.layers.layer_norm(feature_map)
        
        outs = tf.layers.dense(
            feature_map, out_size, use_bias=False,
            kernel_initializer=tf.glorot_normal_initializer(),
            kernel_regularizer=lambda x: weight_decay*tf.nn.l2_loss(x))
        debug = outs

        return outs, debug

def dense_block(
    feature_map, grow_rate, num_layers, filter_height, filter_width, 
    padding="SAME", weight_decay=0.0, scope="dense_block"):
    with tf.variable_scope(scope):
        in_channels = feature_map.get_shape().as_list()[-1]
        feature_list = [feature_map]
        features = feature_map
        for i in range(num_layers):
            filter = tf.get_variable(
                "filter_{}".format(i), [filter_height, filter_width, in_channels, grow_rate],
                initializer=tf.glorot_normal_initializer(), 
                regularizer=lambda x: weight_decay*tf.nn.l2_loss(x))
            ft = tf.nn.conv2d(features, filter, [1,1,1,1], padding=padding)
            feature_list.append(ft)
            features = tf.concat(feature_list, axis = -1)
            # print(features)
            in_channels=features.get_shape().as_list()[-1]
        
        return features
            
def dense_transition_layer(
    feature_map, transition_rate, weight_decay=0.0, scope="transition_layer"):
    with tf.variable_scope(scope):
        in_channels = feature_map.get_shape().as_list()[-1]
        out_dim = int(feature_map.get_shape().as_list()[-1]*transition_rate)
        filter = tf.get_variable(
            "filter", [1,1,in_channels, out_dim], 
            initializer=tf.glorot_normal_initializer(),
            regularizer=lambda x: weight_decay*tf.nn.l2_loss(x))
        feature_map = tf.nn.conv2d(feature_map, filter, [1,1,1,1], padding="SAME")
        feature_map = tf.nn.max_pool(feature_map, [1,2,2,1],[1,2,2,1], "VALID")

        return feature_map

import numpy as np

if __name__ == "__main__":
    out_size = 4
    first_scale_down_ratio=1.0
    first_scale_down_filter=1
    num_blocks=3
    grow_rate=10
    num_block_layers=8
    filter_height=3
    filter_width=3
    transition_rate=0.5

    seq_len = 30
    dim=20
    inputs = np.random.randn(2, seq_len, seq_len, dim)
    p = tf.placeholder(tf.float32, shape=[None, seq_len, seq_len, dim])

    dn_out = dense_net(
        p, out_size, first_scale_down_ratio, first_scale_down_filter, num_blocks, 
        grow_rate, num_block_layers, filter_height, filter_width, transition_rate)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        dn = sess.run(
            dn_out,
            {p: inputs}
        )
        print(dn)
        print(dn.shape)