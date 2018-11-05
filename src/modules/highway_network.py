import tensorflow as tf

def highway_layer(
    inputs, output_size=None,
    is_train=False, dropout_p=0.5, weight_decay=0.0,
    scope="highway_layer"):
    """
    The highway layer module
    Params:
        inputs: the shape is (batch_size, seq_len, dim)
    Returns:
        outputs: the shape is (batch_size, seq_len, output_size)
    """
    with tf.variable_scope(scope):
        inputs = tf.cond(tf.equal(is_train, True), lambda: tf.nn.dropout(inputs, dropout_p), lambda: inputs)
        output_size = inputs.shape[-1] if output_size == None else output_size

        # (batch_size, seq_len, output_size)
        gate = tf.layers.dense(
            inputs, output_size, activation=tf.nn.sigmoid,
            kernel_initializer=tf.glorot_normal_initializer(),
            kernel_regularizer=lambda x: weight_decay*tf.nn.l2_loss(x),
            name="gate_dense")
        trans = tf.layers.dense(
            inputs, output_size, activation=tf.nn.relu,
            kernel_initializer=tf.glorot_normal_initializer(),
            kernel_regularizer=lambda x: weight_decay*tf.nn.l2_loss(x),
            name="trans_dense")
        inputs = tf.layers.dense(
            inputs, output_size,
            kernel_initializer=tf.glorot_normal_initializer(),
            kernel_regularizer=lambda x: weight_decay*tf.nn.l2_loss(x),
            name="inputs_dense")
        outputs = gate * trans + (1 - gate) * inputs

        return outputs

def highway_network(
    inputs, num_layers, output_size=None,
    is_train=False, dropout_p=0.5, weight_decay=0.0,
    scope="highway_network"):
    with tf.variable_scope(scope):
        for i in range(num_layers):
            inputs = highway_layer(
                inputs, output_size, is_train, dropout_p,
                weight_decay, scope="layer_{}".format(i))
    return inputs

import numpy as np

if __name__ == "__main__":
    dim = 5
    inputs = np.random.randn(2, 3, dim)

    p = tf.placeholder(tf.float32, [None, None, dim])
    num_layers = 2
    outputs = highway_network(p, num_layers)
    # print(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    # print(tf.get_collection(tf.GraphKeys.LOSSES))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        outs = sess.run(
            outputs, 
            feed_dict={
                p: inputs
            }  
        )
        print(outs.shape)
        print(outs)
