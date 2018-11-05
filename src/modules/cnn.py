import tensorflow as tf

def conv1d(
    inputs, filter_width, out_channels, 
    padding="VALID", is_train=False, dropout_p=0.5,
    weight_decay=0.0, scope="conv1d"):
    """
    The convolution layer of 1-D
    Params:
        inputs: the shape is (batch_size, in_height, in_width, in_channels)
        filter_width: the number of filter' width
        out_channels: the number of filter's out_channels
    Returns:
        output: the shape is (batch_size, in_height, out_channels)
    Notes:
        in_width >= filter_width is required.
    """
    with tf.variable_scope(scope):
        in_channels = inputs.shape[-1]
        filter = tf.get_variable(
            "filter", shape=[1, filter_width, in_channels, out_channels],
            dtype=tf.float32, initializer=tf.glorot_normal_initializer(),
            regularizer=lambda x: weight_decay*tf.nn.l2_loss(x))
        bias = tf.get_variable(
            "bias", shape=[out_channels],
            dtype=tf.float32, initializer=tf.glorot_normal_initializer(),
            regularizer=lambda x:weight_decay*tf.nn.l2_loss(x))
        strides = [1, 1, 1, 1]
        inputs = tf.cond(tf.equal(is_train, True), lambda: tf.nn.dropout(inputs, dropout_p), lambda: inputs)
        inputs = tf.nn.conv2d(inputs, filter, strides, padding)+bias
        output = tf.reduce_max(tf.nn.relu(inputs), 2)
        return output

def multi_conv1d(
    inputs, filters_width, filters_out_channels,
    padding="VALID", is_train=False, dropout_p=0.5,
    weight_decay=0.0, scope="multi_conv"):
    """
    The multiply convolution layers of 1-D
    Params:
        inputs: the shape is (batch_size, in_height, in_width, in_channels)
        filters_width: the list of filters' width
        filters_out_channels: the list of filters' out_channels
    Returns:
        outputs: the shape is (batch_size, in_height, sum(filters_out_channels))
    """
    with tf.variable_scope(scope):
        assert len(filters_width) == len(filters_out_channels)
        outputs = []
        for filter_width, filter_out_channels in zip(filters_width, filters_out_channels):
            if filter_out_channels == 0:
                continue
            output = conv1d(
                inputs, filter_width, filter_out_channels,
                padding, is_train, dropout_p, weight_decay,
                scope="conv1d_{}".format(filter_width))
            outputs.append(output)
        outputs = tf.concat(outputs, axis=-1)
        
        return outputs

import numpy as np

if __name__ == "__main__":
    dim=5
    inputs = np.random.randn(2, 3, 5, dim)
    p = tf.placeholder(tf.float32, shape=[None, None, None, dim])
    
    filters_width = [5]
    filters_out_channels = [100]
    outputs = multi_conv1d(p, filters_width, filters_out_channels)
    # print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    # print(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

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