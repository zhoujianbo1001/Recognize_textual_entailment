import tensorflow as tf

def unpadded_length(inputs, padding_idx=0):
    """
    Get true length of sequences (without padding), and mask for true-length in max-length.
    Params:
        inputs: the shape is (batch_size, max_seq_length), the type is int.
    Returns:
        length: the shape is (batch_size), the type is int
        mask: the shape is (batch_size, max_seq_length), the number is 0 or 1.
    """
    populated = tf.sign(tf.abs(inputs-padding_idx))
    length = tf.cast(tf.reduce_sum(populated, axis=-1), tf.int32)
    mask = tf.cast(populated, tf.float32)
    return length, mask

import numpy as np

if __name__ == "__main__":
    inputs = [
        [1,1,3,4,5,6,6,0,0,0,0,0],
        [3,6,7,8,9,9,3,1,0,0,0,0]
    ]
    padding_idx = 0
    p = tf.placeholder(tf.float32, [None, None])

    len, mask = unpadded_length(p,padding_idx)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        len_v, mask_v = sess.run(
            [len, mask],
            {
                p: inputs
            }
        )
        print(len_v)
        print(mask_v)