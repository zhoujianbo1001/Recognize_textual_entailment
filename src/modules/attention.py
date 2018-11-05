import tensorflow as tf

VERY_NEGATIVE_NUMBER = -1e30

def self_attention(
    inputs, mask=None, att_func="dot-product", 
    is_train=False, weight_decay=0.0, dropout_p=0.5,
    scope="self_attention"):
    """
    The self attention module
    Params:
        inputs: the shape is (batch_size, seq_len, dim).
        mask: the shape is (batch_size, seq_len), the number is 0 or 1
            if mask is not given, self attention will not express mask.
    Returns:
        outputs: the shape is (batch_size, seq_len, dim)
    """
    with tf.variable_scope(scope):
        ali_out = attention_alignment(
            inputs, inputs, 
            func=att_func,
            is_train=is_train,
            weight_decay=weight_decay,
            dropout_p=dropout_p)
        att_logits = get_att_logits(ali_out, mask)
        att_out = tf.matmul(att_logits, inputs)

        output = fuse_gate(
            inputs, att_out, 
            weight_decay=weight_decay,
            is_train=is_train,
            dropout_p=dropout_p)

        return output

def attention_alignment(
    input1, input2, func="dot-product", 
    is_train=False, weight_decay=0.0, dropout_p=0.5,
    scope="attention_alignment"):
    """
    The attention's alignment model
    Params:
        input1: the main sequence, whose shape is [batch_size, len1, dim]
        input2: the context sequence, whose shape is [batch_size, len2, dim]
    Returns:
        output: [batch_size, len1, len2]
    """
    with tf.variable_scope(scope):
        if func == "dot-product":
            output = tf.matmul(input1, input2, transpose_b=True)

        elif func == "multiplicative-att":
            input2 = tf.cond(tf.equal(is_train, True), lambda: tf.nn.dropout(input2, dropout_p), lambda: input2)
            dim = input1.shape[-1]
            input2 = tf.layers.dense(
                input2, dim, use_bias=False, 
                kernel_initializer=tf.glorot_normal_initializer(),
                kernel_regularizer=lambda x: weight_decay*tf.nn.l2_loss(x),
                name="input2_dense")
            output = tf.matmul(input1, input2, transpose_b=True)

        elif func == "additive-att":
            input1_shape = input1.shape
            input2_shape = input2.shape
            input1 = tf.manip.tile(tf.expand_dims(input1, 2), [1, 1, input2_shape[1], 1])
            input2 = tf.manip.tile(tf.expand_dims(input2, 1), [1, input1_shape[1], 1, 1])
            inputs = tf.concat([input1, input2], -1)
            inputs = tf.cond(tf.equal(is_train, True), lambda: tf.nn.dropout(inputs, dropout_p), lambda: inputs)
            inputs = tf.layers.dense(
                inputs, input1_shape[-1], activation=tf.nn.tanh, use_bias=False, 
                kernel_initializer=tf.glorot_normal_initializer(),
                kernel_regularizer=lambda x: weight_decay*tf.nn.l2_loss(x))
            inputs = tf.cond(tf.equal(is_train, True), lambda: tf.nn.dropout(inputs, dropout_p), lambda: inputs)
            inputs = tf.layers.dense(
                inputs, 1, use_bias=False, 
                kernel_initializer=tf.glorot_normal_initializer(),
                kernel_regularizer=lambda x: weight_decay*tf.nn.l2_loss(x))
            output = tf.squeeze(inputs, -1)

        else:
            raise Exception

        return output

def get_att_logits(inputs, mask=None, scope="get_att_logits"):
    """
    Give the attention logits matrix
    Params:
        inputs: the shape is [batch_size, len1, len2]
        mask: the shape is [batch_size, len2]
    """
    if mask is not None:
        mask = tf.expand_dims(mask, axis=1)
        inputs = exp_mask(inputs, mask)
    return tf.nn.softmax(inputs, axis=-1)

def exp_mask(val, mask, name="exp_mask"):
    """
    Give very negative number to unmasked elements in val.
    For example, [-3, -2, 10], [True, True, False] -> [-3, -2, -1e9].
    Typically, this effectively masks in exponential space (e.g. softmax)
    Params:
        val: values to be masked, whose shape is [batch_size, len1, len2]
        mask: masking boolean tensor, same shape as tensor
        name: name for output tensor
    Returns:
        Same shape as val, where some elements are very small (exponentially zero)
    """
    return tf.add(val, (1 - tf.cast(mask, 'float')) * VERY_NEGATIVE_NUMBER, name=name)

def fuse_gate(
    input1, input2,
    weight_decay=0.1,
    is_train=False,
    dropout_p=0.5,
    scope="fuse_gate"):
    with tf.variable_scope(scope):
        dim = input1.shape[-1]
        inputs = tf.concat([input1, input2], -1)
        inputs = tf.cond(tf.equal(is_train, True), lambda: tf.nn.dropout(inputs, dropout_p), lambda: inputs)
        z = tf.layers.dense(
            inputs, dim, activation=tf.nn.tanh,
            kernel_initializer=tf.glorot_normal_initializer(),
            kernel_regularizer=lambda x: tf.nn.l2_loss(x), 
            name="dense_z")
        r = tf.layers.dense(
            inputs, dim, activation=tf.nn.sigmoid,
            kernel_initializer=tf.glorot_normal_initializer(),
            kernel_regularizer=lambda x: tf.nn.l2_loss(x), 
            name="dense_r")
        f = tf.layers.dense(
            inputs, dim, activation=tf.nn.sigmoid,
            kernel_initializer=tf.glorot_normal_initializer(),
            kernel_regularizer=lambda x: tf.nn.l2_loss(x), 
            name="dense_f")
        output = tf.multiply(r, input1) + tf.multiply(f,z)

        return output


if __name__ == "__main__":
    inputs = tf.constant(
        [
            [
                [1,2,3],
                [3,2,1],
                [1,2,3],
                [3,2,1]
            ],
            [
                [4,5,6],
                [6,5,4],
                [4,5,6],
                [6,5,4]
            ]
        ],
        dtype=tf.float32
    )

    mask = [
        [1,1,1,0],
        [1,1,0,0]
    ]

    att_out = self_attention(
        inputs, mask, 
        att_func="additive-att",
        weight_decay=0.1,
        is_train=True)
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run([att_out]))
    