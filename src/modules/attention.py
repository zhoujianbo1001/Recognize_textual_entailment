import tensorflow as tf

VERY_NEGATIVE_NUMBER = -1e30

def transformer_encoder(
    inputs, output_size, number_layers=1, 
    is_train=False, weight_decay=0.0, dropout_p=1.0,
    scope="transformer_encoder"):
    with tf.variable_scope("transformer_encoder"):
        # [batch_size, seq_len, dim]
        layer_inputs = inputs
        for i in range(number_layers):
            x = multi_head_attention(layer_inputs, layer_inputs)
            x = tf.add(x, inputs)
            x = tf.contrib.layers.layer_norm(x)
            layer_inputs = x
        
        last_dense = tf.layers.dense(
            layer_inputs, output_size, use_bias=False, 
            kernel_initializer=tf.glorot_normal_initializer(),
            kernel_regularizer=lambda x: weight_decay*tf.nn.l2_loss(x),
            name="feed_forward")
        
        return last_dense


def multi_head_attention(
    query, values, h=8, mask=None, func_att="dot-product", scaled=True,
    is_train=False, weight_decay=0.0, dropout_p=1.0,
    scope="multi_head_attention"):
    """
    The multi-head attention module.
    Params:
        query: the shape is (batch_size, query_len, dim).
        values: the shape is (batch_size, values_len, dim).
        h: the number of heads.
        mask: the shape is (batch_size, values_len), the number is 0 or 1
            if mask is not given, attention will not apply mask.
    Returns:
        outputs: the shape is (batch_size, query_len, dim)
    """
    with tf.variable_scope(scope):
        dim = query.shape[-1]
        
        #[batch_size, query_len, dim*h]
        query_dense = tf.layers.dense(
            query, dim*h,use_bias=False, 
            kernel_initializer=tf.glorot_normal_initializer(),
            kernel_regularizer=lambda x: weight_decay*tf.nn.l2_loss(x),
            name="query_dense")
        #[batch_size, values_len, dim*h]
        values_dense = tf.layers.dense(
            values, dim*h, use_bias=False, 
            kernel_initializer=tf.glorot_normal_initializer(),
            kernel_regularizer=lambda x: weight_decay*tf.nn.l2_loss(x),
            name="values_dense")
        # [h*batch_size, query_len, dim]
        query = tf.concat(tf.split(query_dense, h, axis=-1), axis=0)
        # [h*batch_size, value_len, dim]
        values = tf.concat(tf.split(values_dense, h, axis=-1), axis=0)
        if mask != None:
            mask = tf.reshape(
                tf.tile(tf.expand_dims(mask, 0), [h, 1, 1]), 
                [-1, mask.shape[-1]]
            )
        # [h*batch_size, query_len, dim]
        multi_head = attention(
            query, values, mask=mask, func_att=func_att, scaled=scaled,
            is_train=is_train, weight_decay=0.0, dropout_p=1.0)

        # [batch_size, query_len, dim*h]
        multi_head = tf.concat(
            tf.split(multi_head, h ,axis=0),
            axis=-1)
        
        def position_wise_ffn(inputs, scope="position_wise_ffn"):
            with tf.variable_scope(scope):
                filter_1 = tf.get_variable(
                    "filter_1", shape=[1, dim*h, dim], dtype=tf.float32, 
                    initializer=tf.glorot_normal_initializer(),
                    regularizer=lambda x: weight_decay*tf.nn.l2_loss(x))
                bias_1 = tf.get_variable(
                    "bias_1", shape=[dim], dtype=tf.float32, 
                    initializer=tf.glorot_normal_initializer(),
                    regularizer=lambda x: weight_decay*tf.nn.l2_loss(x))
                # [batch_size, len, dim]
                conv_1 = tf.nn.relu(
                    tf.nn.conv1d(inputs, filter_1, 1, "SAME")+bias_1)
                filter_2 = tf.get_variable(
                    "filter_2", shape=[1, dim, dim], dtype=tf.float32, 
                    initializer=tf.glorot_normal_initializer(),
                    regularizer=lambda x: weight_decay*tf.nn.l2_loss(x))
                bias_2 = tf.get_variable(
                    "bias_2", shape=[dim], dtype=tf.float32, 
                    initializer=tf.glorot_normal_initializer(),
                    regularizer=lambda x: weight_decay*tf.nn.l2_loss(x))
                conv_2 = tf.nn.conv1d(conv_1, filter_2, 1, "SAME")+bias_2
                return conv_2
        # [batch_size, query_len, dim]
        outputs = position_wise_ffn(multi_head)
        return outputs

def attention(
    query, values, mask=None, func_att="dot-product", scaled=True,
    is_train=False, weight_decay=0.0, dropout_p=1.0,
    scope="attention"):
    """
    The attention module
    Params:
        query: the shape is (batch_size, query_len, dim).
        values: the shape is (batch_size, values_len, dim).
        mask: the shape is (batch_size, values_len), the number is 0 or 1
            if mask is not given, attention will not apply mask.
        func_att: the function of attention
        scaled: the type is boolen, which reflects whether scaled the func_att or not.
    Returns:
        att_out: the shape is (batch_size, query_len, dim)
    """
    with tf.variable_scope(scope):
        func_out = func_attention(
            query, values, 
            func=func_att,
            scaled=scaled,
            is_train=is_train,
            weight_decay=weight_decay,
            dropout_p=dropout_p)
        att_logits = get_att_logits(func_out, mask)
        att_out = tf.matmul(att_logits, values)

        return att_out

def func_attention(
    query, values, func="dot-product", scaled=True,
    is_train=False, weight_decay=0.0, dropout_p=1.0,
    scope="attention_alignment"):
    """
    The attention function.
    Params:
        query: the shape is [batch_size, query_len, dim]
        values: the shape is [batch_size, values_len, dim]
    Returns:
        output: [batch_size, query_len, values_len]
    """
    with tf.variable_scope(scope):
        if func == "dot-product":
            output = tf.matmul(query, values, transpose_b=True)

        elif func == "multiplicative-att":
            values = tf.cond(tf.equal(is_train, True), lambda: tf.nn.dropout(values, dropout_p), lambda: values)
            dim = query.shape[-1]
            values = tf.layers.dense(
                values, dim, use_bias=False, 
                kernel_initializer=tf.glorot_normal_initializer(),
                kernel_regularizer=lambda x: weight_decay*tf.nn.l2_loss(x),
                name="values_dense")
            output = tf.matmul(query, values, transpose_b=True)

        elif func == "additive-att":
            query_shape = query.shape
            values_shape = values.shape
            # [batch_size, query_len, values_len, dim]
            query = tf.manip.tile(tf.expand_dims(query, 2), [1, 1, values_shape[1], 1])
            values = tf.manip.tile(tf.expand_dims(values, 1), [1, query_shape[1], 1, 1])

            inputs = tf.concat([query, values], -1)

            inputs = tf.cond(tf.equal(is_train, True), lambda: tf.nn.dropout(inputs, dropout_p), lambda: inputs)
            inputs = tf.layers.dense(
                inputs, values_shape[-1], activation=tf.nn.tanh, use_bias=False, 
                kernel_initializer=tf.glorot_normal_initializer(),
                kernel_regularizer=lambda x: weight_decay*tf.nn.l2_loss(x))

            inputs = tf.cond(tf.equal(is_train, True), lambda: tf.nn.dropout(inputs, dropout_p), lambda: inputs)
            inputs = tf.layers.dense(
                inputs, 1, use_bias=False, 
                kernel_initializer=tf.glorot_normal_initializer(),
                kernel_regularizer=lambda x: weight_decay*tf.nn.l2_loss(x))
            output = tf.squeeze(inputs, -1)

        else:
            raise NotImplementedError
        if scaled:
            output = tf.multiply(
                output, 
                tf.rsqrt(
                    tf.cast(query.shape[-1],tf.float32)
                )
            )
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
        inputs = apply_mask(inputs, mask)
    return tf.nn.softmax(inputs, axis=-1)

def apply_mask(val, mask, name="apply_mask"):
    """
    Give very negative number to unmasked elements in val.
    For example, [-3, -2, 10], [1, 1, 0] -> [-3, -2, -1e9].
    Typically, this effectively masks in exponential space (e.g. softmax)
    Params:
        val: values to be masked, whose shape is [batch_size, len1, len2]
        mask: masking boolean tensor, same shape as tensor
        name: name for output tensor
    Returns:
        Same shape as val, where some elements are very small (exponentially zero)
    """
    return tf.add(val, (1 - tf.cast(mask, 'float')) * VERY_NEGATIVE_NUMBER, name=name)

if __name__ == "__main__":
    #[2, 4, 3]
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
    # [2, 4]
    mask = tf.constant([
        [1,1,1,0],
        [1,1,0,0]
    ])
    # test attention
    att_out_1 = attention(
        query=inputs, values=inputs, mask=mask, 
        func_att="dot-product",
        weight_decay=0.1,
        is_train=True)
    att_out_2 = attention(
        query=inputs, values=inputs, mask=mask, 
        func_att="multiplicative-att",
        weight_decay=0.1,
        is_train=True)
    att_out_3 = attention(
        query=inputs, values=inputs, mask=mask, 
        func_att="additive-att",
        weight_decay=0.1,
        is_train=True)
        
    # test multi-head attention
    multi_head_out = multi_head_attention(inputs, inputs, mask=mask)

    # test transformer_encoder
    transformer_encoder_out = transformer_encoder(inputs, 4)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run([att_out_1]))
        print(sess.run([att_out_2]))
        print(sess.run([att_out_3]))
        print(sess.run([multi_head_out]))
        print(sess.run([transformer_encoder_out]))
