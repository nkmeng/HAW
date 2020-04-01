#!/usr/bin/env python
# encoding: utf-8


import tensorflow as tf


def add_cnn_layer(inputs, filter_sizes, num_filters, max_seq, keep_prob, scope_name="cnn_layer", drop_flag=True):
    word_dim = inputs.get_shape().as_list()[-1]
    inputs_expanded = tf.expand_dims(inputs, -1)
    with tf.variable_scope(scope_name):
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv_maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, word_dim, 1, num_filters]  # [filter_size,word_dim,1,num_filters]
                W = tf.Variable(tf.truncated_normal(shape=filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                conv = tf.nn.conv2d(inputs_expanded, W, strides=[1, 1, 1, 1], padding="VALID",
                                    name="conv")
                # conv shape:# VALID [batch_size , max_seq-filter_size+1 , 1, nums_filters]
                #  SAME [batch_size , max_seq  , word_dim, nums_filters]
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Max-pooling over the outputs
                pooled = tf.nn.max_pool(h, ksize=[1, max_seq - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                        padding='VALID', name="pool")  # [batch_size,1,1,nums_filters]
                pooled_outputs.append(pooled)  #

        # Combine all the pooled feature
        num_filters_total = num_filters * len(filter_sizes)

        h_pool = tf.concat(pooled_outputs, 3)  # shape:(batch_size,1,1,num_filters_total)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])  # [batch_size,num_filters_total]

        if drop_flag:
            h_pool_flat = tf.nn.dropout(h_pool_flat, keep_prob)

        return h_pool_flat


def mlp_layer(inputs, layer_sizes, keep_probs, random_base, l2_reg, active_func=None, scope_name='mlp'):
    weight_matrix = zip(layer_sizes, layer_sizes[1:])
    with tf.variable_scope(scope_name):
        cnt = 0
        for n_in, n_out in weight_matrix[:-1]:
            w = tf.get_variable(
                name='mlp_w_' + str(cnt),
                shape=[n_in, n_out],
                initializer=tf.random_uniform_initializer(-random_base, random_base),
                regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
            )
            b = tf.get_variable(
                name='mlp_b_' + str(cnt),
                shape=[n_out],
                initializer=tf.random_uniform_initializer(-random_base, random_base),
                regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
            )
            inputs = tf.nn.dropout(inputs, keep_prob=keep_probs[cnt])
            inputs = active_func(tf.nn.xw_plus_b(inputs, w, b))
    return inputs


def cnn_layer(inputs, filter_shape, strides, padding, random_base, l2_reg, active_func=None, scope_name="cnn"):
    with tf.variable_scope(scope_name):
        w = tf.get_variable(
            name='conv_w',
            shape=filter_shape,
            # initializer=tf.random_normal_initializer(mean=0., stddev=1.0),
            initializer=tf.random_uniform_initializer(-random_base, random_base),
            regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
        )
        b = tf.get_variable(
            name='conv_b',
            shape=[filter_shape[-1]],
            # initializer=tf.random_normal_initializer(mean=0., stddev=1.0),
            initializer=tf.random_uniform_initializer(-random_base, random_base),
            regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
        )
    conv = tf.nn.conv2d(inputs, w, strides, padding)
    h = tf.nn.bias_add(conv, b)
    if active_func is None:
        active_func = tf.nn.relu
    return active_func(h)


def dynamic_rnn(cell, inputs, n_hidden, length, max_len, scope_name, out_type='last'):
    outputs, state = tf.nn.dynamic_rnn(
        cell(n_hidden),
        inputs=inputs,
        sequence_length=length,
        dtype=tf.float32,
        scope=scope_name
    )  # outputs -> batch_size * max_len * n_hidden
    batch_size = tf.shape(outputs)[0]
    if out_type == 'last':
        index = tf.range(0, batch_size) * max_len + (length - 1)
        outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)  # batch_size * n_hidden
    elif out_type == 'all_avg':
        outputs = reduce_mean_with_len(outputs, length)
    return outputs


def gru_layer(cell, inputs, n_hidden, length, max_len, scope_name, out_type='last'):
    outputs, state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs, sequence_length=length, dtype=tf.float32,
                                       scope=scope_name)
    batch_size = tf.shape(outputs)[0]
    if out_type == 'last':  # 取最后
        index = tf.range(0, batch_size) * max_len + (length - 1)
        outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)  # batch_size * n_hidden
    elif out_type == 'all_avg':  # 取所有的均值
        outputs = reduce_mean_with_len(outputs, length)

    # 否则返回所有

    return outputs


def gru_document(cell, inputs, n_hidden, length, max_len, scope_name, out_type='last'):
    outputs, state = tf.nn.dynamic_rnn(cell=cell(n_hidden), inputs=inputs, sequence_length=length, dtype=tf.float32,
                                       scope=scope_name)
    batch_size = tf.shape(outputs)[0]
    if out_type == 'last':  # 取最后
        index = tf.range(0, batch_size) * max_len + (length - 1)
        outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)  # batch_size * n_hidden
    elif out_type == 'all_avg':  # 取所有的均值
        outputs = reduce_mean_with_len(outputs, length)

    # 否则返回所有

    return outputs


def bi_dynamic_rnn(cell, inputs, n_hidden, length, max_len, scope_name, out_type='last'):
    outputs, state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=cell(n_hidden),
        cell_bw=cell(n_hidden),
        inputs=inputs,
        sequence_length=length,
        dtype=tf.float32,
        scope=scope_name
    )
    if out_type == 'last':
        outputs_fw, outputs_bw = outputs
        outputs_bw = tf.reverse_sequence(outputs_bw, tf.cast(length, tf.int64), seq_axis=1)
        outputs = tf.concat([outputs_fw, outputs_bw], 2)
    else:
        outputs = tf.concat(outputs, 2)  # batch_size * max_len * 2n_hidden
    batch_size = tf.shape(outputs)[0]
    if out_type == 'last':
        index = tf.range(0, batch_size) * max_len + (length - 1)
        outputs = tf.gather(tf.reshape(outputs, [-1, 2 * n_hidden]), index)  # batch_size * 2n_hidden
    elif out_type == 'all_avg':
        outputs = reduce_mean_with_len(outputs, length)  # batch_size * 2n_hidden
    return outputs


def bi_dynamic_rnn_diff(cell, inputs_fw, inputs_bw, n_hidden, l_fw, l_bw, max_len, scope_name):
    with tf.name_scope('forward_lstm'):
        outputs_fw, state_fw = tf.nn.dynamic_rnn(
            cell(n_hidden),
            inputs=inputs_fw,
            sequence_length=l_fw,
            dtype=tf.float32,
            scope=scope_name
        )
        batch_size = tf.shape(outputs_fw)[0]
        index = tf.range(0, batch_size) * max_len + (l_fw - 1)
        output_fw = tf.gather(tf.reshape(outputs_fw, [-1, n_hidden]), index)  # batch_size * n_hidden

    with tf.name_scope('backward_lstm'):
        outputs_bw, state_bw = tf.nn.dynamic_rnn(
            cell(n_hidden),
            inputs=inputs_bw,
            sequence_length=l_bw,
            dtype=tf.float32,
            scope=scope_name
        )
        batch_size = tf.shape(outputs_bw)[0]
        index = tf.range(0, batch_size) * max_len + (l_bw - 1)
        output_bw = tf.gather(tf.reshape(outputs_bw, [-1, n_hidden]), index)  # batch_size * n_hidden

    outputs = tf.concat([output_fw, output_bw], 1)  # batch_size * 2n_hidden
    return outputs


def stack_bi_dynamic_rnn(cells_fw, cells_bw, inputs, n_hidden, n_layer, length, max_len, scope_name, out_type='last'):
    outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        cells_fw(n_hidden) * n_layer, cells_bw(n_hidden) * n_layer, inputs,
        sequence_length=length, dtype=tf.float32, scope=scope_name)
    if out_type == 'last':
        outputs_fw, outputs_bw = tf.split(2, 2, outputs)
        outputs_bw = tf.reverse_sequence(outputs_bw, tf.cast(length, tf.int64), seq_axis=1)
        outputs = tf.concat([outputs_fw, outputs_bw], 2)
    batch_size = tf.shape(outputs)[0]
    if out_type == 'last':
        index = tf.range(0, batch_size) * max_len + (length - 1)
        outputs = tf.gather(tf.reshape(outputs, [-1, 2 * n_hidden]), index)  # batch_size * 2n_hidden
    elif out_type == 'all_avg':
        outputs = reduce_mean_with_len(outputs, length)  # batch_size * 2n_hidden
    return outputs


def reduce_mean_with_len(inputs, length):
    """
    :param inputs: 3-D tensor
    :param length: the length of dim [1]
    :return: 2-D tensor
    """
    length = tf.cast(tf.reshape(length, [-1, 1]), tf.float32) + 1e-9
    inputs = tf.reduce_sum(inputs, 1, keep_dims=False) / length
    return inputs


def softmax_layer(inputs, n_hidden, random_base, keep_prob, l2_reg, n_class, scope_name='softmax'):
    with tf.variable_scope(scope_name):
        w = tf.get_variable(
            name='softmax_w',
            shape=[n_hidden, n_class],
            # initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_class))),
            initializer=tf.random_uniform_initializer(-random_base, random_base),
            regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
        )
        b = tf.get_variable(
            name='softmax_b',
            shape=[n_class],
            # initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_class))),
            initializer=tf.random_uniform_initializer(-random_base, random_base),
            regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
        )
    with tf.name_scope('softmax'):
        outputs = tf.nn.dropout(inputs, keep_prob=keep_prob)
        scores = tf.nn.xw_plus_b(outputs, w, b, 'scores')
    return scores
