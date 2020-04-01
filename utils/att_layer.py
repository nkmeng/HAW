# -*- coding: utf-8 -*-
"""
  File Name : att_layer 
  Author : Hemeng
  date : 2020/3/30
  Description :
  Change Activity: 2020/3/30

"""

import numpy as np
import tensorflow as tf


def attention(inputs, attention_size, name_scope="attention"):
    if isinstance(inputs, tuple):  # judge the type of inputs
        inputs = tf.concat(inputs, axis=2)

    sequence_length = inputs.get_shape()[1].value
    hidden_size = inputs.get_shape()[2].value  #

    with tf.variable_scope(name_scope):
        # Attention mechanism
        W_omega = tf.get_variable("W_omega", initializer=tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.get_variable("b_omega", initializer=tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.get_variable("u_omega", initializer=tf.random_normal([attention_size], stddev=0.1))

        v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
        vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
        exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
        alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

        h_flag = tf.reshape(alphas, [-1, sequence_length, 1])

        output = tf.reduce_sum(inputs * h_flag, 1)

        return output, alphas


def softmax_with_len(inputs, length, max_len):
    inputs = tf.cast(inputs, tf.float32)
    inputs = tf.exp(inputs)
    length = tf.reshape(length, [-1])
    mask = tf.reshape(tf.cast(tf.sequence_mask(length, max_len), tf.float32), tf.shape(inputs))
    inputs *= mask
    _sum = tf.reduce_sum(inputs, reduction_indices=-1, keep_dims=True) + 1e-9
    return inputs / _sum


def bilinear_attention_layer(inputs, attend, length, n_hidden, l2_reg, random_base, scope_name='bilinear'):
    """
    :param inputs: batch * max_len * n_hidden
    :param attend: batch * n_hidden
    :return: batch * 1 * max_len
    """
    batch_size = tf.shape(inputs)[0]
    max_len = tf.shape(inputs)[1]
    with tf.variable_scope(scope_name):
        w = tf.get_variable(
            name='att_w',
            shape=[n_hidden, n_hidden],
            # initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_hidden))),
            initializer=tf.random_uniform_initializer(-random_base, random_base),
            regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
        )
    inputs = tf.reshape(inputs, [-1, n_hidden])
    tmp = tf.reshape(tf.matmul(inputs, w), [-1, max_len, n_hidden])
    attend = tf.expand_dims(attend, 2)
    tmp = tf.reshape(tf.matmul(tmp, attend), [batch_size, 1, max_len])
    # M = tf.expand_dims(tf.matmul(attend, w), 2)
    # tmp = tf.reshape(tf.batch_matmul(inputs, M), [batch_size, 1, max_len])
    return softmax_with_len(tmp, length, max_len)


def dot_produce_attention_layer(inputs, length, n_hidden, l2_reg, random_base, scope_name='dot'):
    """
    :param inputs: batch * max_len * n_hidden
    :param length: batch * 1
    :return: batch * 1 * max_len
    """
    batch_size = tf.shape(inputs)[0]
    max_len = tf.shape(inputs)[1]
    with tf.variable_scope(scope_name):
        u = tf.get_variable(
            name='att_u',
            shape=[n_hidden, 1],
            # initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + 1))),
            initializer=tf.random_uniform_initializer(-random_base, random_base),
            regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
        )
    inputs = tf.reshape(inputs, [-1, n_hidden])
    tmp = tf.reshape(tf.matmul(inputs, u), [batch_size, 1, max_len])
    alpha = softmax_with_len(tmp, length, max_len)
    return alpha


def mlp_attention_layer(inputs, length, n_hidden, l2_reg, random_base, scope_name='mlp'):
    """
    :param inputs: batch * max_len * n_hidden
    :param length: batch * 1
    :return: batch * 1 * max_len
    """
    batch_size = tf.shape(inputs)[0]
    max_len = tf.shape(inputs)[1]
    with tf.variable_scope(scope_name):
        w = tf.get_variable(
            name='att_w',
            shape=[n_hidden, n_hidden],
            # initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_hidden))),
            initializer=tf.random_uniform_initializer(-random_base, random_base),
            regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
        )
        b = tf.get_variable(
            name='att_b',
            shape=[n_hidden],
            # initializer=tf.random_normal_initializer(mean=0.0, stddev=np.sqrt(2. / (n_hidden + n_hidden))),
            initializer=tf.random_uniform_initializer(-0., 0.),
            regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
        )
        u = tf.get_variable(
            name='att_u',
            shape=[n_hidden, 1],
            # initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + 1))),
            initializer=tf.random_uniform_initializer(-random_base, random_base),
            regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
        )
    inputs = tf.reshape(inputs, [-1, n_hidden])
    tmp = tf.tanh(tf.matmul(inputs, w) + b)
    tmp = tf.reshape(tf.matmul(tmp, u), [batch_size, 1, max_len])
    alpha = softmax_with_len(tmp, length, max_len)
    return alpha
