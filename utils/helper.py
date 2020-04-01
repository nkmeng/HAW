# -*- coding: utf-8 -*-
"""
  File Name : helper 
  Author : Hemeng
  date : 2020/1/8
  Description :
  Change Activity: 2020/1/8

"""
import os
from utils.config import *
import tensorflow as tf
import numpy as np
from utils.generate_wordvec import load_id2word, load_word2id


def load_stopword():
    stoplist = []
    with open(FLAGS.stop_word_path, encoding="UTF-8", mode="r+") as f:
        for line in f:
            line = line.replace("\n", "")
            stoplist.append(line)
    f.close()
    return stoplist


def split_src_tar(outputs, is_train, batch_size):
    if len(outputs.shape.as_list()) == 2:
        src_feature = lambda: tf.slice(outputs, [0, 0], [batch_size // 2, -1])
        tar_feature = lambda: tf.slice(outputs, [batch_size // 2, 0], [batch_size // 2, -1])
    else:  # len(outputs.shape.as_list()) == 3:
        src_feature = lambda: tf.slice(outputs, [0, 0, 0], [batch_size // 2, -1, -1])
        tar_feature = lambda: tf.slice(outputs, [batch_size // 2, 0, 0], [batch_size // 2, -1, -1])
    src_feats = tf.cond(is_train, src_feature, lambda: outputs)
    tar_feats = tf.cond(is_train, tar_feature, lambda: outputs)
    return src_feats, tar_feats


def lookup_max_sen(data):
    """
    Find the maximum attention sentence index
    """

    length = len(data)
    max_index = 0
    for i in range(1, length):
        if data[i] > data[max_index]:
            max_index = i
    return max_index


def lookup_max_word(data, tag_x):
    """
    Sort by attention value from big to small
    Pick the two that meet part of speech, otherwise output -1
    """
    max_word_index = []
    pick_word_num = 0
    flag = False

    index_list = np.argsort(data)[:]
    for i in range(len(data)):
        if tag_x[index_list[-i - 1]] == 1:  # 是需要词性的attention最大的
            pick_word_num += 1
            flag = True
            max_word_index.append(index_list[-i - 1])
        if pick_word_num == 2:
            break
    if flag:
        return max_word_index
    else:
        print("Can't find the higher Attention needed, return -1")
        return -1


def get_pivots(s_alpha, w_alpha, X, Y, Tag_x, config):
    p_pivots, n_pivots = [], []
    p_pivots_id, n_pivots_id = [], []
    p_pivots_num, n_pivots_num = dict(), dict()

    id2word = load_id2word()
    stoplist = load_stopword()
    max_sen_alpha = []  # save the sen_index of max_sen_alpha
    for docs in s_alpha:
        max_sen_alpha.append(lookup_max_sen(docs))

    w_alpha = np.asarray(w_alpha)
    max_word_alpha = []
    for d_i, d in enumerate(w_alpha):  # 对每个document级
        cur_max_sen = max_sen_alpha[d_i]
        max_word_alpha.append(lookup_max_word(d[cur_max_sen], Tag_x[d_i][cur_max_sen]))

    for d_i, d in enumerate(X):  # 对X的doucment级
        if max_word_alpha[d_i] == -1:
            continue
        for w_i in max_word_alpha[d_i]:
            temp_id = d[max_sen_alpha[d_i]][w_i]
            temp_w = id2word[temp_id]
            if len(temp_w) > 2 and temp_w not in stoplist:
                if Y[d_i][0] == 1.:
                    if temp_w not in p_pivots:
                        p_pivots_id.append(temp_id)
                        p_pivots.append(temp_w)
                        p_pivots_num[temp_w] = 1
                    else:
                        p_pivots_num[temp_w] += 1
                else:
                    if temp_w not in n_pivots:
                        n_pivots_id.append(temp_id)
                        n_pivots.append(temp_w)
                        n_pivots_num[temp_w] = 1
                    else:
                        n_pivots_num[temp_w] += 1


    return p_pivots,n_pivots,p_pivots_num,n_pivots_num

