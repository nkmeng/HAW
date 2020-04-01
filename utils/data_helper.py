# -*- coding: utf-8 -*-
"""
  File Name : data_helper 
  Author : Hemeng
  date : 2020/3/30
  Description :
  Change Activity: 2020/3/30

"""
import os
import pickle
import nltk
import numpy as np
from data_processed.process_original_data import process_dataset
from utils.generate_wordvec import load_word2id, cleanSentences
from sklearn.model_selection import train_test_split
from utils.config import FLAGS


def load_pivots_list(pivots_f):
    pivots_list = []
    with open(pivots_f) as f:
        for line in f:
            line = line.replace("\n", "")
            pivots_list.append(line)
    f.close()
    return pivots_list





def transform_data(words, p_pivots_list, n_pivots_list):
    pnum = 0
    nnum = 0
    # print("start transform...")
    for index, word in enumerate(words):
        if word in p_pivots_list:
            words[index] = "UNK"
            pnum += 1
    for index, word in enumerate(words):
        if word in n_pivots_list:
            words[index] = "UNK"
            nnum += 1
    return words, pnum, nnum


def trans_data(config, txt, word2id, transform=False):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    x, y, sen_len, doc_len, tag_x, p_pivots, n_pivots = [], [], [], [], [], [], []

    # cixing = ["JJ", "JJR", "JJS", "RB", "RBR", "RBS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    cixing = ["JJ", "JJR", "JJS", "RB", "RBR", "RBS"]

    if transform:
        p_pivots_list = load_pivots_list(config.pivots_catalog + "pos.txt")
        n_pivots_list = load_pivots_list(config.pivots_catalog + "neg.txt")

    for line_i, l1 in enumerate(txt):
        flag = False
        p_pivots_num = 0
        n_pivots_num = 0

        t_sen_len = [0] * config.max_doc_len  # how many words are stored in clauses
        t_x = np.zeros((config.max_doc_len, config.max_sen_len))  # value:id,
        tag = np.zeros((config.max_doc_len, config.max_sen_len))  # 0:word tag that not needed  1:needed
        sentences = tokenizer.tokenize(l1)
        i = 0
        for sentence in sentences:  # every clause in document
            j = 0  # how many word in sentence
            sentence = cleanSentences(sentence)
            sentence = sentence.strip()
            sentence = sentence.replace("\n", "")
            if sentence == "":
                pass
            else:
                words = sentence.split()
                if transform:
                    words_p, p_pivots_num, n_pivots_num = transform_data(words, p_pivots_list, n_pivots_list)
                pos_tag = nltk.pos_tag(words)
                for word in pos_tag:
                    if j < config.max_sen_len:
                        if word[0] in word2id:
                            t_x[i, j] = word2id[word[0]]
                            if word[1] in cixing:
                                tag[i, j] = 1
                            else:
                                tag[i, j] = 0
                            j += 1
                        elif word[0] == "UNK":
                            t_x[i, j] = len(word2id)
                            tag[i, j] = 0
                            j += 1
                        else:
                            print("word is ：%s not in word2id" % word[0])
                    else:  # Truncation beyond maximum words
                        break
                if j > 2:  # For clause words greater than 2
                    t_sen_len[i] = j  # t_sen_len: j words in clause i
                    i += 1
                    flag = True

                if i >= config.max_doc_len:  # Truncation beyond maximum clause
                    break

        if not flag:
            print("Sentence with less than 3 words:", sentence)

        doc_len.append(i)  # doc_len: sentence i in each document
        sen_len.append(t_sen_len)  # sen_len: how many word in each sentence
        x.append(t_x)
        tag_x.append(tag)

        if p_pivots_num > 0:
            p_pivots.append([1., 0.])
        else:
            p_pivots.append([0., 1.])
        if n_pivots_num > 0:
            n_pivots.append([1., 0.])
        else:
            n_pivots.append([0., 1.])

    print("the length of x:%d" % len(x))
    return np.asarray(x), np.asarray(sen_len), np.asarray(doc_len), np.asarray(tag_x), np.asarray(p_pivots), np.asarray(
        n_pivots)


def trans_label(labels):
    y = []
    for line_i, label in enumerate(labels):
        if label == 1:
            y.append([1., 0.])
        else:
            y.append([0., 1.])
    return np.asarray(y)


def load_train_data(config, domain, transform, train_flag):
    if train_flag:
        f_name = config.processed_dataset + domain + "/Train"
    else:
        f_name = config.processed_dataset + domain + "/Test"

    if os.path.exists(f_name + "_processed"):
        read_f = open(f_name + "_processed", "rb")
        train_data = pickle.load(read_f)
        test_data = pickle.load(read_f)
    else:
        if not os.path.exists(f_name):
            process_dataset()
        read_f = open(f_name, 'rb')
        save_f = open(f_name + "_processed", 'wb')

        text = pickle.load(read_f)
        labels = pickle.load(read_f)
        word2id = load_word2id()
        x, sen_len, doc_len, tag_x, p_pivots, n_pivots = trans_data(config, text, word2id, transform)
        labels = trans_label(labels)
        train_x, test_x, train_sen_len, test_sen_len, train_doc_len, test_doc_len, train_y, test_y, train_tag_x, test_tag_x, train_p_pivots, test_p_pivots, train_n_pivots, test_n_pivots = train_test_split(
            x, sen_len, doc_len, labels, tag_x, p_pivots, n_pivots, test_size=0.2, random_state=1)

        if transform:
            train_data = [train_x, train_sen_len, train_doc_len, train_y, train_tag_x, train_p_pivots, train_n_pivots]
            test_data = [test_x, test_sen_len, test_doc_len, test_y, test_tag_x, test_p_pivots, test_n_pivots]
        else:
            train_data = [train_x, train_sen_len, train_doc_len, train_y, train_tag_x]
            test_data = [test_x, test_sen_len, test_doc_len, test_y, test_tag_x]

        pickle.dump(train_data, save_f)
        pickle.dump(test_data, save_f)

    return train_data, test_data


def batches_iter(data, batch_size, shuffle=True, transform=False):
    """
    iterate the data
    """

    data_len = len(data[0])
    batch_num = int(data_len / batch_size)  # 一个batch中多少data
    x = np.array(data[0])
    sen_len = np.array(data[1])
    doc_len = np.array(data[2])
    label = np.array(data[3])
    tag_x = np.array(data[4])
    if transform:
        p_pivots = np.array(data[5])
        n_pivots = np.array(data[6])

    if shuffle:
        shuffle_idx = np.random.permutation(np.arange(data_len))  # shuffle the data_len array
        x = x[shuffle_idx]
        sen_len = sen_len[shuffle_idx]
        doc_len = doc_len[shuffle_idx]
        label = label[shuffle_idx]
        tag_x = tag_x[shuffle_idx]
        if transform:
            p_pivots = p_pivots(shuffle_idx)
            n_pivots = n_pivots(shuffle_idx)

    for batch in range(batch_num):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, data_len)
        if transform:
            yield x[start_idx: end_idx], sen_len[start_idx: end_idx], doc_len[start_idx:end_idx], \
                  label[start_idx:end_idx], tag_x[start_idx:end_idx], p_pivots[start_idx:end_idx], \
                  n_pivots[start_idx:end_idx]
        else:
            yield x[start_idx: end_idx], sen_len[start_idx: end_idx], doc_len[start_idx:end_idx], \
                  label[start_idx:end_idx], tag_x[start_idx:end_idx]
