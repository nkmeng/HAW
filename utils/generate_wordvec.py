# -*- coding: utf-8 -*-
"""
  File Name : generate_wordvec 
  Author : Hemeng
  date : 2020/3/30
  Description :
  Change Activity: 2020/3/30

"""

from utils.config import *
import os
import re
import numpy as np
import pickle



def cleanSentences(string):
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, " ", string.lower())


def load_word2id():
    if not os.path.exists(FLAGS.word_list_f):
        generate_allword_list()
    word_id_dict = {}
    with open(FLAGS.word_list_f, encoding="utf-8", mode="r") as f:
        with open(FLAGS.word_list_f, encoding="utf-8", mode="r") as f:
            for line_i, line in enumerate(f):
                line = line.replace("\n", "").strip()
                word_id_dict[line] = line_i
        f.close()
        print("Load wordlist_dic finished! Length is %d" % (len(word_id_dict)))
    return word_id_dict


def load_id2word():
    id2word_dict = {}
    with open(FLAGS.word_list_f, encoding="utf-8", mode="r") as f:
        with open(FLAGS.word_list_f, encoding="utf-8", mode="r") as f:
            for line_i, line in enumerate(f):
                line = line.replace("\n", "").strip()
                id2word_dict[line_i] = line
        f.close()
        print("Load word2id_dic finished! Length is %d" % (len(id2word_dict)))
    return id2word_dict


def generate_allword_list():
    domain = ["books", "dvd", "electronics", "kitchen"]
    train_strings = ["Train", "Test"]
    words = []
    max_seq = 0
    for d in domain:
        for flag_string in train_strings:
            f1 = open(FLAGS.processed_dataset + d + "/" + flag_string, 'rb')
            data = pickle.load(f1)
            # labels=pickle.load(f1)

            for line_i, line in enumerate(data):
                text = cleanSentences(line)
                texts = text.split()
                if len(texts) > max_seq:
                    max_seq = len(texts)
                for w in texts:
                    if w not in words:
                        words.append(w)
    print("The length of word_list is %d" % (len(words)))
    print("max seq:", max_seq)
    with open(FLAGS.word_list_f, encoding="UTF-8", mode="w+") as write_f:
        for word in words:
            write_f.write(word + "\n")
    write_f.close()
    print("Finished write the all word list to file.")


def lookup_word_vector():
    """
    lookup word in google wordvec
    :return:
    """
    wordlist = []
    with open(FLAGS.word_list_f, encoding="utf-8", mode="r") as f:
        for word in f:
            word = word.replace("\n", "")
            wordlist.append(word)
    f.close()
    print("load all_word_list finished! Length is %d" % (len(wordlist)))

    wordVec = []
    with open(FLAGS.pre_trained_wordvec, encoding="utf-8", mode="r") as read_f:

        for line_i, line in enumerate(read_f):
            if line_i == 0 and 'GoogleNews' in FLAGS.pre_trained_wordvec:
                continue
            line = line.replace("\n", "").strip()
            lines = line.split()
            goo_word = lines[0]
            if goo_word in wordlist:
                wordVec.append(line)
            if line_i % 100000 == 0:
                print("Has finished %d line" % line_i)
    read_f.close()
    print("Load wordvec finished! The length of wordVec is %d" % (len(wordVec)))
    np.save(FLAGS.wordVec_f, wordVec)


def generate_matrix(word_dim):
    wordlist_dic = load_word2id()

    # load google pre-train wordvec
    if not os.path.exists(FLAGS.wordVec_f + ".npy"):
        lookup_word_vector()
    wordVec = np.load(FLAGS.wordVec_f + ".npy")
    print("The shape of wordVec is " + str(wordVec.shape))

    embedding_dic = {}  # word-vector
    for line in wordVec:
        line = line.replace("\n", "").strip()
        lines = line.split()
        word = lines[0]
        vector = np.asarray(lines[1:], dtype='float32')
        embedding_dic[word] = vector
    print("Found %d word vector." % (len(embedding_dic)))

    print("Preparing embedding matrix.")
    embedding_matrix = np.zeros([len(wordlist_dic) + 1, word_dim])

    # find all vector in wordVec, save as embedding matrix by index-vector .
    for (word, i) in wordlist_dic.items():
        # if i > len(wordVec):
        #     continue
        embedding_vector = embedding_dic.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.random.normal(0, 0.1, word_dim)

    np.save(FLAGS.embedding_matrix, embedding_matrix)
    print(embedding_matrix.shape)
    print(embedding_matrix)
    print("finished save embedding_matrix.")


if __name__ == "__main__":
    generate_allword_list()
