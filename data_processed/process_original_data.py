# coding=UTF-8  

"""
# @File  : process_original_data.py
# @Author: HM
# @Date  : 2018/4/3

process all unlabeld data
Rating 1.0 and 2.0 => neg,  4.0 and 5.0 => pos
extract context
save unlabeled_polarity

"""
from utils.config import FLAGS
import pickle
import numpy as np
import os


def process_data(f, unlabeled_flag):
    f_t_start, f_t_end = '<review_text>', '</review_text>'
    f_r_start, f_r_end = '<rating>', '</rating>'

    f_list = []
    findP_list, findN_list = [], []
    Pos_y, Neg_y = [], []

    stringP, stringN = '', ''

    for line in f.readlines():
        line = line.strip('\n')
        f_list.append(line)

    list_length = len(f_list)

    i = 0
    while i < list_length:
        if len(findP_list) > FLAGS.test_num and len(findN_list) > FLAGS.test_num and unlabeled_flag and len(
                findP_list) + len(findN_list) > 1000:
            break
        if f_list[i] == f_r_start:
            i += 1  # get rates
            if "4.0" in f_list[i] or "5.0" in f_list[i]:  # positive
                while f_list[i] != f_t_start:
                    i += 1
                i += 1
                stringP = f_list[i]
                while True:
                    i += 1
                    if f_list[i] == f_t_end:
                        findP_list.append(stringP.strip())
                        break
                    elif f_list[i] == '':
                        continue
                    stringP = stringP + f_list[i]
            elif "1.0" in f_list[i] or "2.0" in f_list[i]:  # negative
                while f_list[i] != f_t_start:
                    i += 1
                i += 1
                stringN = f_list[i]
                while True:
                    i += 1
                    if f_list[i] == f_t_end:
                        findN_list.append(stringN.strip())
                        break
                    elif f_list[i] == '':
                        continue
                    stringN = stringN + f_list[i]
        i += 1

    print("finish! write to file!\n")
    for i in range(len(findP_list)):
        Pos_y.append(1)
    for i in range(len(findN_list)):
        Neg_y.append(-1)

    return findP_list,  findN_list,Pos_y, Neg_y

def process_dataset():
    data_catalog = FLAGS.catalog + "raw_dataset/"
    domains = ['books', 'dvd', 'kitchen', 'electronics']
    flag_string = ['negative', 'positive', 'unlabeled']

    for domain_name in domains:
        save_file = FLAGS.save_path + "dataset/" + domain_name + "/"
        train_data, train_y = [], []
        test_data, test_y = [], []

        negative_f = open(data_catalog + domain_name + '/negative.review', encoding="latin-1", mode="r+")
        _, neg_list, _, neg_y = process_data(negative_f, False)
        positive_f = open(data_catalog + domain_name + '/positive.review', encoding="latin-1", mode="r+")
        pos_list, _, pos_y, _ = process_data(positive_f, False)
        train_data = pos_list + neg_list
        train_y = pos_y + neg_y

        unlabeled_f = open(data_catalog + domain_name + '/unlabeled.review', encoding="latin-1", mode="r+")
        pos_list, neg_list, pos_y, neg_y = process_data(unlabeled_f, True)
        test_data = pos_list[:500] + neg_list[:500]
        test_y = pos_y[:500] + neg_y[:500]

        if not os.path.exists(save_file):
            os.makedirs(save_file)

        train_fw = open(save_file + "Train", 'wb')
        pickle.dump(np.asarray(train_data), train_fw, -1)  # Pickle dictionary using protocol 0.
        pickle.dump(np.asarray(train_y), train_fw)

        test_fw = open(save_file + "Test", 'wb')
        pickle.dump(np.asarray(test_data), test_fw, -1)  # Pickle dictionary using protocol 0.
        pickle.dump(np.asarray(test_y), test_fw)


if __name__ == '__main__':
    process_dataset()
