# coding=UTF-8  

"""
# @File  : config.py
# @Author: HM
# @Date  : 2018/4/26
"""


class Config(object):
    model_name = "HAW"  #[cnn_rnn,rnn_cnn,han,hnn,gru,gru_bigru,HAW]
    sentence_level=True

    seed = 123  # random seed default=123


    # raw catalogs

    catalog = "E:/PythonProject/HAW/"
    save_path = catalog + "save/"
    if sentence_level:
        save_path_by_method=save_path+"sentence_level/"+model_name+"/"
    else:
        save_path_by_method=save_path+"document_level/"+model_name+"/"

    processed_dataset = save_path + "dataset/"
    raw_dataset =catalog+"raw_dataset/"

    stop_word_path=raw_dataset+"stopword.txt"
    # wordvec
    pre_trained_wordvec="E:/PythonProject/tools/GoogleNews-vectors-negative300.txt"
    wordVec_f=save_path+"wordvec"
    embedding_matrix=save_path+"embedding_matrix"
    word_list_f = save_path+"all_word.txt"

    # pivots

    src_domain = 'books'  # 'books,dvd,electronics,kitchen
    tar_domain = 'kitchen'  # 'books,dvd,electronics,kitchen,blog
    cross_domain_pairs=src_domain+"_"+tar_domain
    result_catalog = save_path_by_method + cross_domain_pairs+"/result/"
    model_catalog =save_path_by_method + cross_domain_pairs+  "/model/"
    alpha_catalog = save_path_by_method + cross_domain_pairs+  "/alpha/"
    pivots_catalog=save_path_by_method+cross_domain_pairs+"/pivots/"

    # basic setting
    test_num=200
    vocab_size = 44926
    word_dim = 300
    max_seq = 200
    bert_max_seq = 300
    max_doc_len = 10
    max_sen_len = 40
    n_classes = 2

    # training setting
    epoches = 60
    w_training=4
    batch_size = 64
    test_batch_size = 10

    hidden_size = 100
    n_layers = 1
    attention_dim = 200
    l2_reg_lambda = 0.05
    keep_prob = 0.5


FLAGS = Config()


