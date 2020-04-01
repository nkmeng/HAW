# coding=UTF-8  

"""
# @File  : model.py
# @Author: HM
# @Date  : 2018/4/25
HATN的模型
"""
import os

import json
from utils.generate_wordvec import *

from utils.helper import *
from utils.nn_layer import *
from utils.att_layer import *


from utils.data_helper import *





class Model(object):
    def __init__(self, wordVectors, filter_list=(2, 3, 4), filter_num=100):
        self.config = FLAGS
        self.embeddings = wordVectors
        self.weight_decay = 0.8  # 写成变量
        self.filter_list = filter_list
        self.filter_num = filter_num

        self.x = tf.placeholder(tf.int32, [None, self.config.max_doc_len, self.config.max_sen_len], name="input_data")
        self.tag_x = tf.placeholder(tf.int32, [None, self.config.max_doc_len, self.config.max_sen_len],
                                    name="input_pos_tag")
        self.doc_y = tf.placeholder(tf.float32, [None, self.config.n_classes], name="input_label")
        self.domain = tf.placeholder(tf.float32, [None, 2], name="domain_class")
        self.sen_len = tf.placeholder(tf.int32, [None, self.config.max_doc_len],
                                      name="sen_len")
        self.doc_len = tf.placeholder(tf.int32, [None], name="doc_len")
        self.learning_rate = tf.placeholder(tf.float32, [], name="learnning_rate")
        self.l = tf.placeholder(tf.float32, [], name="l")  #
        self.lr_wd_D = tf.placeholder(tf.float32, [], name="learnning_rate_wd_D")
        self.is_train = tf.placeholder(tf.bool, [], name="train_flag")

        with tf.name_scope("embedding"):
            W = tf.get_variable("embedding", shape=[self.config.vocab_size, self.config.word_dim],
                                initializer=tf.constant_initializer(self.embeddings),
                                trainable=True)
            inputs = tf.nn.embedding_lookup(W, self.x)  # (batch_size,max_doc_len,max_sen_len,word_dim)

        with tf.variable_scope('feature_extractor'):
            if self.config.model_name == 'cnn_rnn':
                outputs_doc = self.cnn_rnn(inputs)
            elif self.config.model_name == 'rnn_cnn':
                outputs_doc = self.rnn_cnn(inputs)
            elif self.config.model_name == 'han':
                outputs_doc = self.han(inputs)
            elif self.config.model_name == 'hnn':
                outputs_doc = self.hnn(inputs)
            elif self.config.model_name == "gru":
                outputs_doc = self.gru(inputs)
            elif self.config.model_name == "gru_bigru":
                outputs_doc = self.bigru_gru(inputs)
            elif self.config.model_name == "HAW":
                outputs_doc = self.bigru_bigru(inputs)
            else:
                print("no model!")

        with tf.variable_scope('sentiment_pred'):
            s_feats, t_feats = split_src_tar(outputs_doc, self.is_train, batch_size=self.config.batch_size)
            self.s_labels, _ = split_src_tar(self.doc_y, self.is_train, batch_size=self.config.batch_size)
            s_logits = tf.layers.dense(s_feats, self.config.n_classes)

        with tf.variable_scope('domain_discrepancy'):
            alpha = tf.random_uniform(shape=[self.config.batch_size // 2, 1], minval=0., maxval=1.)
            differences = s_feats - t_feats
            interpolates = t_feats + (alpha * differences)
            h1_whole = tf.concat([outputs_doc, interpolates], 0)

            critic_h1 = tf.layers.dense(h1_whole, 100, activation=tf.nn.relu)
            critic_out = tf.layers.dense(critic_h1, 1, activation=tf.identity)
            critic_s = tf.cond(self.is_train, lambda: tf.slice(critic_out, [0, 0], [self.config.batch_size // 2, -1]),
                               lambda: critic_out)
            critic_t = tf.cond(self.is_train, lambda: tf.slice(critic_out, [self.config.batch_size // 2, 0],
                                                               [self.config.batch_size // 2, -1]), lambda: critic_out)

        with tf.name_scope("accuracy"):
            self.s_pred = tf.argmax(s_logits, 1)
            s_correct_pred = tf.equal(self.s_pred, tf.argmax(self.s_labels, 1))
            self.s_correct_num = tf.reduce_sum(tf.cast(s_correct_pred, tf.float32))
            self.s_acc = tf.reduce_mean(tf.cast(s_correct_pred, tf.float32), name="sen_acc")

        with tf.name_scope("loss"):
            self.senti_loss = tf.nn.softmax_cross_entropy_with_logits(logits=s_logits, labels=self.s_labels)
            self.senti_cost = tf.reduce_mean(self.senti_loss)
            wd_loss = (tf.reduce_mean(critic_s) - tf.reduce_mean(critic_t))

        gradients = tf.gradients(critic_out, [h1_whole])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

        theta_C = [v for v in tf.global_variables() if 'sentiment_pred' in v.name]
        theta_D = [v for v in tf.global_variables() if 'domain_discrepancy' in v.name]
        theta_G = [v for v in tf.global_variables() if 'feature_extractor' in v.name]

        self.wd_d_op = tf.train.AdamOptimizer(self.lr_wd_D).minimize(-wd_loss + 10 * gradient_penalty,
                                                                     var_list=theta_D)

        all_variables = tf.trainable_variables()
        all_variables_p = [var for var in all_variables if
                           'sentiment_pred' in var.name or 'domain_discrepancy' in var.name or 'feature_extractor' in var.name]

        l2_loss_p = 1e-4 * tf.add_n([tf.nn.l2_loss(v) for v in all_variables_p if 'bias' not in v.name])

        self.total_loss_p = self.senti_cost + l2_loss_p + 1 * wd_loss
        self.P_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss_p,
                                                                              var_list=theta_G + theta_C)

    def cnn_rnn(self, inputs):
        print("I am CNN-RNN!")
        inputs = tf.nn.dropout(inputs, keep_prob=self.config.keep_prob)
        inputs = tf.reshape(inputs,
                            [-1, self.config.max_sen_len, self.config.word_dim])  # [b*max_doc_len,max_sen_len,word_dim]
        # word-sentence
        outputs_sen = add_cnn_layer(inputs, self.filter_list, self.filter_num, self.config.max_sen_len,
                                    self.config.keep_prob, scope_name="sen", drop_flag=False)
        outputs_sen_dim = self.filter_num * len(self.filter_list)
        outputs_sen = tf.reshape(outputs_sen, [-1, self.config.max_doc_len, outputs_sen_dim])

        # sentence-document
        cell = tf.contrib.rnn.LSTMCell
        outputs_doc = bi_dynamic_rnn(cell, outputs_sen, self.config.hidden_size, self.doc_len, self.config.max_doc_len,
                                     'doc', 'last')
        return outputs_doc

    def rnn_cnn(self, inputs):
        print("I am RNN-CNN!")
        inputs = tf.nn.dropout(inputs, keep_prob=self.config.keep_prob)
        inputs = tf.reshape(inputs, [-1, self.config.max_sen_len, self.config.word_dim])
        # word-sentence
        cell = tf.contrib.rnn.LSTMCell
        outputs_sen = bi_dynamic_rnn(cell, inputs, self.config.hidden_size, tf.reshape(self.sen_len, [-1]),
                                     self.config.max_sen_len, 'sen', 'last')
        outputs_sen = tf.reshape(outputs_sen, [-1, self.config.max_doc_len, 2 * self.config.hidden_size])
        # sentence-document
        outputs_doc = add_cnn_layer(outputs_sen, self.filter_list, self.filter_num, self.config.max_doc_len,
                                    self.config.keep_prob, scope_name='doc', drop_flag=False)
        return outputs_doc

    def hnn(self, inputs):
        print("I am LSTM-LSTM!")
        inputs = tf.nn.dropout(inputs, keep_prob=self.config.keep_prob)
        inputs = tf.reshape(inputs, [-1, self.config.max_sen_len, self.config.word_dim])
        # word-sentence
        cell = tf.contrib.rnn.LSTMCell
        outputs_sen = bi_dynamic_rnn(cell, inputs, self.config.hidden_size, tf.reshape(self.sen_len, [-1]),
                                     self.config.max_sen_len, 'sen', 'last')
        outputs_sen = tf.reshape(outputs_sen, [-1, self.config.max_doc_len, 2 * self.config.hidden_size])
        # sentence-document
        outputs_doc = bi_dynamic_rnn(cell, outputs_sen, self.config.hidden_size, self.doc_len, self.config.max_doc_len,
                                     'doc', 'last')
        return outputs_doc

    # hierarchical attention neural network
    def han(self, inputs):
        print("I am HAN!")
        inputs = tf.nn.dropout(inputs, keep_prob=self.config.keep_prob)
        inputs = tf.reshape(inputs, [-1, self.config.max_sen_len, self.config.word_dim])
        # word-sentence
        cell = tf.contrib.rnn.LSTMCell
        sen_len = tf.reshape(self.sen_len, [-1])
        hiddens_sen = bi_dynamic_rnn(cell, inputs, self.config.hidden_size, sen_len, self.config.max_sen_len, 'sen',
                                     'all')
        alpha = mlp_attention_layer(hiddens_sen, sen_len, 2 * self.config.hidden_size, self.config.l2_reg_lambda,
                                    self.config.random_base, 'sen')
        outputs_sen = tf.squeeze(tf.matmul(alpha, hiddens_sen))  # 删除大小是1的维度
        outputs_sen = tf.reshape(outputs_sen, [-1, self.config.max_doc_len, 2 * self.config.hidden_size])
        # sentence-document
        hiddens_doc = bi_dynamic_rnn(cell, outputs_sen, self.config.hidden_size, self.doc_len, self.config.max_doc_len,
                                     'doc', 'all')
        alpha = mlp_attention_layer(hiddens_doc, self.doc_len, 2 * self.config.hidden_size, self.config.l2_reg_lambda,
                                    self.config.random_base, 'doc')
        outputs_doc = tf.squeeze(tf.matmul(alpha, hiddens_doc))

        return outputs_doc

    def bigru_gru(self, inputs):  # 文档级句子级都是gru
        """
        word级是一层双向gru+word attention
        sentence级是一层单向的gru+sentence attention
        :param inputs: 
        :return: 
        """
        print("I am bigru_gru!")
        inputs = tf.nn.dropout(inputs, keep_prob=self.config.keep_prob)
        inputs = tf.reshape(inputs, [-1, self.config.max_sen_len, self.config.word_dim])
        # word-sentence
        gru_cells = [tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size) for _ in range(FLAGS.n_layers)]

        sen_len = tf.reshape(self.sen_len, [-1])
        stacked_fw = tf.contrib.rnn.MultiRNNCell(gru_cells, state_is_tuple=True)
        stacked_bw = tf.contrib.rnn.MultiRNNCell(gru_cells, state_is_tuple=True)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(stacked_fw, stacked_bw, inputs=inputs,
                                                          sequence_length=sen_len,
                                                          dtype=tf.float32)  # ,time_major=False
        # outputs(?,200,150)
        outputs = tf.concat(outputs, 2)  # (?,200,300)
        hiddens_sen = outputs

        output, w_alpha = attention(hiddens_sen, self.config.attention_dim,   name_scope="attention")
        self.w_alpha = tf.reshape(w_alpha,
                                  [self.config.batch_size, self.config.max_doc_len, self.config.max_sen_len])

        outputs_sen = tf.reshape(output, [-1, self.config.max_doc_len, self.config.hidden_size * 2])
        # sentence-document

        gru_cells = [tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size) for _ in range(FLAGS.n_layers)]
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(gru_cells)
        # 单向lstm
        outputs, state = tf.nn.dynamic_rnn(stacked_lstm, inputs=outputs_sen, dtype=tf.float32, time_major=False)
        hiddens_doc = outputs

        outputs_doc, s_alpha = attention(hiddens_doc, 2 * self.config.hidden_size,  name_scope="sen_attention")
        self.s_alpha = s_alpha
        return outputs_doc

    def bigru_bigru(self, inputs):  # 文档级句子级都是gru
        """
        word级是一层双向gru+word attention
        sentence级也是一层双向的gru+sentence attention
        :param inputs: 
        :return: 
        """
        print("I am bigru_bigru!")
        inputs = tf.nn.dropout(inputs, keep_prob=self.config.keep_prob)
        inputs = tf.reshape(inputs, [-1, self.config.max_sen_len, self.config.word_dim])
        # word-sentence
        gru_cells = [tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size) for _ in range(FLAGS.n_layers)]

        sen_len = tf.reshape(self.sen_len, [-1])
        stacked_fw = tf.contrib.rnn.MultiRNNCell(gru_cells, state_is_tuple=True)
        stacked_bw = tf.contrib.rnn.MultiRNNCell(gru_cells, state_is_tuple=True)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(stacked_fw, stacked_bw, inputs=inputs,
                                                          sequence_length=sen_len,
                                                          dtype=tf.float32)  # ,time_major=False

        outputs = tf.concat(outputs, 2)  # (?,200,300)
        hiddens_sen = outputs

        output, w_alpha = attention(hiddens_sen, self.config.attention_dim, name_scope="attention")
        self.w_alpha = tf.reshape(w_alpha,
                                  [self.config.batch_size, self.config.max_doc_len, self.config.max_sen_len])

        outputs_sen = tf.reshape(output, [-1, self.config.max_doc_len, self.config.hidden_size * 2])
        # sentence-document

        cell = tf.nn.rnn_cell.GRUCell
        outputs, state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell(self.config.hidden_size * 2),
            cell_bw=cell(self.config.hidden_size * 2),
            inputs=outputs_sen,
            sequence_length=self.doc_len,
            dtype=tf.float32,
            scope='doc')
        hiddens_doc = tf.concat(outputs, 2)

        outputs_doc, s_alpha = attention(hiddens_doc, 2 * self.config.hidden_size ,name_scope="sen_attention")
        self.s_alpha = s_alpha
        return outputs_doc

    def gru(self, inputs):
        print("I am gru!")
        inputs = tf.nn.dropout(inputs, keep_prob=self.config.keep_prob)
        inputs = tf.reshape(inputs, [-1, self.config.max_sen_len, self.config.word_dim])
        # word-sentence
        gru_cells = [tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size) for _ in range(FLAGS.n_layers)]
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(gru_cells)
        sen_len = tf.reshape(self.sen_len, [-1])
        hiddens_sen = gru_layer(stacked_lstm, inputs, self.config.hidden_size, sen_len, self.config.max_sen_len, 'sen',
                                'all')

        output, w_alpha = attention(hiddens_sen, self.config.attention_dim,)
        self.w_alpha = tf.reshape(w_alpha,
                                  [self.config.batch_size, self.config.max_doc_len, self.config.max_sen_len])

        outputs_sen = tf.reshape(output, [-1, self.config.max_doc_len, self.config.hidden_size])
        # sentence-document
        cell = tf.contrib.rnn.LSTMCell

        hiddens_doc = bi_dynamic_rnn(cell, outputs_sen, self.config.hidden_size, self.doc_len, self.config.max_doc_len,
                                     'doc', 'all')

        outputs_doc, s_alpha = attention(hiddens_doc, 2 * self.config.hidden_size,  name_scope="sen_attention")

        self.s_alpha = s_alpha

        return outputs_doc

    # def lstm_cell(self):
    #     cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_size)
    #     return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.config.keep_prob)


def train_run(model):
    print("Training start:\n")
    saver = tf.train.Saver(max_to_keep=3)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if not os.path.exists(model.config.save_path + 'model/'):
            os.makedirs(model.config.save_path + 'model/')
        if not os.path.exists(model.config.save_path + 'result/'):
            os.makedirs(model.config.save_path + 'result/')

        save_model_f = model.config.save_path + 'model/' + model.config.src_domain + '_' + model.config.tar_domain + '/model'
        save_result_f = open(
            model.config.save_path + 'result/' + model.config.src_domain + '_' + model.config.tar_domain + '.txt',
            encoding="UTF-8", mode="w+")

        if os.path.exists(save_model_f):
            saver.restore(sess, save_model_f)

        best_val_acc = 0
        best_test_acc = 0
        best_val_epoch = 0
        best_test_accuracy = 0

        src_train_data, src_valid_data = load_train_data(model.config, model.config.src_domain, transform=False,
                                                         train_flag=True)
        tar_train_data, tar_test_data = load_train_data(model.config, model.config.tar_domain, transform=False,
                                                        train_flag=False)

        for epoch in range(model.config.epoches):
            lr = 1e-3
            lr_wd_D = 1e-3

            print("=" * 20 + " " + model.config.src_domain + "-" + model.config.tar_domain + " Epoch", epoch,
                  "=" * 20 + "\n")
            data_len = len(src_train_data[0])
            iterations = data_len // (model.config.batch_size // 2)
            for step in range(iterations):
                x0, sen_len0, doc_len0, y0, tag_x0 = next(
                    batches_iter(src_train_data, batch_size=model.config.batch_size // 2, transform=False))
                x1, sen_len1, doc_len1, y1, tag_x1 = next(
                    batches_iter(tar_train_data, batch_size=model.config.batch_size // 2, transform=False))

                X = np.vstack([x0, x1])
                Sen_len = np.vstack([sen_len0, sen_len1])
                Doc_len = np.concatenate((doc_len0, doc_len1), axis=0)
                Y = np.vstack([y0, y1])
                Tag_x = np.vstack([tag_x0, tag_x1])

                for _ in range(model.config.w_training):
                    _ = sess.run(model.wd_d_op, feed_dict={model.x: X, model.sen_len: Sen_len, model.doc_len: Doc_len,
                                                           model.is_train: True,
                                                           model.lr_wd_D: lr_wd_D})

                _, total_cost, s_acc, s_cost = sess.run(
                    [model.P_train_op, model.total_loss_p, model.s_acc, model.senti_cost],
                    feed_dict={model.x: X, model.sen_len: Sen_len, model.doc_len: Doc_len,
                               model.doc_y: Y, model.is_train: True,
                               model.learning_rate: lr})

                if step % 10 == 0:
                    print("epoch:%d , step:%d ,total_cost:%f, s_cost:%f , s_acc:%f " % (
                        epoch, step, total_cost, s_cost, s_acc))

            valid_s_acc = sess.run(model.s_acc,
                                   feed_dict={model.x: src_valid_data[0], model.sen_len: src_valid_data[1],
                                              model.doc_len: src_valid_data[2], model.doc_y: src_valid_data[3],
                                              model.is_train: False})

            test_s_acc = sess.run(model.s_acc,
                                  feed_dict={model.x: tar_test_data[0], model.sen_len: tar_test_data[1],
                                             model.doc_len: tar_test_data[2], model.doc_y: tar_test_data[3],
                                             model.is_train: False})
            save_str = 'step:' + str(round(epoch, 4)) + '\tval_s_acc:' + str(
                round(valid_s_acc, 4)) + '\ttest_s_acc:' + str(round(test_s_acc, 4)) + "\n"
            print(save_str)
            save_result_f.write(save_str)

            if valid_s_acc > best_val_acc:
                best_val_acc = valid_s_acc
                best_val_epoch = epoch
                best_test_acc = test_s_acc
                path = saver.save(sess, save_model_f, epoch)
                print("Saved model checkpoint to {}\n".format(path))
            if best_test_accuracy < test_s_acc:
                best_test_accuracy = test_s_acc
            if epoch - best_val_epoch > 10:
                break

            save_str = "Best val acc:" + str(round(best_val_acc, 4)) + "\t test acc:" + str(
                round(best_test_acc, 4)) + "\nBest test acc:" + str(round(best_test_accuracy, 4))
            print(save_str)
            save_result_f.write(save_str)
            print("=" * 50)

        print("Best test acc={}".format(best_test_accuracy))
    print("Training complete!")
    save_result_f.close()


def make_pivots(model):
    print("Start to pick up pivots:\n")
    saver = tf.train.Saver(max_to_keep=3)
    P_pivots, N_pivots, P_pivots_num, N_pivots_num = [], [], dict(), dict()
    cross_domain_pairs = model.config.src_domain + "_" + model.config.tar_domain

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        save_model_f = model.config.save_path + 'model/' + model.config.src_domain + '_' + model.config.tar_domain + '/'
        model_f = tf.train.latest_checkpoint(save_model_f)
        saver.restore(sess, model_f)

        src_train_data, src_valid_data = load_train_data(model.config, model.config.src_domain, transform=False,
                                                         train_flag=True)
        tar_train_data, tar_test_data = load_train_data(model.config, model.config.tar_domain, transform=False,
                                                        train_flag=False)

        data_len = len(src_train_data[0]) // 2
        iterations = data_len // (model.config.batch_size // 2)

        for step in range(iterations):
            print("%d iterations:" % step)
            x0, sen_len0, doc_len0, y0, tag_x0 = next(
                batches_iter(src_train_data, batch_size=model.config.batch_size // 2, transform=False))
            x1, sen_len1, doc_len1, y1, tag_x1 = next(
                batches_iter(tar_train_data, batch_size=model.config.batch_size // 2, transform=False))

            X = np.vstack([x0, x1])
            Sen_len = np.vstack([sen_len0, sen_len1])
            Doc_len = np.concatenate((doc_len0, doc_len1), axis=0)
            Y = np.vstack([y0, y1])
            Tag_x = np.vstack([tag_x0, tag_x1])

            w_alpha, s_alpha = sess.run(
                [model.w_alpha, model.s_alpha],
                feed_dict={model.x: X, model.sen_len: Sen_len, model.doc_len: Doc_len,
                           model.doc_y: Y, model.tag_x: Tag_x, model.is_train: True})

            p_pivots, n_pivots, p_pivots_num, n_pivots_num = get_pivots(s_alpha, w_alpha, X, Y, Tag_x, model.config)

            P_pivots.extend(p_pivots)
            N_pivots.extend(n_pivots)
            P_pivots_num.update(p_pivots_num)
            N_pivots_num.update(n_pivots_num)

        print("P_pivots:", P_pivots_num)
        print("N_pivots:", N_pivots_num)
        pivots_path = model.config.save_path + "pivots/" + model.config.model_name + "/"

        if not os.path.exists(pivots_path):
            os.makedirs(pivots_path)
        pos_f = open(pivots_path + cross_domain_pairs + "_pivots_num_pos.txt", "w")
        neg_f = open(pivots_path + cross_domain_pairs + "_pivots_num_neg.txt", "w")
        pos_f.write(json.dumps(p_pivots_num))
        neg_f.write(json.dumps(n_pivots_num))
        pos_f.close()
        neg_f.close()

        filename = pivots_path + cross_domain_pairs + "_words_pos.txt"
        with open(filename, mode="w+", encoding="UTF-8") as p_f:
            for word in p_pivots:
                p_f.write(word + '\n')
        p_f.close()

        filename = pivots_path + cross_domain_pairs + "_words_neg.txt"
        with open(filename, mode="w+", encoding="UTF-8") as n_f:
            for word in n_pivots:
                n_f.write(word + '\n')
        n_f.close()


with tf.Graph().as_default():
    if not os.path.exists(FLAGS.embedding_matrix + ".npy"):
        generate_matrix(FLAGS.word_dim)
    embedding_matrix = np.load(FLAGS.embedding_matrix + ".npy")
    print('Loaded the word vectors!')
    print(embedding_matrix.shape)
    #
    model = Model(embedding_matrix)

    # 1
    train_run(model)

    # # # 2
    # make_pivots(model)  # pos


