# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import re
from sklearn.model_selection import train_test_split
from train_data_merge import data_merge
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
import os
import copy


class DataSet(object):
    def __init__(self, images, labels, num_examples):
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epochs = 0
        self._num_examples = num_examples

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        start = self._index_in_epochs

        if self._epochs_completed == 0 and start == 0 and shuffle:
            index0 = np.arange(self._num_examples)
            np.random.shuffle(index0)
            self._images = np.array(self._images)[index0]
            self._labels = np.array(self._labels)[index0]

        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            if shuffle:
                index = np.arange(self._num_examples)
                np.random.shuffle(index)
                self._images = self._images[index]
                self._labels = self._labels[index]
            start = 0
            self._index_in_epochs = batch_size - rest_num_examples
            end = self._index_in_epochs
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)

        else:
            self._index_in_epochs += batch_size
            end = self._index_in_epochs
            return self._images[start:end], self._labels[start:end]


def file_name_list(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return files


def process_read_SumSize_data(temp_str):
    temp = re.sub('[\r\n\t\[\]]', '', temp_str)
    temp = temp.split(' ')
    while '' in temp:
        temp.remove('')
    temp = [float(x) for x in temp]
    return temp

def process_read_Slice_data(temp_str):
    temp = re.sub('[\r\n\t\[\]]', '', temp_str)
    temp = temp.split(',')
    temp = [float(x) for x in temp]
    return temp

def RNN(X, weights, biases):
    X = tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, dtype=tf.float32)
    results = tf.matmul(states[1], weights['out']) + biases['out']
    return results


def max_slope(list, interval=0.01):
    l1 = copy.deepcopy(list)
    l1.pop(0)
    l1.append(0)
    l2 = (np.array(l1) - np.array(list)) / interval
    return (round(l2.max(), 1)) / 50


def percentage_value(list, percent):
    arr = np.array(list)
    MAX = max(arr)
    return (len(arr) - len(arr[arr > MAX * percent])) / 100

def my_vote(a):
    counts=np.bincount(a)
    return np.argmax(counts)

if __name__ == '__main__':
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    is_train = False
    # read data
    web_name = 'sina'
    file_dir = './TrainingData/%s' % web_name
    filename_list = file_name_list(file_dir)
    df = data_merge(file_dir, filename_list)
    df['SumSize'] = df['SumSize'].map(process_read_SumSize_data)
    print(df['label'].value_counts())
    df['Max_slope'] = df.apply(lambda x: max_slope(x.SumSize), axis=1)
    percent_list = [0.25, 0.5, 0.75, 0.9]
    for percent in percent_list:
        df['%s' % (percent)] = df.apply(lambda x: percentage_value(x.SumSize, percent), axis=1)
    index_list = [str(i) for i in percent_list]
    index_list.append('Max_slope')
    nn_input = df[index_list].values
    sum_size = np.array(df['SumSize'].map(lambda l: list(reversed(l))).map(lambda l: np.delete(l, -1)).values.tolist())
    label = OneHotEncoder(sparse=False).fit_transform(df['label'].values.reshape((-1, 1)))
    slice_data = np.array(df['Slice'].map(process_read_Slice_data).values.tolist())
    sum_size_train, sum_size, nn_train, nn_input,slice_train,slice_data,y_train,label = train_test_split(sum_size, nn_input,slice_data,label, test_size=0.3, random_state=42)
    with tf.Graph().as_default() as g_NN:
        # lr = 0.003  # learning rate
        # training_iters = 100000
        # batch_size = 128
        n_inputs = 5
        n_hidden_units = 8 * n_inputs
        n_classes = 2

        x = tf.placeholder(tf.float32, [None, n_inputs])
        y = tf.placeholder(tf.float32, [None, n_classes])

        # hidden layer1
        Weights1 = tf.Variable(tf.random_normal([n_inputs, n_hidden_units]))
        biases1 = tf.Variable(tf.zeros([1, n_hidden_units]) + 0.1)
        Wx_plus_b1 = tf.matmul(x, Weights1) + biases1
        l1 = tf.nn.relu(Wx_plus_b1)
        # hidden layer2
        Weights2 = tf.Variable(tf.random_normal([n_hidden_units, n_hidden_units]))
        biases2 = tf.Variable(tf.zeros([1, n_hidden_units]) + 0.1)
        Wx_plus_b2 = tf.matmul(l1, Weights2) + biases2
        l2 = tf.nn.relu(Wx_plus_b2)
        # output layer
        Weights3 = tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
        biases3 = tf.Variable(tf.zeros([1, n_classes]) + 0.1)
        prediction = tf.matmul(l2, Weights3) + biases3
        probability = tf.nn.softmax(prediction)
        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
        # train_op = tf.train.AdamOptimizer(lr).minimize(cost)
        y_pred = tf.argmax(prediction, 1)
        y_true = tf.argmax(y, 1)

        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        with tf.Session(graph=g_NN) as sess:
            sess.run(init)
            print('NN:')
            saver.restore(sess, './my_nn_model/my_nn_model_%s' % (web_name))
            x_drop,x_test,y_drop,y_test=train_test_split(nn_input, label, test_size=0.3, random_state=42)
            nn_input = nn_input.reshape([len(nn_input), n_inputs])
            prob_NN = sess.run(probability, feed_dict={x: nn_input})
            x_test= x_test.reshape([len(x_test), n_inputs])
            y_pred_label = sess.run(y_pred, feed_dict={x: x_test, y: y_test})
            pred_NN=y_pred_label
            y_true_label = sess.run(y_true, feed_dict={x: x_test, y: y_test})
            print('accuracy:\t', accuracy_score(y_true_label, y_pred_label))
            print('precision:\t', precision_score(y_true_label, y_pred_label))
            print('recall:\t', recall_score(y_true_label, y_pred_label))
            print('f1 score:\t', f1_score(y_true_label, y_pred_label))

    with tf.Graph().as_default() as g_LSTM:
        # lr = 0.003
        n_inputs = 1
        n_steps = 99
        n_hidden_units = 256
        n_classes = 2

        x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
        y = tf.placeholder(tf.float32, [None, n_classes])
        weights = {
            # shape(n_inputs, n_hidden_units)
            'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
            # shape(n_hidden_units , n_classes)
            'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
        }

        biases = {
            # shape(n_hidden_units, )
            'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
            # shape(n_classes, )
            'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
        }

        pred = RNN(x, weights, biases)
        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        # train_op = tf.train.AdamOptimizer(lr).minimize(cost)
        y_pred = tf.argmax(pred, 1)
        y_true = tf.argmax(y, 1)
        probability_LSTM = tf.nn.softmax(pred)
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        with tf.Session(graph=g_LSTM) as sess:
            sess.run(init)
            print('LSTM')
            saver.restore(sess, './my_lstm_model/my_lstm_model_%s' % (web_name))
            x_drop,x_test,y_drop,y_test=train_test_split(sum_size, label, test_size=0.3, random_state=42)
            sum_size = sum_size.reshape([len(sum_size), n_steps, n_inputs])
            prob_LSTM = sess.run(probability_LSTM, feed_dict={x: sum_size})
            x_test= x_test.reshape([len(x_test), n_steps, n_inputs])
            y_pred_label = sess.run(y_pred, feed_dict={x: x_test, y: y_test})
            pred_LSTM=y_pred_label
            y_true_label = sess.run(y_true, feed_dict={x: x_test, y: y_test})
            print('accuracy:\t', accuracy_score(y_true_label, y_pred_label))
            print('precision:\t', precision_score(y_true_label, y_pred_label))
            print('recall:\t', recall_score(y_true_label, y_pred_label))
            print('f1 score:\t', f1_score(y_true_label, y_pred_label))

    with tf.Graph().as_default() as g_Slice:
        # lr = 0.003
        # training_iters = 100000
        # batch_size = 128
        n_inputs = 60
        n_hidden_units = 8 * n_inputs
        n_classes = 2

        x = tf.placeholder(tf.float32, [None, n_inputs])
        y = tf.placeholder(tf.float32, [None, n_classes])

        # hidden layer1
        Weights1 = tf.Variable(tf.random_normal([n_inputs, n_hidden_units]))
        biases1 = tf.Variable(tf.zeros([1, n_hidden_units]) + 0.1)
        Wx_plus_b1 = tf.matmul(x, Weights1) + biases1
        l1 = tf.nn.relu(Wx_plus_b1)
        # hidden layer2
        Weights2 = tf.Variable(tf.random_normal([n_hidden_units, n_hidden_units]))
        biases2 = tf.Variable(tf.zeros([1, n_hidden_units]) + 0.1)
        Wx_plus_b2 = tf.matmul(l1, Weights2) + biases2
        l2 = tf.nn.relu(Wx_plus_b2)
        # output layer
        Weights3 = tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
        biases3 = tf.Variable(tf.zeros([1, n_classes]) + 0.1)
        prediction = tf.matmul(l2, Weights3) + biases3
        probability = tf.nn.softmax(prediction)
        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
        # train_op = tf.train.AdamOptimizer(lr).minimize(cost)
        y_pred = tf.argmax(prediction, 1)
        y_true = tf.argmax(y, 1)

        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        with tf.Session(graph=g_Slice) as sess:
            sess.run(init)
            print('Slice:')
            saver.restore(sess, './my_nn_model/my_slice_nn_model_%s' % (web_name))
            x_drop,x_test,y_drop,y_test=train_test_split(slice_data,label, test_size=0.3, random_state=42)
            slice_data = slice_data.reshape([len(slice_data), n_inputs])
            prob_Slice = sess.run(probability, feed_dict={x: slice_data})
            x_test= x_test.reshape([len(x_test), n_inputs])
            y_pred_label = sess.run(y_pred, feed_dict={x: x_test, y: y_test})
            pred_Slice=y_pred_label
            y_true_label = sess.run(y_true, feed_dict={x: x_test, y: y_test})


            print('accuracy:\t', accuracy_score(y_true_label, y_pred_label))
            print('precision:\t', precision_score(y_true_label, y_pred_label))
            print('recall:\t', recall_score(y_true_label, y_pred_label))
            print('f1 score:\t', f1_score(y_true_label, y_pred_label))
            # g1def = graph_util.convert_variables_to_constants(
            # 	sess,
            # 	sess.graph_def,
            # 	["one_output"],
            # 	variable_names_whitelist=None,
            # 	variable_names_blacklist=None)

    #投票阶段
    # print('Vote:')
    # df_pred=pd.DataFrame({'NN':pred_NN,'LSTM':pred_LSTM,'Slice':pred_Slice})
    # my_pred=np.array(list(map(my_vote,df_pred.values)))
    # y_pred_label=OneHotEncoder(sparse=False).fit_transform(my_pred.reshape((-1, 1)))
    # y_true_label=label
    # print('accuracy:\t', accuracy_score(y_true_label, y_pred_label))
    # print('precision:\t', precision_score(y_true_label, y_pred_label,average='macro'))
    # print('recall:\t', recall_score(y_true_label, y_pred_label,average='macro'))
    # print('f1 score:\t', f1_score(y_true_label, y_pred_label,average='macro'))

    with tf.Graph().as_default() as g_combine:
        # NN各种参数定义
        lr = 0.003  # 学习速率
        training_iters = 100000  # 循环次数
        batch_size = 128
        n_inputs = 6  # 代表每个特征有多少维
        n_hidden_units = 8 * n_inputs  # 假设隐藏单元有128个
        n_classes = 2  # 二分类

        x = tf.placeholder(tf.float32, [None, n_inputs])
        y = tf.placeholder(tf.float32, [None, n_classes])

        # 隐层1
        Weights1 = tf.Variable(tf.random_normal([n_inputs, n_hidden_units]))
        biases1 = tf.Variable(tf.zeros([1, n_hidden_units]) + 0.1)
        Wx_plus_b1 = tf.matmul(x, Weights1) + biases1
        l1 = tf.nn.relu(Wx_plus_b1)
        # 隐层2
        Weights2 = tf.Variable(tf.random_normal([n_hidden_units, n_hidden_units]))
        biases2 = tf.Variable(tf.zeros([1, n_hidden_units]) + 0.1)
        Wx_plus_b2 = tf.matmul(l1, Weights2) + biases2
        l2 = tf.nn.relu(Wx_plus_b2)
        # 输出层
        Weights3 = tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
        biases3 = tf.Variable(tf.zeros([1, n_classes]) + 0.1)
        prediction = tf.matmul(l2, Weights3) + biases3

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
        train_op = tf.train.AdamOptimizer(lr).minimize(cost)
        y_pred = tf.argmax(prediction, 1)
        y_true = tf.argmax(y, 1)

        init = tf.global_variables_initializer()
        # 训练数据
        x_train = np.concatenate((prob_LSTM, prob_NN,prob_Slice),axis=1)
        X_train, X_test, y_train, y_test = train_test_split(x_train, label, test_size=0.3, random_state=42)
        train_data = DataSet(X_train, y_train, len(X_train))
        saver = tf.train.Saver()

        if is_train == False:
            with tf.variable_scope('train'):
                with tf.Session() as sess:
                    print('train:')
                    sess.run(init)
                    step = 0
                    while step * batch_size < training_iters:
                        batch_xs, batch_ys = train_data.next_batch(batch_size)
                        batch_xs = batch_xs.reshape([batch_size, n_inputs])
                        sess.run([train_op], feed_dict={x: batch_xs, y: batch_ys})
                        if step % 20 == 0:
                            y_pred_label = sess.run(y_pred, feed_dict={x: batch_xs, y: batch_ys})
                            y_true_label = sess.run(y_true, feed_dict={x: batch_xs, y: batch_ys})
                            print(accuracy_score(y_true_label, y_pred_label))
                        step += 1
                    saver.save(sess, './my_combine_model/my_combine_nn_model_%s' % (web_name))

        with tf.variable_scope('train', reuse=True):
            with tf.Session() as sess:
                print('test:')
                sess.run(init)
                saver.restore(sess, './my_combine_model/my_combine_nn_model_%s' % (web_name))
                num_pred = len(X_test)
                X_test = X_test.reshape([num_pred, n_inputs])
                y_pred_label = sess.run(y_pred, feed_dict={x: X_test, y: y_test})
                y_true_label = sess.run(y_true, feed_dict={x: X_test, y: y_test})
                print('accuracy:\t', accuracy_score(y_true_label, y_pred_label))
                print('precision:\t', precision_score(y_true_label, y_pred_label))
                print('recall:\t', recall_score(y_true_label, y_pred_label))
                print('f1 score:\t', f1_score(y_true_label, y_pred_label))
