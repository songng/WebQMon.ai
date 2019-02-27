# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import re
from sklearn.model_selection import train_test_split
from train_data_merge import data_merge
from sklearn.metrics import recall_score, precision_score, f1_score
import os


class DataSet(object):
    def __init__(self, images, labels, num_examples):
        self._images = images
        self._labels = labels
        self._epochs_completed = 0  # 完成遍历轮数
        self._index_in_epochs = 0  # 调用next_batch()函数后记住上一次位置
        self._num_examples = num_examples  # 训练样本数

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


def process_read_X_data(temp_str):
    temp = re.sub('[\r\n\t\[\]]', '', temp_str)
    temp = temp.split(' ')
    while '' in temp:
        temp.remove('')
    temp = [float(x) for x in temp]
    return temp


def RNN(X, weights, biases):
    # X在输入时是一批128个，每批中有28行，28列，因此其shape为(128, 28, 28)。为了能够进行 weights 的矩阵乘法，我们需要把输入数据转换成二维的数据(128*28, 28)
    X = tf.reshape(X, [-1, n_inputs])

    # 对输入数据根据权重和偏置进行计算, 其shape为(128batch * 28steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']

    # 矩阵计算完成之后，又要转换成3维的数据结构了，(128batch, 28steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell，使用LSTM，其中state_is_tuple用来指示相关的state是否是一个元组结构的，如果是元组结构的话，会在state中包含主线状态和分线状态
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)

    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, dtype=tf.float32)
    # 计算结果，其中states[1]为分线state，也就是最后一个输出值
    results = tf.matmul(states[1], weights['out']) + biases['out']
    return results


if __name__ == '__main__':
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # RNN各种参数定义
    web_name = 'sina-Tenary'
    is_train = False
    lr = 0.003 # 学习速率
    training_iters = 100000  # 循环次数
    batch_size = 128
    n_inputs = 1  # 代表每个特征有多少维
    n_steps = 99  # 代表有多少需要迭代处理的特征
    n_hidden_units = 256
    n_classes = 3  # 二分类

    # 定义输入和输出的placeholder
    x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_classes])
    # 对weights和biases初始值定义
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
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)
    y_pred = tf.argmax(pred, 1)
    y_true = tf.argmax(y, 1)
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

    # 读取数据
    # df = pd.read_csv("F:/coding/Network quality assessment/RNN/Train/train_sina.csv")

    file_dir = './TrainingData/%s' % web_name
    filename_list = file_name_list(file_dir)
    df = data_merge(file_dir, filename_list)
    # print(df['label'].value_counts())
    sum_size = np.array(df['SumSize'].map(process_read_X_data).map(lambda l: list(reversed(l))).map(
        lambda l: np.delete(l, -1)).values.tolist())

    label = OneHotEncoder(sparse=False).fit_transform(df['label'].values.reshape((-1, 1)))
    X_train, X_test, y_train, y_test = train_test_split(sum_size, label, test_size=0.3, random_state=42)
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
                    batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
                    sess.run([train_op], feed_dict={x: batch_xs, y: batch_ys})
                    if step % 20 == 0:
                        print(sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys}))
                    step += 1
                saver.save(sess, './my_lstm_model/my_multi_Rlstm_model_%s' % (web_name))

    with tf.variable_scope('train', reuse=True):
        with tf.Session() as sess:
            print('test:')
            sess.run(init)
            saver.restore(sess, './my_lstm_model/my_multi_Rlstm_model_%s' % (web_name))
            num_pred = len(X_test)
            X_test = X_test.reshape([num_pred, n_steps, n_inputs])
            y_pred_label = sess.run(y_pred, feed_dict={x: X_test, y: y_test})
            y_true_label = sess.run(y_true, feed_dict={x: X_test, y: y_test})
            print('accuracy:\t',sess.run(accuracy, feed_dict={x: X_test, y: y_test}))
            print('precision:\t', precision_score(y_true_label, y_pred_label, average='macro'))
            print('recall:\t', recall_score(y_true_label, y_pred_label, average='macro'))
            print('f1 score:\t', f1_score(y_true_label, y_pred_label, average='macro'))
