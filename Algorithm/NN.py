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
import time


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


if __name__ == '__main__':
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # NN各种参数定义
    lr = 0.003  # 学习速率
    training_iters = 100000  # 循环次数
    batch_size = 128
    n_inputs = 5  # 代表每个特征有多少维
    n_hidden_units = 8 * n_inputs  # 假设隐藏单元有128个
    n_classes = 2  # 二分类
    is_train = False
    web_name = 'amazon'

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
    # 读取数据

    file_dir = './TrainingData/%s' % web_name
    filename_list = file_name_list(file_dir)
    df = data_merge(file_dir, filename_list)
    df['SumSize'] = df['SumSize'].map(process_read_X_data)
    # print(df['label'].value_counts())
    df['Max_slope'] = df.apply(lambda x: max_slope(x.SumSize), axis=1)
    percent_list = [0.25, 0.5, 0.75, 0.9]
    for percent in percent_list:
        df['%s' % (percent)] = df.apply(lambda x: percentage_value(x.SumSize, percent), axis=1)
    index_list = [str(i) for i in percent_list]
    index_list.append('Max_slope')
    x_train = df[index_list].values
    label = OneHotEncoder(sparse=False).fit_transform(df['label'].values.reshape((-1, 1)))
    X_train, X_test, y_train, y_test = train_test_split(x_train, label, test_size=0.3, random_state=42)
    train_data = DataSet(X_train, y_train, len(X_train))
    saver = tf.train.Saver()
    train_start=time.time()
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
                saver.save(sess, './my_nn_model/my_nn_model_%s' % (web_name))
    train_end=time.time()
    with tf.variable_scope('train', reuse=True):
        with tf.Session() as sess:
            print('test:')
            sess.run(init)
            saver.restore(sess, './my_nn_model/my_nn_model_%s' % (web_name))
            num_pred = len(X_test)
            X_test = X_test.reshape([num_pred, n_inputs])
            y_pred_label = sess.run(y_pred, feed_dict={x: X_test, y: y_test})
            y_true_label = sess.run(y_true, feed_dict={x: X_test, y: y_test})
            print('lr\t',lr)
            print('h_units\t',n_hidden_units)
            print('accuracy\t', accuracy_score(y_true_label, y_pred_label))
            print('precision\t', precision_score(y_true_label, y_pred_label))
            print('recall\t', recall_score(y_true_label, y_pred_label))
            print('f1 score\t', f1_score(y_true_label, y_pred_label))
    test_end=time.time()
    print('train time\t',train_end-train_start)
    print('test time\t',test_end-train_end)