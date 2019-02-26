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
import matplotlib.pyplot as plt


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
    temp = temp.split(',')
    temp = [float(x) for x in temp]
    return temp


if __name__ == '__main__':
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # parameters of Slice
    lr = 0.003  # learning rate
    training_iters = 100000
    batch_size = 128
    n_inputs = 60
    n_hidden_units = 8 * n_inputs
    n_classes = 2
    is_train = False
    web_name = 'amazon' #sina/youku/amazon
    pkeep=0.8
    train_loss=[]
    train_accuracy=[]
    test_accuracy=[]
    test_loss=[]

    x = tf.placeholder(tf.float32, [None, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_classes])

    # hidden layer1
    Weights1 = tf.Variable(tf.random_normal([n_inputs, n_hidden_units]))
    biases1 = tf.Variable(tf.zeros([1, n_hidden_units]) + 0.1)
    Wx_plus_b1 = tf.matmul(x, Weights1) + biases1
    l1 = tf.nn.relu(Wx_plus_b1)
    l1d = tf.nn.dropout(l1, pkeep)
    # hidden layer2
    Weights2 = tf.Variable(tf.random_normal([n_hidden_units, n_hidden_units]))
    biases2 = tf.Variable(tf.zeros([1, n_hidden_units]) + 0.1)
    Wx_plus_b2 = tf.matmul(l1d, Weights2) + biases2
    l2 = tf.nn.relu(Wx_plus_b2)
    l2d = tf.nn.dropout(l2, pkeep)
    # output layer
    Weights3 = tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
    biases3 = tf.Variable(tf.zeros([1, n_classes]) + 0.1)
    prediction = tf.matmul(l2d, Weights3) + biases3

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)
    y_pred = tf.argmax(prediction, 1)
    y_true = tf.argmax(y, 1)

    init = tf.global_variables_initializer()
    # read data
    file_dir = './TrainingData/%s' % web_name
    filename_list = file_name_list(file_dir)
    df = data_merge(file_dir, filename_list)
    slice_data = np.array(df['Slice'].map(process_read_X_data).values.tolist())
    print(df['label'].value_counts())
    label = OneHotEncoder(sparse=False).fit_transform(df['label'].values.reshape((-1, 1)))
    X_train, X_test, y_train, y_test = train_test_split(slice_data, label, test_size=0.3, random_state=42)
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
                        train_loss.append(sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}))
                        train_accuracy.append(accuracy_score(y_true_label, y_pred_label))
                        print(accuracy_score(y_true_label, y_pred_label))
                    if step % 20 ==0:
                        num_pred = len(X_test)
                        X_test = X_test.reshape([num_pred, n_inputs])
                        y_pred_label = sess.run(y_pred, feed_dict={x: X_test, y: y_test})
                        y_true_label = sess.run(y_true, feed_dict={x: X_test, y: y_test})
                        test_accuracy.append(accuracy_score(y_true_label, y_pred_label))
                        test_loss.append(sess.run(cost,feed_dict={x: X_test, y: y_test}))
                    step += 1
                saver.save(sess, './my_slice_model/my_slice_model_%s' % (web_name))
    train_end=time.time()
    with tf.variable_scope('train', reuse=True):
        with tf.Session() as sess:
            print('test:')
            sess.run(init)
            saver.restore(sess, './my_slice_model/my_slice_model_%s' % (web_name))
            num_pred = len(X_test)
            X_test = X_test.reshape([num_pred, n_inputs])
            y_pred_label = sess.run(y_pred, feed_dict={x: X_test, y: y_test})
            y_true_label = sess.run(y_true, feed_dict={x: X_test, y: y_test})
            print('accuracy\t', accuracy_score(y_true_label, y_pred_label))
            print('precision\t', precision_score(y_true_label, y_pred_label))
            print('recall\t', recall_score(y_true_label, y_pred_label))
            print('f1 score\t', f1_score(y_true_label, y_pred_label))
    test_end=time.time()
    print('train time\t',train_end-train_start)
    print('test time\t',test_end-train_end)
    print('train loss\t',train_loss)
    print('test loss\t',test_loss)
    print('train accuracy\t',train_accuracy)
    print('test accuracy\t',test_accuracy)
    # plt.figure()
    # plt.plot(train_loss)
    # plt.plot(test_loss)
    # plt.ylim((0))
    # plt.show()