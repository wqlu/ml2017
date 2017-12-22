# -*- coding: utf-8
# !/usr/bin/env python

import numpy as np
import pandas as pd
import os


def load_data(train_data_path, train_label_path, test_data_path):
    """加载数据"""
    X_train = pd.read_csv(train_data_path, sep=',', header=0)
    X_train = np.array(X_train.values)
    Y_train = pd.read_csv(train_label_path, sep=',', header=0)
    Y_train = np.array(Y_train.values)
    X_test = pd.read_csv(test_data_path, sep=',', header=0)
    X_test = np.array(X_test.values)

    return (X_train, Y_train, X_test)


def normalization(X_all, X_test):
    """数据归一化处理"""
    X_train_test = np.concatenate((X_all, X_test), axis=0)
    mu = (sum(X_train_test) / X_train_test.shape[0])
    # std = sqrt(mean(abs(x - x.mean()) ** 2)).
    sigma = np.std(X_train_test, axis=0)

    # 目的就是使维度相同
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma

    # split to train, test
    X_all = X_train_test_normed[0:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
    return X_all, X_test


def _shuffle(X, Y):
    '''将顺序打乱'''
    randomsize = np.arange(len(X))
    np.random.shuffle(randomsize)
    return (X[randomsize], Y[randomsize])


def split_valid_set(X_all, Y_all, percentage):
    '''分出数据作为验证数据'''
    all_data_size = len(X_all)
    valid_data_size = int(float(all_data_size*percentage))

    X_all, Y_all = _shuffle(X_all, Y_all)

    X_train, Y_train = X_all[0: valid_data_size], Y_all[0: valid_data_size]
    X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

    return X_train, Y_train, X_valid, Y_valid


def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    # 限制最小和最大值
    return np.clip(res, 1e-8, 1-(1e-8))


def valid(w, b, X_valid, Y_valid):
    valid_data_size = len(X_valid)

    z = (np.dot(X_valid, w) + b)
    y = sigmoid(z)
    y_ = np.around(y)
    result = (Y_valid == y_)
    print "Validation acc = %f" % (float(result.sum()) / valid_data_size)
    return



def train(X_train, Y_train, save_dir):
    '''训练模型'''
    # split a 10%-validation set from the training set
    valid_set_percentage = 0.1
    X_train, Y_train, X_valid, Y_valid = split_valid_set(X_train, Y_train, valid_set_percentage)

    # Initialize parameter, hyperparameter
    w = np.zeros((106,1))
    b = np.zeros((1,1))
    l_rate = 0.1
    batch_size = 32
    train_data_size = len(X_train)
    epoch_num = 1000
    save_param_iter = 50
    step_num = int(float(train_data_size/batch_size))

    # starting training
    total_loss = 0.0
    for epoch in range(1, epoch_num):
        # do validation and parameter saving
        if (epoch) % save_param_iter == 0:
            print '====Saving Param at epoch %d====' % epoch
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            np.savetxt(os.path.join(save_dir, 'w'), w)
            np.savetxt(os.path.join(save_dir, 'b'), b)
            print 'epoch avg loss = %f' % (total_loss / (float(save_param_iter)*train_data_size))
            total_loss = 0.0

            valid(w, b, X_valid, Y_valid)

        # random shuffle
        X_train, Y_train = _shuffle(X_train, Y_train)

        # training with batch
        for idx in range(step_num):
            X = X_train[idx*batch_size: (idx+1)*batch_size]
            Y = Y_train[idx*batch_size: (idx+1)*batch_size]

            z = np.dot(X, w) + b
            y = sigmoid(z)

            cross_entropy = -1 * (np.dot(np.squeeze(Y), np.log(y)) + np.dot((1 - np.squeeze(Y)), np.log(1 - y)))
            # print cross_entropy.shape
            total_loss += cross_entropy.sum()

            w_grad = np.mean(-1*X*(Y-y), axis=0).reshape((106, 1))
            b_grad = np.mean(-1*(Y-y))
            #
            # SGD updating parameters
            w = w - l_rate*w_grad
            b = b - l_rate*b_grad
    return


def infer(X_test, save_dir):
    test_data_size = len(X_test)

    # load parameters
    print "====loading param===="
    w = np.loadtxt(os.path.join(save_dir, 'w'))
    b = np.loadtxt(os.path.join(save_dir, 'b'))

    # predict
    z = (np.dot(X_test, w) + b)
    y = sigmoid(z)
    y_ = np.around(y)

    with open('answer', 'w') as f:
        f.write('id, label\n')
        for i, v in enumerate(y_):
            f.write("%d,%d\n" % (i+1, v))
        f.close()


if __name__ == '__main__':
    train_data_path = 'X_train'
    train_label_path = 'Y_train'
    test_data_path = 'X_test'
    X_all, Y_all, X_test = load_data(train_data_path, train_label_path, test_data_path)
    X_all, X_test = normalization(X_all, X_test)
    train(X_all, Y_all, 'logistic_params/')
    infer(X_test,'logistic_params/')


