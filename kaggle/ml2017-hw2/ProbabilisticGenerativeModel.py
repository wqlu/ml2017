# -*- coding: utf-8
# !/usr/bin/env python

import pandas as pd
import numpy as np
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


def valid(X_valid, Y_valid, mu1, mu2, shared_sigma, N1, N2):

    sigma_inverse = np.linalg.det(shared_sigma)
    print sigma_inverse
    w = np.dot((mu1 - mu2), sigma_inverse)
    x = X_valid.T
    b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(
        float(N1) / N2)
    a = np.dot(w, x) + b
    y = sigmoid(a)
    y_ = np.around(y)
    result = (np.squeeze(Y_valid) == y_)
    print "Validation acc = %f" % (float(result.sum()) / result.shape[0])
    return


def train(X_all, Y_all, save_dir):
    """training ..."""
    valid_set_percentage = 0.1
    X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, valid_set_percentage)

    # Gaussian distribution parameter
    train_data_size = X_train.shape[0]
    cnt1 = 0
    cnt2 = 0

    mu1 = np.zeros((106,))
    mu2 = np.zeros((106,))
    for i in range(train_data_size):
        if Y_train[i] == 1:
            mu1 += X_train[i]
            cnt1 += 1
        else:
            mu2 += X_train[i]
            cnt2 += 1

    sigma1 = np.zeros((106, 106))
    sigma2 = np.zeros((106, 106))
    for i in range(train_data_size):
        if Y_train[i] == 1:
            sigma1 += np.dot([X_train[i]-mu1], np.transpose([X_train[i]-mu1]))
        else:
            sigma2 += np.dot([X_train[i]-mu2], np.transpose([X_train[i]-mu2]))
    sigma1 /= cnt1
    sigma2 /= cnt2
    shared_sigma = ((float(cnt1)/train_data_size)*sigma1 + (float(cnt2)/train_data_size)*sigma2)
    N1 = cnt1
    N2 = cnt2

    print "====saving param===="
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    param_dict = {'mu1':mu1, 'mu2':mu2, 'shared_sigma':shared_sigma, 'N1':[N1], 'N2':[N2]}
    for key in sorted(param_dict):
        print 'saving %s' % key
        np.savetxt(os.path.join(save_dir, ('%s' % key)), param_dict[key])

    print "====validating===="
    valid(X_valid, Y_valid, mu1, mu2, shared_sigma, N1, N2)
    return


if __name__ == '__main__':

    train_data_path = 'X_train'
    train_label_path = 'Y_train'
    test_data_path = 'X_test'
    X_all, Y_all, X_test = load_data(train_data_path, train_label_path,test_data_path)
    # 归一化处理
    X_all, X_test = normalization(X_all, X_test)
    train(X_all, Y_all, 'generative_params/')
