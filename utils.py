#!/usr/bin/python
# encoding: utf-8

#!/usr/bin/python
# encoding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
import collections
from tqdm import tqdm
import numpy as np 
import cv2
import os
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (list): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet):
        alphabet.remove('基线')
        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by ctc
            self.dict[char] = i + 1
        self.dict['基线'] = 0
        self.dict_reverse = {value: key for key, value in self.dict.items()}


    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """

        length = []
        result = []
        for char in text:
            length.append(1)
            index = self.dict[char]
            result.append(index)
        text = result
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length=1, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        t = t.numpy()
        result = []
        for value in t:
            index = self.dict_reverse[value]
            result.append(index)
        texts = result

        return texts


class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res



def props_to_onehot(props):
    if isinstance(props, list):
        props = np.array(props)
    a = np.argmax(props, axis=1)
    b = np.zeros((len(a), props.shape[1]))
    b[np.arange(len(a)), a] = 1
    return b


def onehot_to_num(onehot):
    if isinstance(onehot, list):
        onehot = np.array(onehot)
    b = np.zeros((onehot.shape[0], 1))
    for i, h in enumerate(onehot):
        b[i, 0] = np.argwhere(onehot[i] == 1)
    return b


def draw(preds, x_train_batch,x_label,ax):
    predsnp = preds.cpu().detach().numpy()

    x_train_batchnp = x_train_batch.cpu().detach().numpy()
    x_label = x_label.cpu().detach().numpy()
    # print(predsnp.shape, x_train_batchnp.shape)  # (2000,6)
    predsnp = props_to_onehot(predsnp)
    # print(predsnp)
    predsnp = onehot_to_num(predsnp)
    # print(max(predsnp))


    #对原数据进行kmeans分类

    estimator = KMeans(n_clusters=2)  # 构造聚类器
    estimator.fit(x_train_batchnp)  # 聚类
    label_pred = estimator.labels_  # 获取聚类标签
    # 绘制k-means结果
    if label_pred[0]==1:
        label_pred = 1-label_pred

    # plt.plot(np.argwhere(label_pred == 0), np.zeros(len(np.argwhere(label_pred == 0)))*x_label,'go-')
    # plt.plot(np.argwhere(label_pred == 1), np.ones(len(np.argwhere(label_pred == 1))) * x_label,'go-')
    ax.scatter(np.argwhere(label_pred == 0), np.zeros(len(np.argwhere(label_pred == 0)))*x_label, c="green", marker='o',s = 10, label='kmeans')
    ax.scatter(np.argwhere(label_pred == 1), np.ones(len(np.argwhere(label_pred == 1)))*x_label, c="green", marker='o',s = 10, label='kmeans')


    for i in range(int(max(predsnp))+1):
        x= np.argwhere(predsnp == i)[:,0]
        y = np.ones(len(x))*i
        # plt.plot(x, y, c = "red")
        ax.scatter(x, y, c = "red", marker='.', label='pred',s = 5)

