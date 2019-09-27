# -*- coding: utf-8 -*-
# @Author  : LG

from Utils import euclidean_distance
from Utils import data_normer
import numpy as np
from collections import Counter


class knn(object):
    def __init__(self, samples, labels, k, norm=True):
        """
        K近邻算法,采用测量不同特征值之间的距离进行分类.
        理论: 存在一个样本数据集合，也称作训练样本集,并且样本集每个数据存在标签.即我们知道样
            本集每一数据与所属分类的对应关系.输入没有标签的新数据后,将新数据的每个特征与样本
            集中数据对应的特征进行比较,然后算法提取样本集中特征最相似数据(最近邻)的分类标签.
        :param samples: 样本
        :param labels:  样本标签
        :param k:       k
        """
        self.samples = samples
        self.labels = labels
        self.k = k
        self.norm = norm
        if self.norm:
            self.data_normer = data_normer(self.samples)
            self.samples = self.data_normer(self.samples)
    def __call__(self, x):
        assert x.ndim == 1
        if self.norm:
            x = self.data_normer(x)
        d = euclidean_distance(x,self.samples)  # 依次计算输入x 与样本中数据的欧氏距离
        index = np.argsort(d)                   # 对距离排序,获取原始距离数据的下标
        labels = self.labels[index]             # 使用下标,获取对应的排序后的label
        labels = labels[:self.k]                # 取前k个排序后的label
        label=Counter(labels)                   # 统计个数
        label = label.most_common(1)[0][0]      # 获取标签label
        return label


if __name__ == '__main__':
    from Utils.Data_Reader import data_reader

    iris_reader = data_reader('wine_data.csv',split_rate=0.8)
    train_features, train_labels, test_features, test_labels = iris_reader.get_all()

    fn = knn(samples=train_features,labels=train_labels,k=5, norm=True)
    unacc = 0
    for test_feature, test_label in zip(test_features,test_labels):

        label = fn(test_feature)
        if label != test_label:
            print(label, '---',test_label)
            unacc+=1

    print("acc : {:.2f}%".format((1-unacc/len(test_labels))*100))
