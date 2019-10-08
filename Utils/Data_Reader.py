# -*- coding: utf-8 -*-
# @Author  : LG

import csv
import numpy as np

__all__ = ['data_reader']

class data_reader(object):
    def __init__(self, csv_file, shuffer=True, train_rate=0.8):
        assert csv_file.endswith('.csv')
        self.shuffer = shuffer

        with open(csv_file) as csv_f:
            data_file = csv.reader(csv_f)
            # 第一行: 样本数, 特征数, 类别数(如果是1,则为回归), 类别名.., 特征名...
            temp = next(data_file)
            self.n_samples = int(temp[0])
            self.n_features = int(temp[1])
            self.n_labels = int(temp[2])
            self.label_names = temp[3:3+self.n_labels]
            self.feature_names = temp[3+self.n_labels: 3+self.n_labels+self.n_features]
            self.features_list = []
            self.labels_list = []
            for i, data in enumerate(data_file):
                self.features_list.append(np.asarray(data[:-1], dtype=np.float64))
                if self.n_labels ==1:
                    # 类别为1 时,为回归问题
                    self.labels_list.append(np.asarray(data[-1], dtype=np.float64))
                else:
                    # 分类问题
                    self.labels_list.append(np.asarray(data[-1], dtype=np.int))
        print(" dataset {} has sample :{} , features :{} , label :{}".format(csv_file, self.n_samples, self.n_features, self.n_labels))
        print(" train sample : {} , test sample :{} ".format(int(self.n_samples*train_rate),self.n_samples-int(self.n_samples*train_rate)))
        if self.shuffer:    # 打乱数据
            import random
            data_list = list(zip(self.features_list, self.labels_list))
            random.shuffle(data_list)
            self.features_list, self.labels_list = zip(*data_list)

        # 划分数据集
        self.train_features_list = self.features_list[:int(self.n_samples * train_rate)]
        self.test_features_list = self.features_list[int(self.n_samples * train_rate):]
        self.train_labels_list = self.labels_list[:int(self.n_samples * train_rate)]
        self.test_labels_list = self.labels_list[int(self.n_samples * train_rate):]


    def __getitem__(self, index):
        # 按照索引依次返回
        return self.features_list[index], self.labels_list[index]

    def __len__(self):
        return self.n_samples

    def get_all(self):
        # 矩阵形式返回所有数据
        return np.array(self.train_features_list), np.array(self.train_labels_list),\
               np.array(self.test_features_list), np.array(self.test_labels_list)

if __name__ == '__main__':
    # demo  数据读取器
    iris_reader = data_reader('/home/super/PycharmProjects/Machine-Learning/KNN/iris.csv')
    # 特征名称, 标签名称
    features_name = iris_reader.feature_names
    labels_name = iris_reader.label_names
    print(features_name)
    print(labels_name)

    #  取出全部数据
    train_features, train_labels, test_features, test_labels = iris_reader.get_all()
    print(train_features.shape)
    print(train_labels.shape)
    print(test_features.shape)
    print(test_labels.shape)

    print('---' * 20)

    # 从所有数据中迭代 取出数据
    for i, (feature, label) in enumerate(iris_reader):
        print(i)
        print('feature:',feature,end=' ')
        print('label:',label)
        if i==3:
            break
    print('---'*20)

    # 取出指定的某一条数据, 例如第6条(索引从0开始计算)
    feature, label = iris_reader.__getitem__(5)
    print('feature:',feature, 'label:',label)