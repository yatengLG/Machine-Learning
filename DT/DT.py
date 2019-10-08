# -*- coding: utf-8 -*-
# @Author  : LG

from collections import Counter
import numpy as np

class decision_tree(object):
    def __init__(self, samples):
        self.samples = samples
        self.data = samples[1:,:-1]
        self.labels = samples[1:,-1]
        self.feature_names = samples[0,:-1]
        conditional_entropy = self.cal_conditional_entropy(self.data, self.labels)
        self.sorted_index = sorted(range(len(conditional_entropy)), key=lambda k: conditional_entropy[k])

        self.sorted_index.reverse()
        self.tree = self.build_tree(self.samples[1:], self.feature_names)   # 建树

    def __call__(self, features):
        if not isinstance(features, np.ndarray):
            features = np.array(features)
        x = dict(zip(features[0], features[1]))
        tree = self.tree
        while True:
            if isinstance(tree,dict):
                dict_keys = list(x.keys())
                tree_key = list(tree.keys())[0]
                if tree_key in dict_keys:
                    try:
                        tree = tree[tree_key][x[tree_key]]
                    except:
                        return tree
                else:
                    break
            else:
                return tree
        return tree

    def build_tree(self, data, feature_names):
        # 如果样本只剩一个,直接返回样本的标签
        if data.shape[0]==1:
            return data[:,-1].tolist()
        # 如果样本数量特征小于一个,直接返回
        elif data.shape[1]<2:
            return data[:,-1].tolist()
        # 如果节点下的所有标签相同,则直接返回;这样大大的简化了树的结构,可以把这部分注释掉看结果
        elif list(data[:,-1]).count(data[:,-1][0]) == len(data):
            return data[:,-1][0]
        else:
            # ** 计算信息增益时,由于不同节点下,数据不同,所以不同节点下存在特征优先级不同的情况是正常的.**
            conditional_entropy = self.cal_conditional_entropy(data[:,:-1], data[:,-1]) # 计算数据的信息增益,确定最重要的特征.
            self.sorted_index = sorted(range(len(conditional_entropy)), key=lambda k: conditional_entropy[k])
            self.sorted_index.reverse()
            best_feature_index = self.sorted_index[0]
            types = set(data[:, best_feature_index])
            node = feature_names[best_feature_index]
            tree = {node:{}}
            for type in types:
                data1 = data[data[:,best_feature_index]==type]
                data1 = np.delete(data1, best_feature_index, 1) # 删除已经用过的特征列
                feature_names1 = np.delete(feature_names, best_feature_index,0) # 删除已经用过的特征名
                tree[node][type] = self.build_tree(data1, feature_names1)   # 递归建树
            return tree

    def cal_shannon_entropy(self, labels):
        """
        计算香农熵
        :param labels:  需计算的标签, size: [n] n样本数
        :return:    经验熵,香农熵
        """
        num_samples = len(labels)
        counts = Counter(labels)
        entropy = 0
        for count in counts.values():
            entropy -= (count/num_samples)*np.log2(count/num_samples)
        return entropy

    def cal_conditional_entropy(self, samples, labels):
        """
        计算每列特征的信息增益(条件熵)
        :param samples: 样本数据, size: [n,m] n样本数,m特征数
        :param labels:  标签      size: [n]
        :return: 特征的条件熵,也就是对应的信息增益, size: [m]
        """
        conditional_entropys = []
        num_samples = len(samples)
        entropy = self.cal_shannon_entropy(labels)
        features = samples.T
        for feature in features:
            counts = Counter(feature)
            # print(feature)
            # print(counts)
            # print(num_samples)
            conditional_entropy = entropy
            for count,num in zip(counts.keys(), counts.values()):
                p = num/num_samples
                # print('num:',num)
                # print('num_samples:', num_samples)
                mask = [feature==count]
                # print('labels: ',labels[tuple(mask)])
                # print("p: ",p)
                # print('entropy = ',self.cal_shannon_entropy(labels[tuple(mask)]))
                conditional_entropy -= p*self.cal_shannon_entropy(labels[tuple(mask)])
            # print('--------'*10)
            conditional_entropys.append(conditional_entropy)
        return conditional_entropys

if __name__ == '__main__':

    samples = np.array([['年龄', '有无工作', '有无房', '信誉', '是否给予贷款'],
                         ['青年', '无', '否', '一般', '否'],
                         ['青年', '无', '否', '好', '否'],
                         ['青年', '有', '否', '好', '是'],
                         ['青年', '有', '是', '一般', '是'],
                         ['青年', '无', '否', '一般', '否'],
                         ['中年', '无', '否', '一般', '否'],
                         ['中年', '无', '否', '好', '否'],
                         ['中年', '有', '是', '好', '是'],
                         ['中年', '无', '是', '非常好', '是'],
                         ['中年', '无', '是', '非常好', '是'],
                         ['老年', '无', '是', '非常好', '是'],
                         ['老年', '无', '是', '好', '是'],
                         ['老年', '有', '否', '好', '是'],
                         ['老年', '有', '否', '非常好', '是'],
                         ['老年', '无', '是', '一般', '否']])

    dt = decision_tree(samples)
    x = [['年龄', '有无工作', '有无房', '信誉'],
         ['中年', '有', '是', '一般']]
    y = dt(x)
    print(y)