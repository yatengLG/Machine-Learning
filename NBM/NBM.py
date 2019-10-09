# -*- coding: utf-8 -*-
# @Author  : LG
import numpy as np

class naive_bayesian_model(object):
    def __init__(self,samples):
        self.samples = samples
        self.prior_P, self.condition_P, self.series_P = self.cal_dispersed_P(samples)

    def __call__(self, sample):
        labels = list(self.prior_P.keys())
        titles = sample[0]
        features = sample[1]
        P_dict = {}
        for label in labels:
            P_dict[label] = self.prior_P[label] # 对应label的先验概率
            for title, feature in zip(titles, features):
                if self.is_dispersed(feature):
                    # 对于离散特征,直接从离散特征条件概率字典中读取对应值.
                    P_dict[label] *= self.condition_P[label][title][feature]
                else:
                    # 对于连续特征,这里使用概率密度函数计算
                    P_dict[label] *= (np.exp(-np.power((self.series_P[label][title]['mean']-feature),2)/
                                             np.power(self.series_P[label][title]['std'],2))/
                                      (np.sqrt(2*np.pi)*self.series_P[label][title]['std']))

        # 取最大概率对应的标签为最终预测
        for key, value in P_dict.items():
            print("{} 的概率为 : {:.4f}".format(key, value))
            if value==max(P_dict.values()):
                P = key
        return P
    # 分别计算先验概率和条件概率,并存入字典
    def cal_dispersed_P(self,samples):
        num_feature = len(samples[0])-1
        num_sample = len(samples)-1
        labels = set(samples[1:,-1])
        prior_P = {}    # 用于存储各类别的先验概率
        condition_P = {}    # 用于存储离散特征的条件概率
        series_P = {}       # 用于存储连续特征的均值与方差
        for label in labels:
            count = sum(samples[1:,-1]==label)
            prior_P[label] = count/num_sample
            condition_P[label]={}
            series_P[label] = {}
            for i in range(num_feature):
                features = samples[:,i]
                title = features[0]

                types = list(set(features[1:]))
                features = samples[samples[:,-1]==label]
                if self.is_dispersed(types):
                    condition_P[label][title] = {}
                    for type in types:
                        condition_P[label][title][type] = sum(features[:,i]==type)/count
                else:
                    series_P[label][title] = {}
                    datas = features[:,i]
                    if not isinstance(datas,np.ndarray):
                        datas = np.array(datas)
                    series_P[label][title]['mean'] = np.mean(datas.astype(float))
                    series_P[label][title]['std'] = np.std(datas.astype(float))
        return prior_P, condition_P, series_P

    # 判断输入特征是否是离散的
    def is_dispersed(self,datas):
        if not isinstance(datas,np.ndarray):
            datas = np.array(datas)
            # 这里默认可以转化为数字则特征为连续
            try :
                datas.astype(float)
                return False
            except:
                return True



if __name__ == '__main__':

    sample = np.array([['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率', '是否好瓜'],
                       ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, '是'],
                       ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, '是'],
                       ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, '是'],
                       ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, '是'],
                       ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, '是'],
                       ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, '是'],
                       ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, '是'],
                       ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, '是'],
                       ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, '否'],
                       ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, '否'],
                       ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, '否'],
                       ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, '否'],
                       ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, '否'],
                       ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, '否'],
                       ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370, '否'],
                       ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042, '否'],
                       ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, '否'],
                       ])
    nb=naive_bayesian_model(sample)
    x = [['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率'],
        ['浅白', '蜷缩', '浊响', '清晰', '稍凹', '硬滑', 0.403, 0.226]]
    y = nb(x)
    print(y)

