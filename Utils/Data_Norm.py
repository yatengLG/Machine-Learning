# -*- coding: utf-8 -*-
# @Author  : LG

class data_normer(object):
    """
    数据集正则化
    """
    def __init__(self, samples):
        self.means = samples.mean(axis=0)
        self.std = samples.std(axis=0)

    def __call__(self, datas):

        datas = (datas-self.means)/self.std
        return datas
