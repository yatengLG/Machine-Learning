# -*- coding: utf-8 -*-
# @Author  : LG

import numpy as np

# 欧氏距离
def euclidean_distance(x, y ):
    """
    d = sqrt( (x1-y1)^2 + (x2-y2)^2 +... )
    :param x:   [x1, x2, x3, ...]
    :param y:   [[y1, y2, y3, ...],[], [] ,...]
    :return: 欧式距离
    """
    return np.sqrt(np.power((y-x), 2).sum(axis=1))

# 曼哈顿距离(城市街区距离)
def city_block_distance(x, y):
    """
    d = |x1-y1| + |x2-y2| + ...
    :param x:   [x1, x2, x3, ...]
    :param y:   [y1, y2, y3, ...]
    :return: 城市街区距离
    """
    return np.abs(y-x).sum()

# 切比雪夫距离
def chebyshev_distance(x, y):
    """
    d = max(|x1-y1|, |x2-y2|, ...)
    :param x:   [x1, x2, x3, ...]
    :param y:   [y1, y2, y3, ...]
    :return:
    """
    return np.abs(y-x).max()
