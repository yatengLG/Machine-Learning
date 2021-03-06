# 机器学习算法python实现
# Machine-Learning
基于python3以及numpy.部分数据展示使用了pandas.

实现代码均有详细中文注释.

每个算法均有两个例子.(jupyter笔记展示)
🙈🙈🙈🙈
## 目录
1.  KNN(k-近邻算法)
    
    K近邻算法,采用测量不同特征值之间的距离进行分类.(本项目以欧氏距离作为距离计算方法)
   
    理论: 
    存在一个样本数据集合，也称作训练样本集,并且样本集每个数据存在标签.即我们知道样本
    集每一数据与所属分类的对应关系.输入没有标签的新数据后,将新数据的每个特征与样本
    集中数据对应的特征进行比较,然后算法提取样本集中特征最相似数据(最近邻)的分类标签.
   
    [实现代码](KNN/KNN.py)  [使用例子](KNN.ipynb) 
   
2.  DT(决策树)
   
    决策树算法根据数据的属性采用树状结构建立决策模型
    
    本项目决策树以信息增益作为评估特征重要性的手段,进行决策树搭建.
    
    [实现代码](KNN/KNN.py) [使用例子](DT.ipynb)
   
3.  NBM(朴素贝叶斯模型)
    
    朴素贝叶斯算法是基于贝叶斯定理的一类算法.
    
    朴素贝叶斯模型假定"特征条件独立",通过计算先验概率以及不同特征条件概率,计算样本对属于不同类别的概率,最终取概率最大的作为最终预测值.
    避免了难以从有限的训练样本中直接估计所有属性上的联合概率.

    [实现代码](KNN/KNN.py) [使用例子](NBM.ipynb)

4. 待完成

    线性回归 
    逻辑回归 
    凸优化 
    支持向量机 
    随机森林 
    GBDT 
    XGBoost 
    矩阵分解 
    K-Means 
    GMM 
    主题模型 
    EM 
    聚类 
    PCA 
    
    不定时更新