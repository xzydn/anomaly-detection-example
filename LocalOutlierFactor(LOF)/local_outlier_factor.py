# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import LocalOutlierFactor


class LOF:
    """
    局部离群因子类
    """

    def __init__(self, X):
        self.X = X
        self.D_K = []  # X中每个样本点的k距离
        self.NEIGHBORS = []  # X中每个样本点的k距离邻域
        self.LRD = []  # X中每个样本点的局部可达密度
        self.k = None
        self.LOF = []  # 局部异常因子
        self.dist = cdist(X, X, metric='euclidean')  # 计算任意两点之间距离
        print("X shape:", np.shape(X))

    def calc_lof(self, k):
        # 计算LOF
        X = self.X
        self.lof_init(k)
        self.calc_lrd()
        for i in range(0, len(X)):  # 计算每个样本点的局部离群因子
            self.LOF[i] = self.lof(i)

    def lof_init(self, k):
        # 计算LOF初始化
        self.k = k
        X = self.X
        self.D_K = np.zeros(len(X))  # X中每个样本点的k距离
        self.NEIGHBORS = []  # X中每个样本点的k距离邻域
        self.LRD = np.zeros(len(X))  # X中每个样本点的局部可达密度
        self.LOF = np.zeros(len(X))  # 局部异常因子

    def calc_lrd(self):
        # 计算局部可达密度
        k = self.k
        X = self.X
        for i in range(0, len(X)):
            # print(i)
            # x_i距所有样本点的欧几里得距离
            x_and_d = self.dist[i, :]
            index_x_and_d = np.argsort(x_and_d)  # 将特征值按从小到大排序，index保留的是对应原矩阵中的下标
            self.D_K[i] = x_and_d[index_x_and_d[k]]  # x_i的k距离，此处假设x_and_d中没有重复的距离，除去自身点
            # self.NEIGHBORS.append(index_x_and_d[1:k + 1])  # x_i的k距离邻域，除去自身点（这里有问题，至少有k个）
            neighbors = np.argwhere(x_and_d <= self.D_K[i])
            self.NEIGHBORS.append([x[0] for x in neighbors if x[0] != i]) # x_i的k距离邻域，除去自身点
        for i in range(0, len(X)):  # 计算每个样本点的局部可达密度
            self.LRD[i] = self.lrd(i)

    def lrd(self, p):
        # p的局部可达密度
        neighbors = self.NEIGHBORS[p]
        rd = np.maximum(self.D_K[neighbors], self.dist[p, neighbors])
        return 1 / (np.sum(rd) / len(neighbors))

    def lof(self, p):
        # p的局部离群因子
        neighbors = self.NEIGHBORS[p]
        return np.sum(self.LRD[neighbors]) / self.LRD[p] / len(neighbors)


def create_train():
    np.random.seed(42)  # 设置seed使每次生成的随机数都相等
    # 生成100个2维数据，它们是以0为均值、以1为标准差的正态分布
    X_inliers = 0.3 * np.random.randn(100, 2)
    # 构造两组间隔一定距离的样本点作为训练数据
    X_inliers = np.r_[X_inliers + 2, X_inliers - 2]
    # 构造20个可能的异常数据，从一个均匀分布[low,high)中随机采样
    X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
    # 将X_inliers和X_outliers连接起来作为训练集
    return np.r_[X_inliers, X_outliers]


def lof_sklearn(X, k):
    clf = LocalOutlierFactor(n_neighbors=k)
    clf.fit_predict(X)
    LOF = -clf.negative_outlier_factor_
    return LOF


if __name__ == '__main__':
    X = create_train()
    k = 20

    lof = LOF(X)
    lof.calc_lof(k)
    # print(lof.LRD.tolist())
    print("LOF Mine:\n", lof.LOF)

    LOF_skleran = lof_sklearn(X, k)
    print("LOF sklearn:\n", LOF_skleran)
