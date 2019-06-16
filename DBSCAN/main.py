# -*- coding: utf-8 -*-
# Copyright (c) 2019 ke.com, Inc. All Rights Reserved
"""
 Module Summmary: 实现简单的DBSCAN算法 
 Author: zhaoqun002
 Mail: zhaoqun002@ke.com
 DATE: 19/06/16
"""

import numpy as np
import math
import evaluate as eva
import matplotlib.pyplot as plt
import queue

data_path = "./data/test.data" #小批量聚类数据集
NOISE = 0  
UNASSIGNED = -1 #缺失值补充-1

def load_data():
    """
    读取数据
    """
    points = np.loadtxt(data_path, delimiter='\t')
    return points

def dist(a, b):
    """
    计算两个点的平方差距离
    """
    return math.sqrt(np.power(a-b, 2).sum())

def neighbor_points(data, pointId, radius):
    """
    得到邻域内所有样本点的Id
    输入：
        data 样本点
        pointId 核心点
        radius 半径
    输出: 
        邻域内所用样本Id
    """
    points = []
    for i in range(len(data)):
        if dist(data[i, 0: 2], data[pointId, 0: 2]) < radius:
            points.append(i)
    return np.asarray(points)

def to_cluster(data, clusterRes, pointId, clusterId, radius, minPts):
    """
    判断一个点是否是核心点
    输入:
        clusterRes 聚类结果
        pointId  样本Id
        clusterId 类Id
        radius 半径
        minPts 最小局部密度
    输出:  
        返回是否能将点PointId分配给一个类
    """
    points = neighbor_points(data, pointId, radius)
    points = points.tolist()
    q = queue.Queue()
    if len(points) < minPts:
        clusterRes[pointId] = NOISE
        return False
    else:
        clusterRes[pointId] = clusterId
    for point in points:
        if clusterRes[point] == UNASSIGNED:
            q.put(point)
            clusterRes[point] = clusterId

    while not q.empty():
        neighborRes = neighbor_points(data, q.get(), radius)
        if len(neighborRes) >= minPts:  # 核心点
            for i in range(len(neighborRes)):
                resultPoint = neighborRes[i]
                if clusterRes[resultPoint] == UNASSIGNED:
                    q.put(resultPoint)
                    clusterRes[resultPoint] = clusterId
                elif clusterRes[clusterId] == NOISE:
                    clusterRes[resultPoint] = clusterId
    return True

def dbscan(data, radius, minPts):
    """
    扫描整个数据集，为每个数据集打上核心点，边界点和噪声点标签的同时为样本集聚类
    输入：
        data 样本集
        radius 半径
        minPts 最小局部密度
    输出:
        返回聚类结果和clusterId(类id集合)
    """
    clusterId = 1
    nPoints = len(data)
    clusterRes = [UNASSIGNED] * nPoints
    for pointId in range(nPoints):
        if clusterRes[pointId] == UNASSIGNED:
            if to_cluster(data, clusterRes, pointId, clusterId, radius, minPts):
                clusterId = clusterId + 1
    return np.asarray(clusterRes), clusterId

def plotRes(data, clusterRes, clusterNum):
    """
    画图，展示聚类效果
    """
    nPoints = len(data)
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
    for i in range(clusterNum):
        color = scatterColors[i % len(scatterColors)]
        x1 = [] 
        y1 = []
        for j in range(nPoints):
            if clusterRes[j] == i:
                x1.append(data[j, 0])
                y1.append(data[j, 1])
        plt.scatter(x1, y1, c=color, alpha=1, marker='+')

if __name__ == '__main__':
    print "Heros(we could be)"
    data = load_data()
    cluster = np.asarray(data[:, 2])
    clusterRes, clusterNum = dbscan(data, 0.8, 3)
    plotRes(data, clusterRes, clusterNum)
    nmi, acc, purity = eva.eva(clusterRes, cluster)
    print(nmi, acc, purity)
    plt.show()


