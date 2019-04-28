# -*- coding: utf-8 -*-
# Copyright (c) 2019 ke.com, Inc. All Rights Reserved
"""
 Module Summmary: 利用numpy实现简单的kd-tree, 只可以找到最近的点 
 Author: zhaoqun002
 Mail: zhaoqun002@ke.com
 DATE: 19/04/27
"""

import numpy as np
from operator import itemgetter
from collections import namedtuple
from pprint import pformat


class Node(namedtuple('Node', 'loc left_child right_child')):
    """
    kd-tree中基本结点，主要包括
        loc 点处于多维空间的位置 array
        left_child 左孩子 Node
        right_child 右孩子 Node
    """
    def __repr__(self):
        return pformat(tuple(self))

class KDtree():
    def __init__(self, points):
        self.tree = self._build_kdtree(points)
        if len(points) > 0:
            self.dim = len(points[0])
        else:
            self.dim = None

    def _build_kdtree(self, points, depth = 0):
        """
        递归构建kd树
        输入:
            points kdnode组成的list
        """
        #空值处理
        if not points:
            return None
        dim = len(points[0])
        axis = depth % dim #在哪个维度上进行区间划分，原始的kd-tree算法需要在最多的方差的维度划分，这里简单的依次从各维度划分
        points.sort(key = itemgetter(axis))
        mid = len(points) / 2 
        return Node(loc=points[mid], left_child=self._build_kdtree(points[:mid], depth+1), right_child=self._build_kdtree(points[mid+1:], depth+1))

    def find_nearest(self, point, root=None, axis=0):
        """
        找到kdtree中点某个点最近的点
        输入:
            point 需要search的点 array
            root kdtree
            axis 从哪个维度开始分割
        输出：
            kdtree的Node
        """
        #空值处理
        if root is None:
            root = self.tree
            self._best = None
        #递归搜索
        if root.left_child or root.right_child:
            new_axis = (axis + 1) % self.dim
            if point[axis] < root.loc[axis] and root.left_child:
                self.find_nearest(point, root.left_child, new_axis)
            elif root.right_child:
                self.find_nearest(point, root.right_child, new_axis)
        #回溯更新_best
        dist_func=lambda x, y: np.linalg.norm(x - y)
        dist = dist_func(root.loc, point)
        if self._best is None or dist < self._best[0]:
            self._best = (dist, root.loc)
        #超球面与一个矩形空间相交，正面该空间存在更近的点
        if abs(point[axis] - root.loc[axis]) < self._best[0]:
            new_axis = (axis + 1) % self.dim
            if root.left_child and point[axis] >= root.loc[axis]:
                self.find_nearest(point, root.left_child, new_axis)
            elif root.right_child and point[axis] < root.loc[axis]:
                self.find_nearest(point, root.right_child, new_axis)
        return self._best


if __name__ == "__main__":
    print "All We Know - The Chainsmokers / Phoebe Ryan"
    #二维
    points = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]
    tree = KDtree(points)
    target = np.array([5, 5])
    print tree.find_nearest(target)
    #三维
    points = [(2, 3, 3), (5, 4, 4), (9, 6, 7), (4, 7, 7), (8, 1, 1), (7, 2, 2)]
    tree = KDtree(points)
    target = np.array([5, 5, 5])
    print tree.find_nearest(target)
