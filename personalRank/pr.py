# -*- coding: utf-8 -*-
"""
 Module Summmary: personalRank算法，《推荐系统实战》（项亮），矩阵实现
 Author: zhaoqun002
 Mail: zhaoqun002@ke.com
 DATE: 19/04/07
"""

import numpy as np
from numpy.linalg import solve
from scipy.sparse.linalg import gmres,lgmres
from scipy.sparse import csr_matrix
import copy

def personalRank(matrix, vertexes, start, alpha = 0.8):
    """
    使用矩阵实现该算法，利用稀疏矩阵解线性方程速度更快
    输入：
        matrix 邻接矩阵 list
        vertexes 顶点名 list
        start 开始游走的点,vertexes内部的点 str
        alpha 随机游走的概率 float
    输出：
        print 相关信息
    """
    M = np.matrix(matrix)
    #构造start_pos的矩阵表示
    pos =  [[0]]*len(vertexes)  #从哪个点开始该位置的值不为0
    s = np.matrix(pos)
    idx_pos = vertexes.index(start)
    s[idx_pos][0] = 1
    n = len(vertexes)
    
    #method 1
    print "**直接使用矩阵计算**"
    A = np.eye(n) - alpha * M.T
    b = (1 - alpha) * s
    r = solve(A,b)
    #输出结果
    res = {}
    for j in xrange(n):
        res[vertexes[j]] = r[j]
    for ele in sorted(res.items(), key=lambda x:x[1], reverse=True):
        print "%s:%.3f,\t" %(ele[0], ele[1])
    
    #method 2
    print "**使用稀疏矩阵计算**"
    data = list()
    row_ind = list()
    col_ind = list()
    #压缩存储
    for row in xrange(n):
        for col in xrange(n):
            data.append(A[row, col])
            row_ind.append(row)
            col_ind.append(col)
    AA = csr_matrix((data, (row_ind,col_ind)), shape=(n,n))
    r = gmres(AA, b, tol=1e-08, maxiter=1)[0] #利用压缩矩阵解线性方程
    #输出结果
    res = {}
    for j in xrange(n):
        res[vertexes[j]] = r[j]
    for ele in sorted(res.items(), key=lambda x:x[1], reverse=True):
        print "%s:%.3f,\t" %(ele[0], ele[1])


    return 0

if __name__ == "__main__":
    print "Thinking Out Loud - Ed Sheeran "
    m =[ [   0,   0,   0, 0.5,   0, 0.5,    0],
         [   0,   0,   0,0.25,0.25,0.25, 0.25],
         [   0,   0,   0,   0,   0, 0.5,  0.5],
         [ 0.5, 0.5,   0,   0,   0,   0,    0],
         [   0,   1,   0,   0,   0,   0,    0],
         [0.33,0.33,0.33,   0,   0,   0,    0],
         [   0, 0.5, 0.5,   0,   0,   0,    0] ]
    v = ['A','B','C','a','b','c','d'] #和m对应，用大小写区分相识度
    start = 'b' #从这个点出发
    personalRank(m, v, start)
        
