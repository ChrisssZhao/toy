# -*- coding: utf-8 -*-
# Copyright (c) 2019 ke.com, Inc. All Rights Reserved
"""
 Module Summmary: 使用tensorflow实现简单的pca 
 Author: zhaoqun002
 Mail: zhaoqun002@ke.com
 DATE: 19/03/24
"""

import tensorflow as tf
tf.enable_eager_execution() #动态图
import numpy as np
from sklearn import datasets




def pca(x,dim= 2):
    """
    利用pca算法降维
    输入：
        x 原始特征矩阵
        dim 降维后的维度
    输出：
        降维后的矩阵
    """
    with tf.name_scope("PCA"):
        #获取x的shape
        m,n= tf.to_float(x.get_shape()[0]),tf.to_int32(x.get_shape()[1])
        #标准化
        mean = tf.reduce_mean(x,axis=1)
        x_standard = x - tf.reshape(mean,(-1,1))
        #求协方差矩阵
        cov = tf.matmul(x_standard, x_standard, transpose_a=True)/(m - 1)
        e,v = tf.linalg.eigh(cov,name="eigh") #特征值和特征向量
        #按特征值大小取特征向量
        e_index_sort = tf.math.top_k(e,sorted=True,k=dim)[1] 
        v_new = tf.gather(v,indices=e_index_sort)
        #矩阵相乘得到降维后的矩阵
        res = tf.matmul(x_standard,v_new,transpose_b=True)
    return res



if __name__ == "__main__":
    print "new day, new life"
    data = datasets.load_iris(return_X_y=False) #iris数据
    data_preprocess = tf.constant(np.reshape(data.data,(data.data.shape[0],-1)),dtype=tf.float32)
    pca_data = pca(data_preprocess, dim=2)

