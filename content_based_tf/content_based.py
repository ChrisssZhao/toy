# -*- coding: utf-8 -*-
# Copyright (c) 2019 ke.com, Inc. All Rights Reserved
"""
 Module Summmary: 基于内容的推荐（矩阵） 
 Author: zhaoqun002
 Mail: zhaoqun002@ke.com
 DATE: 19/05/19
"""

import pandas as pd
import numpy as np
import tensorflow as tf

### 加载数据 ###
ratings_df = pd.read_csv('./data/ml-latest-small/ratings.csv')
movies_df = pd.read_csv('./data/ml-latest-small/movies.csv')
movies_df['movieRow'] = movies_df.index
movies_df = movies_df[['movieRow', 'movieId', 'title']] 
ratings_df = pd.merge(ratings_df, movies_df, on='movieId') 
ratings_df = ratings_df[['userId', 'movieRow', 'rating']]

### 预处理（构建矩阵） ###
user_max = ratings_df['userId'].max() + 1
movie_max = ratings_df['movieRow'].max() + 1
rating = np.zeros((movie_max, user_max)) #初始化
ratings_df_length = np.shape(ratings_df)[0]
#按行处理
for index, row in ratings_df.iterrows():
    rating[int(row['movieRow'])][int(row['userId'])] = row['rating']
#标准化
def normalizeRatings(rating, record):
    """输出rating的处理结果"""
    m, n = rating.shape #m表示电影数量，n表示用户数
    rating_mean = np.zeros((m, 1))
    rating_norm = np.zeros((m, n))
    for i in range(m):
        idx = (record[i, :] != 0)
        rating_mean[i] = np.mean(rating[i, idx])
        rating_norm[i, idx] = rating[i, idx] - rating_mean[i]
    return rating_norm, rating_mean
record = np.array(rating > 0, dtype=int)
rating_norm, rating_mean = normalizeRatings(rating, record)
rating_norm = np.nan_to_num(rating_norm) #空值替换成0

### tf构建模型 ###
num_features = 12
#X代表电影特征，theta代表用户喜好
X = tf.Variable(tf.random_normal([movie_max, num_features], stddev = 0.35))
theta = tf.Variable(tf.random_normal([user_max, num_features], stddev = 0.35))
#损失函数
loss = 1/2 * tf.reduce_sum(((tf.matmul(X, theta, transpose_b=True) - rating_norm) * record) ** 2) + \
        0.5*(1/2 * (tf.reduce_sum(X ** 2) + tf.reduce_sum(theta ** 2)))
train = tf.train.AdamOptimizer(1e-3).minimize(loss)

### 训练模型 ###
tf.summary.scalar('train_loss', loss) #summary保存训练信息
info_log = tf.summary.merge_all()
f = tf.summary.FileWriter('./data/ml-latest-small/movie_tensorborad') #保存地址
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(2000):
    _, summary = sess.run([train, info_log])
    f.add_summary(summary, i) #保存结果

### eval ###
movie, favor = sess.run([X, theta])
predicts = np.dot(movie, favor.T) + rating_mean
errors = np.sqrt(np.sum(np.nan_to_num(predicts - rating) **2 / predicts.shape[0]))
print "rmse: ", errors

