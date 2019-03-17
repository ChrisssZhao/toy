# -*- coding: utf-8 -*-
"""
 Module Summmary: 协同过滤 item-based 
 Author: zhaoqun002
 Mail: zhaoqun002@ke.com
 DATE: 19/03/16
"""

import random
import math
from collections import defaultdict

class itemCF(object):
    """
    模型的calss，（养成封装的好习惯）
    重要方法包括：
        getDataset 从文件中获取数据集
        calSim 计算电影之间的相似度
        evaluate 评估模型
        recommend 给用户推荐电影
    """
    
    def __init__(self):
       """
       初始化参数
       """
       self.K = 5
       self.N = 10 #找出前N个推荐K个，N中可能包含user已经看过的电影
       self.train_set = {} #数据集
       self.test_set = {}
       self.movie_sim_mat = {} #相似度矩阵
       self.movie_popular = {}
       self.movie_count = 0
       print "每个人用户推荐 ", self.K, " 个电影"


    def getDataset(self, file_name, pivot = 0.8):
        """
        解析数据获取训练集和测试集
        输入：file_name 原始文件的位置
              pivot 随机数小于于pivot放入训练集，大于放在测试集
        输出：给 train_set和test_set赋值
        """
        for line in self.loadFile(file_name):
            user, movie, rating, _ = line.split('::')
            if random.random() < pivot: #随机划分打乱位置
                self.train_set.setdefault(user, {})
                self.train_set[user][movie] = int(rating)
            else:
                self.test_set.setdefault(user, {})
                self.test_set[user][movie] = int(rating)
        print "数据集划分完成！"

    def loadFile(self, file_name):
        """
        按行解析文件
        输入：文件位置
        输出：按行返回处理
        """
        with open(file_name, 'r') as f:
            for i, line in enumerate(f):
                yield line.strip('\r\n')
       
    def calSim(self ):
        """
        计算相识度
        输入：初始化的超参数和划分好的数据集
        输出：相似矩阵 movie_sim_mat
        """
        #计数的方式统计电影的受欢迎度
        for user, movies in self.train_set.items():
            for movie in movies:
                if movie not in self.movie_popular:
                    self.movie_popular[movie] = 0
                self.movie_popular[movie] += 1
        self.movie_count = len(self.movie_popular)
        #计算相似矩阵
        for user, movies in self.train_set.items(): #暴力搜索效率低，这地方可以优化
            for m1 in movies:
                self.movie_sim_mat.setdefault(m1, defaultdict(int))
                for m2 in movies:
                    if m1 == m2:
                        continue
                    self.movie_sim_mat[m1][m2] += 1
        
        for m1, related_movies in self.movie_sim_mat.items():
            for m2, count in related_movies.items():
                self.movie_sim_mat[m1][m2] = count / math.sqrt(
                    self.movie_popular[m1] * self.movie_popular[m2])
    
    def recommend(self, user):
        """
        给用户推荐电影,推荐k个
        输入：用户id
        输出：推荐的电影
        """
        rank = {}
        watched_movies = self.train_set[user] #用户的历史浏览
        
        for movie, rating in watched_movies.items():
            for related_movie, similarity_factor in sorted(self.movie_sim_mat[movie].items(),
                    key=lambda x:x[1], reverse=True)[:self.N]:
                if len(rank) >= self.K: #只需要推荐K个
                    return sorted(rank.items(), key=lambda x:x[1], reverse=True) 
                if related_movie in watched_movies:#过滤掉已经推荐的
                    continue
                rank.setdefault(related_movie, 0)
                rank[related_movie] += similarity_factor * rating
        return sorted(rank.items(), key=lambda x:x[1], reverse=True) 


    def evaluate(self):
        """
        简单的通过precision, recall评价模型
        利用命中数进行统计
        """
        hit = 0 
        rec_count = 0 #推荐的个数
        test_count = 0
        all_rec_movies = set()
        for i, user in enumerate(self.train_set):
            test_movies = self.test_set.get(user, {})
            rec_movies = self.recommend(user)
            for movie, _ in rec_movies:
                if movie in test_movies:
                    hit += 1
            rec_count += len(rec_movies)
            test_count += len(test_movies)
        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        print "top ", self.K
        print "precision: ", precision
        print "recall: ", recall

if __name__ == "__main__":
    print "good good study, day day up"
    ic = itemCF()
    ic.getDataset('./data/ratings.dat')
    ic.calSim()
    ic.evaluate()
    print "推荐user3223: ",
    print ic.recommend('3223')


