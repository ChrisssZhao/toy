# -*- coding: utf-8 -*-
# Copyright (c) 2019 ke.com, Inc. All Rights Reserved
"""
 Module Summmary: 实现简单的ftrl优化算法 
 Author: zhaoqun002
 Mail: zhaoqun002@ke.com
 DATE: 19/06/23
"""

class LR(object):
    """基分类器为LR"""
    @staticmethod
    def fn(w, x):
        """sigmoid"""
        return 1.0 / (1.0 + np.exp(-w.dot(x)))

    @staticmethod
    def loss(y, y_hat):
        """损失函数为交叉熵"""
        return np.sum(np.nan_to_num(-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)))

    @staticmethod
    def grad(y, y_hat, x):
        """交叉熵的一阶导"""
        return (y_hat - y) * x

class FTRL(object):
    """FTRL优化器"""
    def __init__(self, dim, l1, l2, alpha, beta, decisionFunc=LR):
        self.dim = dim
        self.decisionFunc = decisionFunc
        self.z = np.zeros(dim)
        self.n = np.zeros(dim)
        self.w = np.zeros(dim)
        self.l1 = l1
        self.l2 = l2
        self.alpha = alpha
        self.beta = beta

    def predict(self, x):
        """利用LR预测"""
        return self.decisionFunc.fn(self.w, x)

    def update(self, x, y):
        """计算梯度，更新参数"""
        self.w = np.array([0 if np.abs(self.z[i]) <= self.l1 else (np.sign(
            self.z[i]) * self.l1 - self.z[i]) / (self.l2 + (self.beta + np.sqrt(self.n[i])) / self.alpha) for i in xrange(self.dim)])
        y_hat = self.predict(x)
        g = self.decisionFunc.grad(y, y_hat, x)
        sigma = (np.sqrt(self.n + g * g) - np.sqrt(self.n)) / self.alpha
        self.z += g - sigma * self.w
        self.n += g * g
        return self.decisionFunc.loss(y, y_hat)

    def train(self, trainSet, verbos=False, max_itr=100000000, eta=0.01, epochs=100):
        """迭代训练"""
        itr = 0
        n = 0
        while True:
            for x, y in trainSet:
                loss = self.update(x, y)
                if verbos:
                    print "itr=" + str(n) + "\tloss=" + str(loss)
                #停止条件是loss小于eta，或者达到一定的轮数
                if loss < eta:
                    itr += 1
                else:
                    itr = 0
                if itr >= epochs:
                    return
                n += 1
                if n >= max_itr:
                    print "reach max iteration", max_itr
                    return

class Corpus(object):
    """读取数据集"""
    def __init__(self, file, d):
        self.d = d
        self.file = file

    def __iter__(self):
        with open(self.file, 'r') as f_in:
            for line in f_in:
                arr = line.strip().split()
                if len(arr) >= (self.d + 1):
                    yield (np.array([float(x) for x in arr[0:self.d]]), float(arr[self.d]))

if __name__ == '__main__':
    print "Wake Me Up (TYMA Remix)"
    d = 4
    #训练集是一个4维向量，0\1 标签为最后一列
    corpus = Corpus("train.txt", d)
    ftrl = FTRL(dim=d, l1=1.0, l2=1.0, alpha=0.1, beta=1.0)
    ftrl.train(corpus, verbos=False, max_itr=100000, eta=0.01, epochs=100)
    w = ftrl.w
    print w

    correct = 0
    wrong = 0
    for x, y in corpus:
        y_hat = 1.0 if ftrl.predict(x) > 0.5 else 0.0
        if y == y_hat:
            correct += 1
        else:
            wrong += 1
    print "correct ratio", 1.0 * correct / (correct + wrong)


